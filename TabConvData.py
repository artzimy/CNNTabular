#!/usr/bin/env python
# coding: utf-8

# In[3]:


from fastai.vision import *
from fastai.tabular import *
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import model_selection

from Macros import *

# # TabConvData Representation

# **Combined Item contains MRI image and patient metadata**

# In[ ]:


class TabConvItem(TabularLine):
    
    def __init__(self, conts, classes, col_names, img, pid, orig_age, cats=[]):
        super().__init__(cats, conts, classes, col_names)
        self.img = img
        self.pid = pid
        self.orig_age = orig_age
        self.data.append(self.img.data)
        self.tab_data, self.img_data = self.data[0:1], self.data[2:]
        self.obj = self.data
        
    def apply_tfms(self, tfms, **kwargs):
        self.img = self.img.apply_tfms(tfms, **kwargs)
        self.data[-1] = self.img.data
        return self
    
    def __str__(self):
        res = super().__str__() + f'Image: {self.img}'
        return res
    
    def show(self, ax:plt.Axes=None, title:Optional[str]=None, figsize:tuple=(3,3), hide_axis:bool=True, **kwargs):
        ax = show_image(self.img, ax=ax, hide_axis=hide_axis, figsize=figsize)
        if title is not None:
            title = title.split('/')
            title = [title[0], '\n', title[1], '\n', title[2], ' / ', title[3]]
            title = "".join(title)
            ax.set_title(title)


# **Process tabular data only**

# In[4]:


class TabConvProcessor(TabularProcessor):
    
    def __init__(self, ds, procs=[Normalize]):
        super().__init__(ds, procs=procs)

    def process(self, ds):
        super().process(ds)    
        ds.preprocessed = True 


# **Data list contains images and patients metadata**

# In[ ]:


class TabConvList(TabularList):
    _item_cls = TabConvItem
    _bunch = DataBunch
    _label_cls = CategoryList
    _processor = TabConvProcessor
    
    def __init__(self, items, cont_names, procs, imgs, convert_mode='RGB', cat_names=None, after_open:Callable=None,  **kwargs):
        super().__init__(items, cat_names, cont_names, procs, **kwargs)
        self.convert_mode,self.after_open = convert_mode,after_open
        self.c , self.sizes = 3 , {}
        self.copy_new.extend(['imgs','convert_mode','after_open'])
        self.cols = [] if cat_names == None else cat_names.copy()
        self.imgs = imgs
        if cont_names: self.cols += cont_names.copy()
        if imgs: self.cols += imgs.copy()
        self.preprocessed = False

    def open(self, fn):
        "Open image in `fn`, subclass and overwrite for custom behavior."
        return open_image(fn, convert_mode=self.convert_mode, after_open=self.after_open)

    def get(self, i):
        if not self.preprocessed: 
            return self.inner_df.iloc[i][self.cols] if hasattr(self, 'inner_df') else self.items[i]

        conts = [] if self.conts is None else self.conts[i]
        pid = self.inner_df['Patient'].iloc[i]
        orig_age = self.inner_df[ORIG_AGE].iloc[i]
        fn = self.inner_df[self.imgs[0]].iloc[i]
        res = self.open(os.path.join(self.path,fn+'.png'))
        self.sizes[i] = res.size
        return self._item_cls(conts, self.classes, self.col_names, img=res, pid=pid, orig_age=orig_age)


    @classmethod
    def from_df(cls, df:DataFrame, cont_names:OptStrList=None, procs=None, 
                path=None, suffix:str='', imgs=None, cols:IntsOrStrs=0, folder:PathOrStr=None, **kwargs):
        
        return cls(items=range(len(df)), cont_names=cont_names,
        procs=procs,
        path=path, imgs=imgs, inner_df=df, **kwargs)
        
    def reconstruct(self, t):
        self.inner_df[AGE] = np.round(self.inner_df[AGE], decimals=4)
        results = self.inner_df[AGE].isin([np.round(float(t[0]), decimals=4)])
        rows = list(results[results==True].index)
        orig_age = self.inner_df[ORIG_AGE][rows[0]]
        return TabConvItem(t[0], self.classes, self.col_names, t[1], None, np.round(float(orig_age), decimals=2))

    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(9,10), **kwargs):
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        x = to_data(xs)
        age = [item.orig_age for item in xs]
        img = torch.cat([s[2].unsqueeze(0) for s in x], 0)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            ax.imshow(img[i].permute(1,2,0))
            ax.set_title(''.join([str(ys[i]) , ' ; ', str(age[i])]), fontsize=12)
            ax.axis('off')
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=(9,10), **kwargs):
        "Show the `xs` and `ys` and 'zs' (predictions) on a figure of `figsize`. `kwargs` are passed to the show method."
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        x = to_data(xs)
        age = [item.orig_age for item in xs]
        img = torch.cat([s[2].unsqueeze(0) for s in x], 0)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            ax.imshow(img[i].permute(1,2,0))
            ax.set_title(''.join([str(ys[i]) , ' ; ',
            self.trim(age[i]), '\n', data.classes[int(zs[i])]]), fontsize=12)

            ax.axis('off')
        plt.tight_layout()


# In[ ]:


def my_collate(batch):
    x, y = zip(*batch)
    x = to_data(list(x))
    age = torch.cat([s[1].unsqueeze(0) for s in x], 0)
    img = torch.cat([s[2].unsqueeze(0) for s in x], 0)
    y = to_data(y)
    y = torch.tensor(y)
    return [age, img], y


# # Preprocessing Data

# **Set data frame before learning according to given fold index**

# In[ ]:


def filter_df(df, valid_idx):
    df_data = df.copy(deep=True)
    to_remove = [ind for ind in range(len(df_data)) 
                 if (df_data.iloc[ind][valid_idx] == 1 and df_data.iloc[ind]["Orig"] == 0)]
    df_data = df_data.drop(to_remove)
    df_data.drop(columns=df_data.columns[0],inplace=True)
    return df_data

def create_orig_age(df):
    df[ORIG_AGE] = df[AGE]
    
def set_df(df, valid_idx):
    new_df = filter_df(df, valid_idx)
    create_orig_age(new_df)
    return new_df


# **Split data to train and validation sets - Optional**
# 
# Helper functions alongisde loading the data, stratifying, splitting into test and validation and a sanity check
# 
# Training and validation are split by idx - these should be passed to .split_by_idx in the fastai ImageList constructor

# In[ ]:


def get_class_count(df) -> dict:
    grp = df.groupby([CLASS])[ID].nunique()
    return {key:grp[key] for key in list(grp.keys())}

def get_class_proportions(df) -> dict:
    class_counts = get_class_count(df)
    return {val[0]:round(val[1]/df.shape[0], 4) for val in class_counts.items()}

def print_class_distribution(train_idx, val_idx, df):
    print("Train data class distribution: ", get_class_proportions(df.iloc[train_idx]))
    print("Test data class distribution: ", get_class_proportions(df.iloc[val_idx]))

### create a list of patients with corresponding classes ###    
def get_patient_list(df_data):
    patients = df_data[PATIENT].unique()
    classes = []
    for i in range(len(patients)):
        tmp = df_data[df_data[PATIENT] == patients[i]]
        tmp = tmp[CLASS]
        classes.append(tmp.iloc[0])
    return patients, classes

### split patients to train and validation ###
def split_train_val_patient_proportionate(data:pd.DataFrame, pct=0.2, label='label'):
    patients, classes = get_patient_list(data)
    return model_selection.train_test_split(patients, classes, test_size=pct, stratify=classes)

def get_patient_idx(data, patients):
    out = []
    for p in patients:
        out += data.index[data[PATIENT] == p].tolist()
    return out 
        
### get indices according to patient split ###
def get_split_idx(data,x_train_patients, x_val_patients):
    return get_patient_idx(data, x_train_patients), get_patient_idx(data, x_val_patients)
    
### final split function ###
def split_train_val(data:pd.DataFrame, pct=0.2, label='label'):
    x_train_patients, x_val_patients, y_train_patients, y_val_patients = split_train_val_patient_proportionate(data, label=CLASS)
    return get_split_idx(data, x_train_patients, x_val_patients)


# # Show Data

# **Plot Augmented MetaData histogram**

# In[ ]:


def plot_age_histogram(df):
    for label in df[CLASS].unique():
        plt.hist(df[AGE].loc[df[CLASS] == label],bins=20,label=label,density=True, alpha=0.5)
    plt.legend(loc='upper right')
    plt.title('Age distribution in subject groups')
    plt.xlabel('Age (Months)')

    plt.show()

