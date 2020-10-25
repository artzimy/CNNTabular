#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from Macros import *
from TabConvData import *
from TabConvModel import *
from Inference import *


# In[ ]:


cuda_num = '0'
torch.cuda.set_device(int(cuda_num))


# Set batch size according to useable memory and imsize according to model needs

# In[ ]:


bs = 32
imsize = 256

np.random.seed(2)


# **Load Data**

# In[ ]:


path = "/media/df3-moran/Omri/Pediatric_Data_Set_New_256_noaug/3ch_data/diff train_val/"
df = pd.read_csv(path+'train_val_images_5fold_augment.csv')


# Define validation fold index (1 to 5) 

# In[ ]:


valid_idx = 'fold_3'


# In[ ]:


df_data = set_df(df, valid_idx)


# In[ ]:


tfms = get_transforms(do_flip=False, max_rotate=5, max_zoom=1.05)

procs = [Normalize]


# Prepare image, tabular and combined data bunches.
# split to train and validation sets according to given column

# In[ ]:


img_data = (ImageList.from_df(df_data, path, suffix='.png')
            .split_from_df(col=[valid_idx])
            .label_from_df()
            .transform(tfms, size=imsize, resize_method=ResizeMethod.SQUISH,padding_mode='zeros')
            .databunch(device=torch.device('cuda:'+cuda_num), bs=bs))

tab_data = (TabularList.from_df(df_data, cont_names=cont_names, procs=procs)
            .split_from_df(col=[valid_idx])
            .label_from_df(cols=CLASS)
            .databunch(device=torch.device('cuda:'+cuda_num), bs=bs))

data = (TabConvList.from_df(df_data, cont_names=cont_names, procs=procs,
                           path=path, imgs=[ID],suffix='.png')
        .split_from_df(col=[valid_idx])
        .label_from_df(cols=CLASS, label_cls=CategoryList)
        .transform(tfms, size=imsize, resize_method=ResizeMethod.SQUISH,padding_mode='zeros')
        .databunch(device=torch.device('cuda:'+cuda_num), bs=bs, collate_fn=my_collate))

data.show_batch(rows=9, figsize=(9,9))


# **Initialize TabConvModel**

# In[ ]:


tab_out_sz = 4
conv_out_sz = 4
ps = 0.25
wd = 0.001

loss_func = CrossEntropyFlat()

tab_learn = tabular_learner(tab_data, metrics=[accuracy,KappaScore(weights='quadratic')], layers=[tab_out_sz])

img_learn = cnn_learner(img_data, models.resnet50, pretrained=True, metrics=[accuracy,KappaScore(weights='quadratic')])

learn = fuse_models(data, tab_learn, img_learn, n_lin_tab=tab_out_sz, n_lin_conv=conv_out_sz, ps=ps, 
                    wd=wd, metrics=[accuracy,KappaScore(weights='quadratic')], loss_func=loss_func)

learn.model_dir = "/media/df3-moran/Pediatric_PFT/MetaData_Project/Models/Final_Models_AgeOnly/"


# In[ ]:


print(learn.summary())


# **Training**

# In[ ]:


lr_find(learn)
learn.recorder.plot(suggestion=True)


# In[ ]:


lr = 1e-2
epochs = 50

learn.fit_one_cycle(epochs, max_lr=slice(lr/3,lr), callbacks=[callbacks.SaveModelCallback(learn, every='improvement', monitor='accuracy',name='best')]) 


# In[ ]:


lr = 1e-3
epochs = 10

learn.fit_one_cycle(epochs, max_lr=slice(lr/3,lr), callbacks=[callbacks.SaveModelCallback(learn, every='improvement', monitor='accuracy',name='best')])


# Save & Load trained model

# In[ ]:


# learn.save("PFT_classifier_AgeOnly_NewArch_"+valid_idx)


# In[ ]:


# learn = learn.load("PFT_classifier_AgeOnly_NewArch_" + valid_idx)
learn = learn.load("PFT_best_classifier_MetaData_AgeOnly_" + valid_idx)


# **View Results on Validation Set**

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(6,6), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)
losses,idxs = interp.top_losses(9)


# In[ ]:


preds, true_labels = learn.get_preds()
predicted_labels = torch.argmax(preds, dim=1)
classification_report(true_labels, predicted_labels, output_dict=True)


# **Test Inference**

# In[ ]:


test_path = "/media/df3-moran/Omri/Pediatric_Data_Set_New_256_noaug/3ch_data/diff test"
test_csv = "test_images.csv"


# In[ ]:


true_labels, predicted_labels = predict_test(test_path, test_csv, learn, majority_of_votes=False, show_preds=False)

# calculate metrics: recall, presicion, f_score
classification_report(true_labels, predicted_labels, output_dict=True)


# In[ ]:


true_labels, predicted_labels = predict_test(test_path, test_csv, learn, majority_of_votes=True)


# In[ ]:


show_test_results(test_path, test_csv, learn, 13, 29)

