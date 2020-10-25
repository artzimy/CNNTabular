#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision import *
from fastai.tabular import *
from fastai.layers import *


# # Define Model

# **Build combined model architecture using tabular and image cnn learners**

# In[2]:


class TabConvModel(nn.Module):
    def __init__(self, tab_model, img_model, layers, drops):
        super().__init__()
        self.tab_model = tab_model #based on fastai_tabular learner
        self.img_model = img_model #based on fastai cnn_learner
        lst_layers = []

        activs = [nn.ReLU(inplace=True),] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            lst_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)

        self.layers = nn.Sequential(*lst_layers)

    def forward(self, x_cont, img, x_cat=None):
        x_tab = self.tab_model(x_cat, x_cont)
        x_img = self.img_model(img)

        x = torch.cat([x_tab, x_img], dim=1)
        return self.layers(x)


# **Create combined learner based on TabConvModel**

# In[3]:


def fuse_models(data, tab_learner, img_learner, n_lin_tab=4, n_lin_conv=4, fused_out_sz=4, 
                ps=0.25, wd=0.001, metrics=None, callback_fns=None, loss_func=CrossEntropyFlat):
    
    concat_sz = n_lin_tab + n_lin_conv
    lin_layers = [concat_sz, concat_sz, fused_out_sz]
    ps_list = [ps, ps]
    model = TabConvModel(tab_learner.model, img_learner.model, lin_layers, ps_list)

    layer_groups = [nn.Sequential(*flatten_model(img_learner.layer_groups[0])),
                    nn.Sequential(*flatten_model(img_learner.layer_groups[1])),
                    nn.Sequential(*(flatten_model(img_learner.layer_groups[2]) +
                                    flatten_model(model.tab_model) +
                                    flatten_model(model.layers)))]

    fused_learner = Learner(data, model, layer_groups=layer_groups, loss_func=loss_func, 
                            metrics=metrics, callback_fns=callback_fns, wd=wd)
    return fused_learner

