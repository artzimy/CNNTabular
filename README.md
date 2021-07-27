# Clinical_CNN.Tabular
Fused architecture comprised of 2 neural networks, pre-trained ResNet-50 Convolutional Neural Network (CNN) and tabular based network for the classification brain pathology

The proposed architecture built in fast.ai environment (fast.ai version 1.0.61 built on top of PyTorch environment), and was tested for magnetic resonance images and tubular data of age and gender

![CNN.Tabular architecture](https://github.com/artzimy/Clinical_CNN.Tabular-/blob/main/Clinical_CNN.Tabular.png) 

The fused architecture received tabular information in the input layer along with images. The architecture is comprised of 2 neural networks: A 2D ResNet50 CNN [He et al. IEEE 2016] as the base model and the tabular network. The CNN is identical to ResNet50's architecture up until the linear layer, and the tabular network consists of batch-norm embedding layer followed by a 4 node linear layer. 

Contact info: artzimy@gmail.com

# Overview

**IMPORTANT:** this code is built on fast.ai version 1.4. If you have version 2, please downgrade.

## Data organization

The dataset should include:

[Images] folder with the training and validation images and csv/xlsx file with data names and labels
  
[Test] folder with the test images and csv/xlsx file with data names and labels

*for the csv/xlsx files - see the demo examples files


## Included files
`Main`: Execution file

`Macros.py`: Define macros according to names in excel for readability

`TabConvData.py`: Combined Item contains MRI image and patient metadata

`TabConvModel.py`: The combined model class and fusion methods.

`Inference.py`: Predict on test set, print accuracy (majority vote iff majority_of_votes is True). Retrieve true and predicted labels for metrics calculation

DEMOtrain_val_images_5fold: example to the CSV file

## Data prerquistites
Test / train data should be organized in a single folder for each set and separate csvs. 
Column names should be:
```
ImageName | Group | Age | Sex | fold_1 | fold_2 | fold_3 | fold_4 | fold_5
``` 
Where `Group` refers to the data's labels, `fold_<i>` determines the training \ validation split for fold i (0 means training)

These are changeable in `Macros.py` to suit your needs.

## Model summary
The model is a fusion of two `Learner` instances from fast.ai's API. Hence, in order to create a fused learner for training we initialize two learners for each of our submodels.
This in turn requires initializing 3 `DataBunch` instances: image, tabular, and fused:
```python
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

```
These are then used in the learner instantiation, also preformed 3 times:
```python
tab_learn = tabular_learner(tab_data, metrics=[accuracy,KappaScore(weights='quadratic')], layers=[tab_out_sz])

img_learn = cnn_learner(img_data, models.resnet50, pretrained=True, metrics=[accuracy,KappaScore(weights='quadratic')])

learn = fuse_models(data, tab_learn, img_learn, n_lin_tab=tab_out_sz, n_lin_conv=conv_out_sz, ps=ps, 
                    wd=wd, metrics=[accuracy,KappaScore(weights='quadratic')], loss_func=loss_func)
``` 

From here on `learn` is trainable using `learn.fit`, `learn.fit_one_cycle`, and other fastai (verison 1)
methods.

## Citation
M. Artzi et al., "Classification of Pediatric Posterior Fossa Tumors Using Convolutional Neural Network and Tabular Data," in IEEE Access, vol. 9, pp. 91966-91973, 2021, doi: 10.1109/ACCESS.2021.3085771.

## Authors
Moran Artzi, Erez Redmard, Oron Tzemach, Jonathan Zeltser
