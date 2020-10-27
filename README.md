# Clinical_CNN.Tabular
Fused architecture comprised of 2 neural networks, pre-trained ResNet-50 Convolutional Neural Network (CNN) and tabular based network for the classification brain pathology

The proposed architecture built in fast.ai environment (fast.ai version 1.0.61 built on top of PyTorch environment), and was tested for magnetic resonance images and tubular data of age and gender

![CNN.Tabular architecture](https://github.com/artzimy/Clinical_CNN.Tabular-/blob/main/Clinical_CNN.Tabular.png) 

The fused architecture received tabular information in the input layer along with images. The architecture is comprised of 2 neural networks: A 2D ResNet50 CNN [He et al. IEEE 2016] as the base model and the tabular network. The CNN is identical to ResNet50's architecture up until the linear layer, and the tabular network consists of batch-norm embedding layer followed by a 4 node linear layer. 

Contact info: artzimy@gmail.com

# Overview
## Data organization

The dataset should include:

[Images] folder with the training and validation images and csv/xlsx file with data names and labels
  
[Test] folder with the test images and csv/xlsx file with data names and labels

*for the csv/xlsx files - see the demo examples files


## Included files
Main: Execution file

Macros: Define macros according to names in excel for readability

TabConvData: Combined Item contains MRI image and patient metadata

Inference: Predict on test set, print accuracy (majority vote iff majority_of_votes is True). Retrieve true and predicted labels for metrics calculation

DEMOtrain_val_images_5fold: example to the CSV file

## Citation
[]

## Authors
Moran Artzi, Erez Redmard, Oron Tzemach, Jonathan Zeltser
