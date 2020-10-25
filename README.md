# Clinical_CNN.Tabular-
Fused architecture comprised of 2 neural networks, pre-trained ResNet-50 Convolutional Neural Network (CNN) and tabular based network for the classification brain pathology

The proposed architecture built in fast.ai environment supports various CNN architectures (fast.ai version 1.0.61 built on top of PyTorch environment).

![CNN.Tabular architecture](https://github.com/artzimy/Clinical_CNN.Tabular-/blob/main/Clinical_CNN.Tabular.tif)

The fused architecture is an integrated architecture which received tabular information in the input layer along with images (tested for MRI data). The architecture is comprised of 2 neural networks: A 2D ResNet50 CNN [He et al. IEEE 2016] as the base model and the tabular network. The CNN is identical to ResNet50's architecture up until the linear layer, and the tabular network consists of batch-norm embedding layer followed by a 4 node linear layer. The inputs are the scan and tabular data to each model respectively; CNN receives a 256*256*3 tensor representing the scan; Tabular neural network receives normalized age. The CNN's 512 node linear layer and the tabular network's 4 node layer are then concatenated, batch normalized and dropped out, before arriving at the 4 node final layer.
