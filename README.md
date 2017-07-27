Centered Weight Normalization
======================================

This project is the code of the paper: Centered Weight Normalization  in Accelerating Training of Deep Neural Networks ( ICCV 2017).


## Requirements and Dependency
* install [Torch](http://torch.ch) with CUDA GPU
* install [cudnn v5](http://torch.ch)
* install dependent lua packages optnet by run:
luarocks install optnet


## Experiments in the paper

#### 1. MLP architecture over SVHN dataset

* Dataset prepraration, by running:
 (1)  'cd dataset';
 (2)  'th preProcess_div256.lua';
We get the preprocessed SVHN dataset for MLP architecture.
Note that this script is based on the dataset process script at: https://github.com/torch/tutorials/blob/master/A_datasets/svhn.lua

*	Execute:  'th exp_MLP.lua '



#### 2. VGG-A architecture over Cifar-10 dataset
* Dataset preparations: the dataset is based on the preprocessed script on: https://github.com/szagoruyko/cifar.torch, and put the data file in the directory: ./dataset/cifar_provider.t7

* Execute: th exp_vggA.lua –dataPath './dataset/cifar_provider.t7'

#### 3. GoogLeNet architecture over Cifar datasets

 *	Dataset preparations: The dataset is based on [whitened CIFAR datasets](https://yadi.sk/d/em4b0FMgrnqxy).  
 * Execute: th exp_GoogleNet_dataWhitening.lua –dataPath './dataset/cifar100_whitened.t7'.
 <br>
  The GoogLeNet model is based on the project on: https://github.com/soumith/imagenet-multiGPU.torch

#### 4. Residual network architecture over Cifar datasets

 *	Dataset preparations: The dataset is based on [original CIFAR datasets](https://yadi.sk/d/eFmOduZyxaBrT), and the data file should put in the directory: ./dataset/cifar_original.t7.  
 *	Execute: th exp_res_dataNorm.lua –dataPath './dataset/cifar10_original.t7'. 
 <br>
  The normlization of Cifar dataset is in the script th exp_res_dataNorm.lua. The residual network model and respective script are based on [facebook ResNet](https://github.com/facebook/fb.resnet.torch).

####  5. GoogLeNet over ImageNet
This experiment is based on the project at: https://github.com/soumith/imagenet-multiGPU.torch.
<br>
The proposed model are in: './models/imagenet/'

