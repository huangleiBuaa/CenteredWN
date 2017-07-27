Centered Weight Normalization
======================================

This project is the code of the paper: Centered Weight Normalization  in Accelerating Training of Deep Neural Networks ( ICCV 2017).


# Requirements and Dependency
* install [Torch](http://torch.ch) with CUDA GPU
<br>

* install [cudnn v5](http://torch.ch)
<br>

* install dependent lua packages optnet by run:
luarocks install optnet


# Experiments in the paper

## MLP architecture over SVHN dataset

* 	Dataset prepraration, by running:
 (1)  'cd dataset';
 (2)  'th preProcess_div256.lua';
<br> 
We get the preprocessed SVHN dataset for MLP architecture.
<br> 
Note that this script is based on the dataset process script at: https://github.com/torch/tutorials/blob/master/A_datasets/svhn.lua
<br>

*	Execute:  'th exp_MLP.lua '



## VGG-A architecture over Cifar-10 dataset
*	 Dataset preparations: the dataset is based on the preprocessed script on: https://github.com/szagoruyko/cifar.torch, and put the data file in the directory: ./dataset/cifar_provider.t7
<br>

*		Execute: th exp_vggA.lua –dataPath './dataset/cifar_provider.t7'

## GoogLeNet architecture over Cifar datasets

 *	Dataset preparations: The dataset is whitened and can be found on https://yadi.sk/d/em4b0FMgrnqxy.  
<br>
 *	Excute: th exp_GoogleNet_dataWhitening.lua –dataPath './dataset/cifar100_whitened.t7'
<br>

  The GoogLeNet model is based on the project on: https://github.com/soumith/imagenet-multiGPU.torch

## Residual network architecture over Cifar datasets

 *	Dataset preparations: The dataset is based on https://yadi.sk/d/eFmOduZyxaBrT, and the data file should put in the directory: ./dataset/cifar_original.t7.  

<br>
 *	Execute: th exp_res_dataNorm.lua –dataPath './dataset/cifar10_original.t7'
<br>
  The residual network model and respective script are based on the project: https://github.com/facebook/fb.resnet.torch


##  GoogLeNet over ImageNet
This experiment is based on the project at: https://github.com/soumith/imagenet-multiGPU.torch.
<br>
The proposed model are in: './models/imagenet/'

