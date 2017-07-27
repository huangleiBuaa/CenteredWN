Centered Weight Normalization
======================================

This project is the code of the paper: Centered Weight Normalization  in Accelerating Training of Deep Neural Networks ( ICCV 2017).

The code depends on Torch http://torch.ch  and the optnet package by run: <br>
luarocks install optnet



## The experiment on MLP architecture over SVHN dataset.

(1)	Dataset prepraration:
 'cd dataset' and run the script  preProcess_div256.lua, by run: th preProcess_div256.lua, then we get the preprocessed SVHN dataset for MLP architecture. <br> 
Note that this script is based on the dataset process script at: https://github.com/torch/tutorials/blob/master/A_datasets/svhn.lua
<br>
<br>
(2)	Excute:  th exp_MLP.lua 



## The experiment on vgg-A architecture over Cifar-10 dataset:
(1)	 Dataset preparations: The dataset is based on the preprocessed script on: https://github.com/szagoruyko/cifar.torch, following the operation and putting the data file on the directory: ./dataset/cifar_provider.t7
<br>
<br>
(2)	Excute: th exp_vggA.lua –dataPath ***

## The experiment on GoogLeNet architecture over Cifar datasets:

(1)	Dataset preparations: The dataset is whitened and can be found on https://yadi.sk/d/em4b0FMgrnqxy.  
<br>
<br>
(2)	Excute: th exp_GoogleNet_dataWhitening.lua –dataPath ***
<br>
<br>
  The GoogLeNet model is base on the project on: https://github.com/soumith/imagenet-multiGPU.torch

## The experiment on residual network architecture over Cifar datasets:

(1)	Dataset preparations: The dataset is base on https://yadi.sk/d/eFmOduZyxaBrT, and the data file should put in the directory: ./dataset/cifar_original.t7.  
<br>
<br>
(3)	Excute: th exp_res_dataNorm.lua –dataPath ***
<br>
<br>
  The residual network model and respective script are based on the project: https://github.com/facebook/fb.resnet.torch


## The experiment on GoogLeNet over ImageNet:
For this experiment, we based on the project at: https://github.com/soumith/imagenet-multiGPU.torch.
<br>
<br>
The proposed model are in: ./models/imagenet

