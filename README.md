# HiFuse
![](https://img.shields.io/github/license/huoxiangzuo/HiFuse)  
This repo. is the official implementation of "**HiFuse: Global and Local Feature Hierarchical Fusion Network for Medical Image Classification**"  
Authors: Xiangzuo Huo, Aolun Li, Shengwei Tian, et al.  
Enjoy the code and find its convenience to produce more awesome works!

***06/06/2022***
Pretrained models of [HiFuse on ImageNet-1K](https://drive.google.com/file/d/1HnvSncnU9GzeIXnskGXD8_gigZlr3lzX/view?usp=sharing) are released.

## Overview
![paper1_1_01(1)](https://user-images.githubusercontent.com/57312968/170870503-0b2c1728-daa8-4f80-a79b-d66c6748ac83.png)

## Grad-CAM results in ablation experiments
<img src="https://user-images.githubusercontent.com/57312968/170870613-41fbdeb6-f8db-4117-9a2c-133e0ee23d18.png" width="66%"/>

## Run
0. Requirements:
* python3.6/3.7/3.8
* pytorch 1.10
* torchvision 0.11.1
1. Training:
* Prepare the required images and store them in categories, set up training image folders and validation image folders respectively
* Run `python train.py`
2. Resume training:
* Modify `parser.add_argument('--RESUME', type=bool, default=True)` in `python train.py`
* Run `python train.py`
3. Testing:
* Run `python test.py`

## TensorBoard
Run `tensorboard --logdir runs --port 6006` to view training progress

## Reference
Some of the codes in this repo are borrowed from:  
* [Swin Transformer](https://github.com/microsoft/Swin-Transformer)  
* [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)  
* [WZMIAOMIAO](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)

## Citation

If you find our paper/code is helpful, please consider citing:
