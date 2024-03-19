<div align="center">
<h1> PAP-Net: Pixel Affinity-Parsing Network for Amodal Instance Segmentation </h1>

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/jiaoZ7688/YOLOPX/blob/main/LICENSE) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.12+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
<br>

Jiao Zhan, Chi Guo, Bohan Yang, Yarong Luo, Yejun Wu, Jingyi Deng and Jingnan Liu
</div>

Table of Contents
* [Introduction](#introduction)
* [Usage](#usage)
* [Trained models](#trained-models)
* [Acknowledgement](#acknowledgement)

## News
* `2024-2-16`:  We've uploaded some code, and the full code will be released soon!

## Introduction
<div align = 'None'>
  <img src="figure/figure 3.jpg" width="100%" />
</div>
The figure above illustrates the prediction results of PAP-Net.

## Usage
### 1. Installation
```
conda create -n PAP-Net python=3.8 -y
source activate PAP-Net 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

git clone https://github.com/jiaoZ7688/PAP-Net
cd PAP-Net/
pip install -r requirements.txt
python3 setup.py build develop
(!!! Detectron2 must be installed successfully !!!)

```
### 2. Data preparation
#### KINS dataset
Download the [Images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip)
from [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d). 

The [Amodal Annotations](https://drive.google.com/drive/folders/1FuXz1Rrv5rrGG4n7KcQHVWKvSyr3Tkyo?usp=sharing)
could be found at [KINS dataset](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset)

#### D2SA dataset
The D2S Amodal dataset could be found at [mvtec-d2sa](https://www.mvtec.com/company/research/datasets/mvtec-d2s/).

#### COCOA-cls dataset
The COCOA dataset annotation from [here](https://drive.google.com/file/d/1n1vvOaT701dAttxxGeMKQa7k9OD_Ds51/view) (reference from github.com/YihongSun/Bayesian-Amodal)
The images of COCOA dataset is the train2014 and val2014 of [COCO dataset](http://cocodataset.org/).

#### Expected folder structure for each dataset
PAP-Net support datasets as coco format. It can be as follow (not necessarily the same as it depends on register data code)
```
KINS/
|--train_imgs
|--test_imgs/
|--annotations/
|----train.json
|----test.json
```
Then, See [here](detectron2/data/datasets/builtin.py) for more details on data registration

After registering, run the preprocessing scripts to generate occluder mask annotation, for example:
```
python -m detectron2.data.datasets.process_data_amodal \
   /path/to/KINS/train.json \
   /path/to/KINS/train_imgs \
   kins_dataset_train
```
the expected new annotation can be as follow:
```
KINS/
|--train_imgs
|--test_imgs/
|--annotations/
|----train.json
|----train_amodal.json
|----test.json
```

### 3. Training, Testing and Demo
Configuration files for training PAP-Net on each datasets are available [here](configs/).
To train, test and run demo, see the example scripts at [`scripts/`](scripts/):

## Trained models
- PAP-Net R50 on KINS (TBA)
- PAP-Net R50 on D2SA (<a href="https://pan.baidu.com/s/1gxJDSh7uEoZkoulKax-l6g">here</a>). Extraction code：meju
- PAP-Net R50 on COCOA-cls (TBA)

## Acknowledgement
- This code utilize [AISFormer](https://github.com/UARK-AICV/AISFormer) as the basis. 
- This code utilize [BCNet](https://github.com/lkeab/BCNet) for dataset mapping with occluder, [VRSP-Net](https://github.com/YutingXiao/Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior) for amodal evalutation, and [detectron2](https://github.com/facebookresearch/detectron2) as entire pipeline with Mask R-CNN meta arch.
