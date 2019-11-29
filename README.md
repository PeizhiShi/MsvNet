## MsvNet
Created by Peizhi Shi at University of Huddersfield

Acknowledgements: We would like to thank Zhibo Zhang for providing the dataset and source code of FeatureNet on Github. 

### Introduction

The MsvNet is a novel learning-based feature recognition method using multiple sectional view representation. At the time of its release, the MsvNet achieves the state-of-the-art single feature recognition results by using only a few training samples, and outperforms the state-of-the-art learning-based multi-feature recognition method in terms of recognition performances.

This repository provides the source codes of the MsvNet for both single and multi-feature recognition, a reimplemented version of the FeatureNet for multi-feature recognition, and a benchmark dataset which contains 1000 3D models with multiple features.


### Single feature recognition

1. Prerequisites: python 3.+, pytorch, torchvision, numpy, cupy, scipy, PIL
2. Get the MsvNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/MsvNet.git`.
3. Download the FeatureNet [dataset](https://github.com/madlabub/Machining-feature-dataset), convert them into voxel models via [binvox](https://www.patrickmin.com/binvox/), and put them in the folder `data\64\`. `64` refers to the resolution of the voxel models.
4. Run `python single_train.py` to train the neural network. Please note that data augmentation is employed in this experiment. Thus, the training accuracy is lower than the val/test accuracy.


### Benchmark dataset for multi-feature recognition

1. Prerequisites: python 3.+, csv, pyvista
2. Get the MsvNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/MsvNet.git`.
3. Download the benchmark multi-feature [dataset], and put them in the folder `data\`.
4. Run `python visualize.py` to visualize a 3D model in this dataset.


### Multi-feature recognition

1. Prerequisites: python 3.+, [selective search](https://github.com/AlpacaDB/selectivesearch), numpy, cupy, tensorflow, pytorch, torchvision, skimage, scipy, PIL 
2. Get the MsvNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/MsvNet.git`.
3. Download the benchmark multi-feature [dataset](https://1drv.ms/u/s!At5UoWCCWHUKbKWk96k6GvQ4Hgs?e=deoxRs), and put them in the folder `data\`.
4. Download the pretrained MsvNet and FeatureNet [models](https://1drv.ms/u/s!At5UoWCCWHUKaM5mfNTkvL1tl_c?e=OHVMBR), and put them int the folder `models\`. These models could produce the multi-feature recognition results reported in the paper.
5. Run `python multi_test.py` to test the performances of the MsvNet and FeatureNet for multi-feature recognition. Please note that the multi-feature recognition part of the FeatureNet is only a reimplemented version. Detailed information about the FeatureNet can be found from their [original paper](https://doi.org/10.1016/j.cad.2018.03.006).

If you have any questions about the code, please feel free to contact me (p.shi@hud.ac.uk).
