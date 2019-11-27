## MsvNet
Created by Peizhi Shi at University of Huddersfield

Acknowledgements: We would like to thank Zhibo Zhang for providing the dataset and source code of FeatureNet on Github. 

### Introduction

The MsvNet is a novel learning-based feature recognition method using multiple sectional view representation. At the time of its release, the MsvNet achieves the state-of-the-art single feature recognition results by using only a few training samples, and outperforms the state-of-the-art learning-based multi-feature recognition method in terms of recognition performances.

This repository provides the source codes of the MsvNet for both single and multi-feature recognition, a reimplemented version of the FeatureNet for multi-feature recognition, and a benchmark dataset which contains 1000 3D models with multiple features.


### Single feature recognition

1. Prerequisites: python 3.+, pytorch, torchvision, numpy, cupy, 
2. Get the MsvNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/MsvNet.git`.
3. Download the FeatureNet [dataset](https://github.com/madlabub/Machining-feature-dataset), convert them into voxel models via [binvox](https://www.patrickmin.com/binvox/), and put them in the folder `data\64\`.
4. Run `python single_train.py` to train the neural network.


### Benchmark dataset for multi-feature recognition

1. Prerequisites: python 3.+, 
2. Get the MsvNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/MsvNet.git`.
3. Download the benchmark multi-feature [dataset], and put them in the folder `data\`.
4. Run `python visualize.py` to visualize a 3D model with multiple features.


### Multi-feature recognition

1. Prerequisites: python 3.+, [selective search](https://github.com/AlpacaDB/selectivesearch)
2. Get the MsvNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/MsvNet.git`.
3. Download the benchmark multi-feature [dataset], and put them in the folder `data\`.
4. Download the pretrained MsvNet and FeatureNet [models], and put them int the folder `models\`. These models could produce the multi-feature recognition results reported in the paper.
5. Run `python multi_test.py -m MsvNet -d 5` to test the performances of the MsvNet for multi-feature recognition on data group 5.
6. Run `python multi_test.py -m FeatureNet -d 6` to test the performances of the FeatureNet for multi-feature recognition on data group 6. Please note that this is only a reimplemented version of the FeatureNet for multi-feature recognition. Detailed information can be found from their [original paper](https://doi.org/10.1016/j.cad.2018.03.006).
