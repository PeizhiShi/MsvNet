## MsvNet
Created by Peizhi Shi at University of Huddersfield

Acknowledgements: We would like to thank Zhibo Zhang for providing the dataset and source code of FeatureNet on Github. 

### Introduction
The MsvNet is a novel learning-based feature recognition method using multiple sectional view representation. At the time of its release, the MsvNet achieves the state-of-the-art single feature recognition results by using only a few training samples, and outperforms the state-of-the-art learning-based multi-feature recognition method in terms of recognition performances.

### Single feature recognition

1. Prerequisites: python 3.+, pytorch, torchvision, numpy, cupy, 
2. Download the FeatureNet [dataset](https://github.com/madlabub/Machining-feature-dataset), and convert them into voxel models via binvox
3. Get the MsvNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/MsvNet.git`
4. Run `single.py`


### Multi-feature recognition
