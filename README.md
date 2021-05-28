## MsvNet

*Update (15/10/2020): Please check out our recent work published on IEEE TII ([paper](https://doi.org/10.1109/TII.2020.3030620), [code](https://github.com/PeizhiShi/SsdNet)), which yields better results than the MsvNet.

Created by Peizhi Shi at University of Huddersfield

Acknowledgements: We would like to thank Zhibo Zhang for providing the dataset and source code of FeatureNet on Github. 

### Introduction

The MsvNet is a novel learning-based feature recognition method using multiple sectional view representation. At the time of its release, the MsvNet achieves the state-of-the-art single feature recognition results by using only a few training samples, and outperforms the state-of-the-art learning-based multi-feature recognition method in terms of recognition performances.

This repository provides the source codes of the MsvNet for both single and multi-feature recognition, a reimplemented version of the FeatureNet for multi-feature recognition, and a benchmark dataset which contains 1000 3D models with multiple features.

If this project is useful to you, please consider citing our paper:

    @article{shi2020novel,
      title={A novel learning-based feature recognition method using multiple sectional view representation},
      author={Shi, Peizhi and Qi, Qunfen and Qin, Yuchu and Scott, Paul J and Jiang, Xiangqian},
      journal={Journal of Intelligent Manufacturing},
      volume={31},
      number={5},
      pages={1291--1309},
      year={2020},
      publisher={Springer}
    }
    
  
This is a peer-reviewed paper, which is available [online](https://link.springer.com/article/10.1007/s10845-020-01533-w).

You will NOT utilise the code for military, nuclear, missile, animal slaughter, meat production, weaponry end uses or conduct any other activities where human/animal life or property may be at stake.

### Experimental configuration

1. CUDA (10.0.130)
2. cupy-cuda100 (6.2.0)
3. numpy (1.17.4)
4. Pillow (6.2.1)
5. python (3.6.8)
6. pyvista (0.22.4)
7. scikit-image (0.16.2)
8. scipy (1.3.3)
9. selectivesearch (0.4)
10. tensorflow-estimator (1.14.0)
11. tensorflow-gpu (1.14.0)
12. torch (1.1.0)
13. torchvision (0.3.0)

All the experiments mentioned in our paper are conducted on Ubuntu 18.04 under the above experimental configurations. If you run the code on the Windows or under different configurations, slightly different results might be achieved.


### Single feature recognition

1. Get the MsvNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/MsvNet.git`.
2. Download the FeatureNet [dataset](https://github.com/madlabub/Machining-feature-dataset), and convert them into voxel models via [binvox](https://www.patrickmin.com/binvox/). The filename format is `label_index.binvox`. Then put all the `*.binvox` files in a same folder `data/64/`. `64` refers to the resolution of the voxel models. This folder is supposed to contain 24,000 `*.binvox` files. Please note there are some unlabelled/mislabelled files in category 8 (rectangular_blind_slot) and 12 (triangular_blind_step). Before moving these files in the same folder, please correct these filenames.
3. Run `python single_train.py` to train the neural network. Please note that data augmentation is employed in this experiment. Thus, the training accuracy is lower than the val/test accuracy.


### Benchmark dataset for multi-feature recognition

1. Get the MsvNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/MsvNet.git`.
2. Download the benchmark multi-feature [dataset](https://1drv.ms/u/s!At5UoWCCWHUKafomIKnOJnsl0Dg?e=lbK8iw), and put them in the folder `data/`.
3. Run `python visualize.py` to visualize a 3D model in this dataset.


### Multi-feature recognition

1. Get the MsvNet source code by cloning the repository: `git clone https://github.com/PeizhiShi/MsvNet.git`.
2. Download the benchmark multi-feature [dataset](https://1drv.ms/u/s!At5UoWCCWHUKafomIKnOJnsl0Dg?e=lbK8iw), and put them in the folder `data/`.
3. Download the pretrained optimal MsvNet and FeatureNet [models](https://1drv.ms/u/s!At5UoWCCWHUKaM5mfNTkvL1tl_c?e=OHVMBR), and put them int the folder `models/`. These models are trained under the optimal settings (instead of near-optimal settings) mentioned in our paper, which could produce the multi-feature recognition results reported in the paper. 
4. Run `python multi_test.py` to test the performances of the MsvNet and FeatureNet for multi-feature recognition. Please note that the multi-feature recognition part of the FeatureNet is only a reimplemented version. Watershed algorithm with the default setting is employed. Detailed information about the FeatureNet can be found from their [original paper](https://doi.org/10.1016/j.cad.2018.03.006).

If you have any questions about the code, please feel free to contact me (p.shi@hud.ac.uk).
