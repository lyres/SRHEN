# SRHEN

This is a better and simpler implementation for "SRHEN: Stepwise-Refining Homography Estimation Network via Parsing Geometric Correspondences in Deep Latent Space". 

**MACE=1.18** on Synthetic COCO dataset. (MACE=9.19 in the original paper, without using the coarse-to-fine framework).

If you find this work useful, please consider citing:

>@inproceedings{li2020srhen,  
>title={SRHEN: Stepwise-Refining Homography Estimation Network via Parsing Geometric Correspondences in Deep Latent Space},  
>author={Li, Yi and Pei, Wenjie and He, Zhenyu},  
>booktitle={Proceedings of the 28th ACM International Conference on Multimedia},  
>pages={3063--3071},  
>year={2020}  
>}  

# Some modifications
1. We use pretrained ResNet34 instead of VGG-like network as backbone.
2. COCO images are resized to 320X320 rather than 320X240, to better avoid black border.
3. Patch pairs are generated online rather then offline, to alleviate the overfitting problem.
4. We compute a global cost volume (i.e., the correspondence map in the paper) rather than a local one.
5. We use inner product rather than cosine similarity to compute the cost volume.
6. The coarse-to-fine framework and the pyramidal supervision scheme is NOT included in this implementation.

# Requirements
* python 3.6.7
* opencv-python 4.1.0
* torch 1.10.0
* torchvision 0.7.0

# Preparation
1. Download COCO dataset. https://paperswithcode.com/dataset/coco.
2. Change the directory setting in `"preprocess_images_offline.py"`, i.e., `DIR_IMG` and `DIR_OUT` according to your own directory.
3. Run `"python preprocess_images_offline.py"`.

# Train
1. Change the directory setting in `"train.py"`, i.e., `DIR_IMG` and `DIR_MOD` for train images and trained models, respectively.
2. Run `"python train.py"`.

# Test
1. Change the directory setting in `"test.py"`, i.e., `DIR_IMG` and `DIR_MOD` for test images and saved models, respectively.
2. Run `"python test.py"`.
