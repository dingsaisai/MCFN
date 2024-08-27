# MCFN
## Multimodal Co-attention Fusion Network with Online Data Augmentation for Cancer Subtype Classification
[(Accepted by IEEE TMI 2024)](https://ieeexplore.ieee.org/document/10539123/)

![Image text](https://github.com/dingsaisai/MCFN/blob/main/overview.png)

## Abstract

It is an essential task to accurately diagnose cancer subtypes in computational pathology for personalized cancer treatment. Recent studies have indicated that the combination of multimodal data, such as whole slide images (WSIs) and multiomics data, could achieve more accurate diagnosis. However, robust cancer diagnosis remains challenging due to the heterogeneity among multimodal data, as well as the performance degradation caused by insufficient multimodal patient data. In this work, we propose a novel multimodal co-attention fusion network (MCFN) with online data augmentation (ODA) for cancer subtype classification. Specifically, a multimodal mutual -guided coattention (MMC) module is proposed to effectively perform dense multimodal interactions. It enables multimodal data to mutually guide and calibrate each other during the integration process to alleviate inter- and intra-modal heterogeneities. Subsequently, a self-normalizing network (SNN)-Mixer is developed to allow information communication among different omics data and alleviate the high-dimensional small-sample size problem in multiomics data. Most importantly , to compensate for insufficient multimodal samples for model training, we propose an ODA module in MCFN. The ODA module leverages the multimodal knowledge to guide the data augmentations of WSIs and maximize the data diversity during model training. Extensive experiments are conducted on the public TCGA dataset. The experimental results demonstrate that the proposed MCFN outperforms all the compared algorithms, suggesting its effectiveness.

## Requirements
- Python >= 3.7 
- [PyTorch](https://pytorch.org/) >= 1.9 

## Data processing
### 1. Downloading TCGA Data
To download diagnostic WSIs (formatted as .svs files), please refer to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/). WSIs for each cancer type can be downloaded using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/).

### 2. Molecular Features
Processed molecular profile features containing mutation status, copy number variation, and RNA-Seq abundance can be downloaded from the [cBioPortal](https://www.cbioportal.org/). For RNA-Seq abundance, we selected the top 2000 genes with the largest median absolute deviation for inclusion. 

### 3. Processing Whole Slide Images
To process the WSI data we used the publicaly available [Slideflow Studio: a visualization tool for interacting with models and whole-slide images.](https://github.com/slideflow/slideflow/assets/48372806/7f43d8cb-dc80-427d-84c4-3e5a35fa1472). First, the tissue regions in each biopsy slide are segmented. The 299 x 299 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. Consequently, a pretrained ctranspath (https://github.com/Xiyue-Wang/TransPath) is used to encode raw image patches into 768-dim feature vector. Using the Slideflow toolbox, the features are saved as matrices of torch tensors of size N x 768, where N is the number of patches from each WSI (varies from slide to slide). 

## Train
After setting the parameters and save path in the file train.py, you can directly use the command line **python train.py** for training. The training process will be printed out, and the prediction results will be saved in the path.


The folder **data_csv** contains the all used patients ID.

## Citation
- If you found our work useful in your research, please consider citing our works(s) at:
```
@article{Ding2024Multimodal,
  title={Multimodal Co-attention Fusion Network with Online Data Augmentation for Cancer Subtype Classification},
  author={ Ding, Saisai  and  Li, Juncheng  and  Wang, Jun  and  Ying, Shihui  and  Shi, Jun },
  journal={IEEE Transactions on Medical Imaging},
  year={2024},
  publisher={IEEE}
}
```

