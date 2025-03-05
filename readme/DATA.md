# Dataset Preparation Overview

## PENGWIN Dataset
The PENGWIN segmentation task aims to improve automated methods for fracture segmentation in pelvic CT scans. It features CT scans from 150 patients across various institutions, each scheduled for pelvic reduction surgery. These scans include diverse patient demographics and fracture types, with semi-automatically annotated sacrum and hipbone segmentations validated by medical experts.

- **Access the dataset:** Download from [Pelvic Fracture Segmentation on CT](https://zenodo.org/records/10927452).

- **Data setup instructions:** Organize the images and corresponding masks in the following structure:

~~~
${MobileViM_ROOT}
|-- dataset
`-- |-- PENGWIN
    `-- |--- train
            |--- 001.nii.gz
            |--- 002.nii.gz
            |--- ...
            |--- 085.nii.gz
            |--- ...
        |--- test
            |--- 009.nii.gz
            |--- 012.nii.gz
            |--- ...
            |--- 092.nii.gz
            |--- ...
        |--- mask
            |--- 001.nii.gz
            |--- 002.nii.gz
            |--- ...
            |--- 092.nii.gz
            |--- ...
~~~

## BraTS2024 Dataset
The BraTS 2024 challenge introduces significant changes, focusing on radiotherapy planning MRIs for meningioma. The dataset includes brain MRI exams specifically for this treatment context.

- **Access the dataset:** Download from [Segmentation - Meningioma Radiotherapy](https://www.synapse.org/Synapse:syn53708249/wiki/627503).

- **Data setup instructions:**

~~~
${MobileViM_ROOT}
|-- dataset
`-- |-- BraTS2024
    `-- |--- train
            |--- BraTS-MEN-RT-0002-1.nii.gz
            |--- BraTS-MEN-RT-0003-1.nii.gz
            |--- ...
            |--- BraTS-MEN-RT-0144-1.nii.gz
            |--- ...
        |--- test
            |--- BraTS-MEN-RT-0016-1.nii.gz
            |--- BraTS-MEN-RT-0018-1.nii.gz
            |--- ...
            |--- BraTS-MEN-RT-0501-1.nii.gz
            |--- ...
        |--- mask
            |--- BraTS-MEN-RT-0002-1.nii.gz
            |--- BraTS-MEN-RT-0003-1.nii.gz
            |--- ...
            |--- BraTS-MEN-RT-0501-1.nii.gz
            |--- ...
~~~


## ATLAS Dataset
This dataset includes 90 T1 CE-MRI scans from patients with unresectable hepatocellular carcinoma, segmented into liver and tumor masks.

- **Access the dataset:** Download from [A Tumor and Liver Automatic Segmentation](https://atlas-challenge.u-bourgogne.fr).

- **Data setup instructions:**

~~~
${MobileViM_ROOT}
|-- dataset
`-- |-- ATLAS
    `-- |--- train
            |--- im1.nii.gz
            |--- im2.nii.gz
            |--- ...
            |--- im38.nii.gz
            |--- ...
        |--- test
            |--- im0.nii.gz
            |--- im5.nii.gz
            |--- ...
            |--- im42.nii.gz
            |--- ...
        |--- mask
            |--- im0.nii.gz
            |--- im1.nii.gz
            |--- ...
            |--- im38.nii.gz
            |--- ...
~~~

## ToothFairy2 Dataset
The ToothFairy2 dataset adheres to the nnU-Net format, including raw images, segmentation maps, and metadata in a dataset.json file. It classifies 42 dental and anatomical structures.

- **Access the dataset:** Download from [ToothFairy2 Dataset](https://ditto.ing.unimore.it/toothfairy2/).

- **Data setup instructions:**

~~~
${MobileViM_ROOT}
|-- dataset
`-- |-- ToothFairy2
    `-- |--- train
            |--- ToothFairy2F_001.nii.gz
            |--- ToothFairy2F_002.nii.gz
            |--- ...
            |--- ToothFairy2P_406.nii.gz
            |--- ...
        |--- test
            |--- ToothFairy2F_010.nii.gz
            |--- ToothFairy2F_012.nii.gz
            |--- ...
            |--- ToothFairy2P_411.nii.gz
            |--- ...
        |--- mask
            |--- ToothFairy2F_001.nii.gz
            |--- ToothFairy2F_002.nii.gz
            |--- ...
            |--- ToothFairy2P_411.nii.gz
            |--- ...
~~~


## References

For any scholarly use of these datasets, please consider citing the relevant papers:

~~~
@inproceedings{liu2023pelvic,
  title={Pelvic Fracture Segmentation Using a Multi-scale Distance-Weighted Neural Network},
  author={Liu, Yanzhen and Yibulayimu, Sutuke and Sang, Yudi and Zhu, Gang and Wang, Yu and Zhao, Chunpeng and Wu, Xinbao},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={312--321},
  year={2023},
  organization={Springer}
}

@article{labella2024brain,
  title={Brain tumor segmentation ({BraTS}) challenge 2024: Meningioma radiotherapy planning automated segmentation},
  author={LaBella, Dominic and Schumacher, Katherine and Mix, Michael and Leu, Kevin and McBurney-Lin, Shan and Nedelec, Pierre and Villanueva-Meyer, Javier and Shapey, Jonathan and Vercauteren, Tom and Chia, Kazumi and others},
  journal={arXiv preprint arXiv:2405.18383},
  year={2024}
}

@article{quinton2023tumour,
  title={A Tumour and Liver Automatic Segmentation ({ATLAS}) Dataset on Contrast-Enhanced Magnetic Resonance Imaging for Hepatocellular Carcinoma},
  author={Quinton, F{\'e}lix and Popoff, Romain and Presles, Beno{\^\i}t and Leclerc, Sarah and Meriaudeau, Fabrice and Nodari, Guillaume and Lopez, Olivier and Pellegrinelli, Julie and Chevallier, Olivier and Ginhac, Dominique and others},
  journal={Data},
  volume={8},
  number={5},
  pages={79},
  year={2023},
  publisher={MDPI}
}

@article{lumetti2024enhancing,
  title={Enhancing Patch-Based Learning for the Segmentation of the Mandibular Canal},
  author={Lumetti, Luca and Pipoli, Vittorio and Bolelli, Federico and Ficarra, Elisa and Grana, Costantino},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}

@misc{dai2025mobilevim,
      title={MobileViM: A Light-weight and Dimension-independent Vision Mamba for 3D Medical Image Analysis},
      author={Dai, Wei and Liu, Jun},
      year={2025},
      eprint={2502.13524},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.13524}, 
}
~~~
