# MobileViM: A Light-weight and Dimension-independent Vision Mamba for 3D Medical Image Analysis
Welcome to the official implementation of ``[MobileViM: A Light-weight and Dimension-independent Vision Mamba for 3D Medical Image Analysis](https://arxiv.org/abs/2502.13524)''. This repository provides a comprehensive toolkit optimized for deep learning and computer vision applications, particularly focusing on 3D semantic segmentation. It includes functionalities for visualizing training progress, logging activities, and computing standard performance metrics.

<p align="center" width="100%">
    <img width="100%" src=./readme/architecture_animation.gif>
    <br>Architecture overview.
</p>

<p align="center" width="100%">
    <img width="88%" src=./readme/dimin.jpg>
    <br>Dimension-independent mechanism.
</p>

<p align="center" width="100%">
    <img width="55%" src=./readme/mamba.jpg>
    <br>Dual-direction Mamba.
</p>

**The manuscript is currently under review by a peer-reviewed journal. The full code is accessible only to the reviewers and will be made publicly available upon the manuscript's acceptance.**


## Setup

For installation instructions of MobileViM, please consult the comprehensive guide in [INSTALL.md](readme/INSTALL.md).

## Benchmarking and Evaluation

Refer to [DATA.md](readme/DATA.md) for detailed instructions on data preparation for benchmarking and model training.

To begin training and evaluation, follow the configuration details in the [ours.sh](shell/ours.sh) script.

## Ablation studies

For specifics on ablation studies and further experiments, please see the scripts in [ablation.sh](shell/ablation.sh).

## Citing Our Work

If our implementation aids your research, please acknowledge it by citing our paper:

    @misc{dai2025mobilevim,
      title={MobileViM: A Light-weight and Dimension-independent Vision Mamba for 3D Medical Image Analysis},
      author={Dai, Wei and Wang, Steven and Liu, Jun},
      year={2025},
      eprint={2502.13524},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.13524}, 
}
