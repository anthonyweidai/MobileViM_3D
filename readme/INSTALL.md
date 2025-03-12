# Installation Instructions

The setup was validated using [Anaconda](https://www.anaconda.com/download), Python 3.10.13, CUDA 12.1, and [PyTorch]((http://pytorch.org/)) 2.1.2.
Following Anaconda installation, proceed as follows:

0. [Optional but highly recommended] Establish a new conda environment.
    Create the environment using the command:
    ~~~
    conda create -n mobilevim python=3.10.13
    ~~~

    Then activate this environment:
    ~~~
    conda activate mobilevim
    ~~~

1. Install PyTorch:
    ~~~
    conda install pytorch=2.1.2 torchvision=0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    ~~~

2. Clone the repository: 

    ~~~
    git clone git@github.com:anthonyweidai/MobileViM_3D.git
    ~~~

3. Install required Python packages:

    ~~~
    cd $MobileViM_ROOT
    pip install -r requirements.txt
    ~~~
    
4. Install additional Python libraries:

    ~~~
    pip install causal-conv1d>=1.4.0
    pip install mamba-ssm
    ~~~
