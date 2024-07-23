# P3M-Net Setup Guide

1. Change directory to P3M-Net:
   ```bash
   cd P3M-Net

1. Create a new conda environment named p3m with Python 3.7.7:
   ```bash
   conda create -n p3m python=3.7.7

1. Activate the conda environment:
   ```bash
   conda activate p3m
   
1. Install required packages from requirements.txt:
   ```bash
    pip install -r requirements.txt

1. Install specific versions of PyTorch, torchvision, torchaudio, and cudatoolkit:
   ```bash
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch

1. Download the P3M-Net (ViTAE-S) model and place it in the /scripts/models directory:
   ```bash
    Download link
    https://drive.google.com/file/d/1QbSjPA_Mxs7rITp_a9OJiPeFRDwxemqK/view
    Place your test image in the \scripts\p3m_out\samples\original directory.
    Run test
    cd scripts
    ./test_samples.sh
