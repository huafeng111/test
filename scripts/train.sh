#!/bin/bash

# Rethinking Portrait Matting with Privacy Preserving
# 
# Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
# Licensed under the MIT License (see LICENSE for details)
# Github repo: https://github.com/ViTAE-Transformer/P3M-Net
# Paper link: https://arxiv.org/abs/2203.16828


cfg='core/configs/ViTAE_S.yaml'  # change config file here
nEpochs=150
lr=0.00001
nickname=debug_train_vitae_s  # change run name here
batchSize=8

python core/train.py \
    --cfg $cfg \
    --tag $nickname \
    --nEpochs $nEpochs \
    --lr=$lr \
    --batchSize=$batchSize