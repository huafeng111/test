#!/bin/bash

# Rethinking Portrait Matting with Privacy Preserving
# 
# Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
# Licensed under the MIT License (see LICENSE for details)
# Github repo: https://github.com/ViTAE-Transformer/P3M-Net
# Paper link: https://arxiv.org/abs/2203.16828

tag=''  # run name
dataset_choice='P3M_500_P'
ckpt_name='ckpt_latest'
test_choice='HYBRID'

python core/test.py \
     --tag=$tag \
     --test_dataset=$dataset_choice \
     --test_ckpt=$ckpt_name \
     --test_method=$test_choice \
     --fast_test \
     --test_privacy