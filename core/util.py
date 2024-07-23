"""
Rethinking Portrait Matting with Privacy Preserving

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting.git
Paper link: https://arxiv.org/abs/2203.16828

"""
import os
import shutil
import cv2
import numpy as np
import torch
import torch.distributed as dist
import glob
import functools
from torchvision import transforms
from config import *


def get_wandb_config(config):
    wandb_args = {}
    wandb_args['model'] = config.MODEL.TYPE
    wandb_args['logname'] = config.TAG
    return wandb_args

def get_wandb_key(filename):
    f = open(filename)
    keys = f.readline().strip()
    f.close()
    return keys

##########################
### Pure functions
##########################

def GET_RANK():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

def GET_WORLD_SIZE():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        1

def is_main_process():
    if not dist.is_initialized():
        return True
    if dist.is_initialized() and dist.get_rank() == 0:
        return True
    return False

def extract_pure_name(original_name):
    pure_name, extention = os.path.splitext(original_name)
    return pure_name

def listdir_nohidden(path):
    new_list = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            new_list.append(f)
    new_list.sort()
    return new_list

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def refresh_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

def save_test_result(save_dir, predict):
    predict = (predict * 255).astype(np.uint8)
    cv2.imwrite(save_dir, predict)

def generate_composite_img(img, alpha_channel):
    b_channel, g_channel, r_channel = cv2.split(img)
    b_channel = b_channel * alpha_channel
    g_channel = g_channel * alpha_channel
    r_channel = r_channel * alpha_channel
    alpha_channel = (alpha_channel*255).astype(b_channel.dtype)	
    img_BGRA = cv2.merge((r_channel,g_channel,b_channel,alpha_channel))
    return img_BGRA

##########################
### for dataset processing
##########################
def trim_img(img):
    if img.ndim>2:
        img = img[:,:,0]
    return img

def gen_trimap_with_dilate(alpha, kernel_size):	
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    dilate =  cv2.dilate(fg_and_unknown, kernel, iterations=1)
    erode = cv2.erode(fg, kernel, iterations=1)
    trimap = erode *255 + (dilate-erode)*128
    return trimap.astype(np.uint8)

def normalize_batch_torch(data_t):
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    new_data = []
    for i in range(data_t.shape[0]):
        new_data.append(normalize_transform(data_t[i]))
    return torch.stack(new_data, dim=0)

##########################
### Functions for fusion 
##########################
def gen_trimap_from_segmap_e2e(segmap):
    trimap = np.argmax(segmap, axis=1)[0]
    trimap = trimap.astype(np.int64)	
    trimap[trimap==1]=128
    trimap[trimap==2]=255
    return trimap.astype(np.uint8)

def get_masked_local_from_global(global_sigmoid, local_sigmoid):
    values, index = torch.max(global_sigmoid,1)
    index = index[:,None,:,:].float()
    ### index <===> [0, 1, 2]
    ### bg_mask <===> [1, 0, 0]
    bg_mask = index.clone()
    bg_mask[bg_mask==2]=1
    bg_mask = 1- bg_mask
    ### trimap_mask <===> [0, 1, 0]
    trimap_mask = index.clone()
    trimap_mask[trimap_mask==2]=0
    ### fg_mask <===> [0, 0, 1]
    fg_mask = index.clone()
    fg_mask[fg_mask==1]=0
    fg_mask[fg_mask==2]=1
    fusion_sigmoid = local_sigmoid*trimap_mask+fg_mask
    return fusion_sigmoid

def get_masked_local_from_global_test(global_result, local_result):
    weighted_global = np.ones(global_result.shape)
    weighted_global[global_result==255] = 0
    weighted_global[global_result==0] = 0
    fusion_result = global_result*(1.-weighted_global)/255+local_result*weighted_global
    return fusion_result

#######################################
### Function to generate training data
#######################################

def generate_paths_for_dataset(dataset="P3M10K", trainset="TRAIN"):
    ORI_PATH = DATASET_PATHS_DICT[dataset][trainset]['ORIGINAL_PATH']
    MASK_PATH = DATASET_PATHS_DICT[dataset][trainset]['MASK_PATH']
    FG_PATH = DATASET_PATHS_DICT[dataset][trainset]['FG_PATH']
    BG_PATH = DATASET_PATHS_DICT[dataset][trainset]['BG_PATH']
    FACEMASK_PATH = DATASET_PATHS_DICT[dataset][trainset]['PRIVACY_MASK_PATH']	
    mask_list = listdir_nohidden(MASK_PATH)
    total_number = len(mask_list)
    paths_list = []
    for mask_name in mask_list:
        path_list = []
        ori_path = ORI_PATH+extract_pure_name(mask_name)+'.jpg'
        mask_path = MASK_PATH+mask_name
        fg_path = FG_PATH+mask_name
        bg_path = BG_PATH+extract_pure_name(mask_name)+'.jpg'
        facemask_path = FACEMASK_PATH+mask_name
        path_list.append(ori_path)	
        path_list.append(mask_path)	
        path_list.append(fg_path)
        path_list.append(bg_path)
        path_list.append(facemask_path)
        paths_list.append(path_list)
    return paths_list


def get_valid_names(*dirs):
    # Extract valid names
    name_sets = [get_name_set(d) for d in dirs]

    # Reduce
    def _join_and(a, b):
        return a & b

    valid_names = list(functools.reduce(_join_and, name_sets))
    if len(valid_names) == 0:
        return None
    
    valid_names.sort()

    return valid_names

def get_name_set(dir_name):
    path_list = glob.glob(os.path.join(dir_name, '*'))
    name_set = set()
    for path in path_list:
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        if name.startswith(".DS"): continue
        name_set.add(name)
    return name_set

def list_abspath(data_dir, ext, data_list):
    return [os.path.join(data_dir, name + ext)
            for name in data_list]