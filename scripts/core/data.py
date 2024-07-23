"""
Rethinking Portrait Matting with Privacy Preserving
Inferernce file.

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/P3M-Net
Paper link: https://arxiv.org/abs/2203.16828

"""
import torch
import cv2
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from copy import deepcopy
import ipdb
import math
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from collections.abc import Iterable

from data_util import *
from config import *
from util import *


#########################
## collate
#########################

def collate_two_stage(batch):
    transposed = list(zip(*batch))
    global_data = transposed[:6]
    local_data = transposed[6:12]
    params = transposed[12:14]

    global_batch = [default_collate(list(elem)) for elem in global_data]
    local_batch = [torch.cat(list(elem), dim=0) for elem in local_data]
    return [*global_batch, *local_batch, *params]

#########################
## Data transformer
#########################
class MattingTransform(object):
    def __init__(self, crop_size, resize_size):
        super(MattingTransform, self).__init__()
        self.crop_size = crop_size
        self.resize_size = resize_size

    # args: image(blurred), mask, fg, bg, trimap, facemask, source_img, source_facemask
    def __call__(self, *args):
        ori = args[0]
        trimap = args[4]

        h, w, c = ori.shape
        crop_size = random.choice(self.crop_size)
        crop_size = crop_size if crop_size < min(h, w) else 512
        resize_size = self.resize_size

        target = np.where(trimap == 128) if random.random() < 0.5 else np.where(trimap > -100)
        if len(target[0]) == 0:
            target = np.where(trimap > -100)
        
        random_idx = np.random.choice(len(target[0]))
        centerh = target[0][random_idx]
        centerw = target[1][random_idx]
        crop_loc = self.safe_crop([centerh, centerw], crop_size, trimap.shape)

        flip_flag = True if random.random() < 0.5 else False

        args_transform = []
        for item in args:
            item = item[crop_loc[0]:crop_loc[2], crop_loc[1]:crop_loc[3]]
            if flip_flag:
                item = cv2.flip(item, 1)
            item = cv2.resize(item, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
            args_transform.append(item)
        
        return args_transform
    
    def safe_crop(self, center_pt, crop_size, img_size):
        h, w = img_size[:2]
        crop_size = min(h, w, crop_size)  # make sure crop_size <= min(h,w)

        center_h, center_w = center_pt

        left_top_h = max(center_h-crop_size//2, 0)
        right_bottom_h = min(h, left_top_h+crop_size)
        left_top_h = min(left_top_h, right_bottom_h-crop_size)

        left_top_w = max(center_w-crop_size//2, 0)
        right_bottom_w = min(w, left_top_w+crop_size)
        left_top_w = min(left_top_w, right_bottom_w-crop_size)

        return (left_top_h, left_top_w, right_bottom_h, right_bottom_w)


class MattingDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform):
        # Prepare transform
        self.transform = transform
        
        # Load data
        self.samples=[]
        self.samples += generate_paths_for_dataset(dataset=config.DATA.DATASET, trainset=config.DATA.TRAIN_SET)
        
    def __getitem__(self,index):
        # Prepare training sample paths
        ori_path, mask_path, fg_path, bg_path, facemask_path = self.samples[index]

        ori = np.array(Image.open(ori_path))
        mask = trim_img(np.array(Image.open(mask_path)))
        fg = np.array(Image.open(fg_path))
        bg = np.array(Image.open(bg_path))
        facemask = np.array(Image.open(facemask_path))[:,:,0:1]
        # Generate trimap/dilation/erosion online
        kernel_size = random.randint(15, 30)
        trimap = gen_trimap_with_dilate(mask, kernel_size)
        
        # Data transformation to generate samples (crop/flip/resize)
        # Transform input order: ori, mask, fg, bg, trimap
        argv = self.transform(ori, mask, fg, bg, trimap, facemask)
        argv_transform = []
        for item in argv:
            if item.ndim<3:
                item = torch.from_numpy(item.astype(np.float32)[np.newaxis, :, :])
            else:
                item = torch.from_numpy(item.astype(np.float32)).permute(2, 0, 1)
            argv_transform.append(item)

        [ori, mask, fg, bg, trimap, facemask] = argv_transform

        trimap[trimap > 180] = 255
        trimap[trimap < 50] = 0
        trimap[(trimap < 255) * (trimap > 0)] = 128

        facemask[facemask>100] = 255
        facemask[facemask<=100] = 0

        # normalize ori, fg, bg
        ori = ori/255.0
        fg = fg/255.0
        bg = bg/255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ori = normalize(ori)
        fg = normalize(fg)
        bg = normalize(bg)

        # output order: ori, mask, fg, bg, trimap
        return ori, mask, fg, bg, trimap, facemask

    def __len__(self):
        return len(self.samples)

