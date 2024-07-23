"""
Rethinking Portrait Matting with Privacy Preserving
Inferernce file.

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/P3M-Net
Paper link: https://arxiv.org/abs/2203.16828

"""

import ipdb
import random
import numpy as np
from copy import deepcopy
import math


#########################################
# Pure Functions
#########################################

def safe_crop(center_pt, crop_size, img_size):
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


#########################################
# Functions for Affine Transform
#########################################


def get_random_params_for_inverse_affine_matrix(degrees, translate, scale_ranges, shears, flip, img_size):
    """Get parameters for affine transformation

    Returns:
        sequence: params to be passed to the affine transformation
    """
    angle = random.uniform(-degrees, degrees)
    assert translate is None, "bugs here, figure out the xy axis and img size problem"
    if translate is not None:
        max_dx = translate[0] * img_size[0]
        max_dy = translate[1] * img_size[1]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))
    else:
        translations = (0, 0)

    if scale_ranges is not None:
        scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                    random.uniform(scale_ranges[0], scale_ranges[1]))
    else:
        scale = (1.0, 1.0)

    if shears is not None:
        shear = random.uniform(shears[0], shears[1])
    else:
        shear = 0.0

    if flip is not None:
        flip = 1 - (np.random.rand(2) < flip).astype(np.int) * 2
        flip = flip.tolist()
    else:
        flip = [1.0,1.0]

    return angle, translations, scale, shear, flip


def get_inverse_affine_matrix(center, angle, translate=None, scale=None, shear=None, flip=None):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    # RSS is rotation with scale and shear matrix
    # It is different from the original function in torchvision
    # The order are changed to flip -> scale -> rotation -> shear
    # x and y have different scale factors
    # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
    # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
    # [     0                       0                      1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    """
    Args:
        center (tuple(int,int))     : the center of the image, required
        angle (int)                 : angle for rotation, range [-360,360], required
        translate (tuple(int,int))  : default (0,0). Bugs here, so DON'T use it.
        scale (tuple(double,double)): default (1.,1.), scale for x and y axis
        shear (double)              : default 0.0
        flip (tuple(int,int))       : default no flip, choices [0 horizonal, 1 vertical]
    """
    # assertions, check param range
    if translate is not None:
        assert translate == (0,0), "BUG UNSOLVED"
    else:
        translate = (0,0)
    
    if scale is not None:
        assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
        "scale should be a list or tuple and it must be of length 2."
        for s in scale:
            if s <= 0:
                raise ValueError("scale values should be positive")
    else:
        scale = (1.,1.)
    
    if shear is None:
        shear = 0.0
    
    if flip is not None:
        assert isinstance(flip, (tuple, list)) and len(flip) == 2, \
        "flip should be a list or tuple and it must be of length 2."
        for f in flip:
            if f != -1 and f != 1:
                raise ValueError("flip values should be -1 or 1.")
    else:
        # final flip value -1, means flip
        # final flip value 1, means not flip
        # flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1
        # flip[0] horizonal
        # flip[1] vertical
        flip = (1.0,1.0)

    angle = math.radians(angle)
    shear = math.radians(shear)
    scale_x = 1.0 / scale[0] * flip[0]
    scale_y = 1.0 / scale[1] * flip[1]

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
    matrix = [
        math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
        -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
    ]
    matrix = [m / d for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]

    return matrix

