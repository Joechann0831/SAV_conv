# @Time = 2019.12.10
# @Author = Zhen

"""
    utils.py is used to define useful functions
"""

import math
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
# from code_for_models.LFVDSR_config import bicubic_imresize, bicubic_interpolation
import os
import h5py
from scipy.signal import convolve2d
import argparse
# import torch.nn.functional as F

def get_cur_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def load_params(src_dict, des_model):
    # src_dict: state dict of source model
    # des_model: model of the destination model
    des_dict = des_model.state_dict()
    for_des = {k: v for k, v in src_dict.items() if k in des_dict.keys()}
    des_dict.update(for_des)
    des_model.load_state_dict(des_dict)
    return des_model

class train_data_args():
    file_path = ""
    patch_size = 96
    random_flip_vertical = True
    random_flip_horizontal = True
    random_rotation = True

def get_time_gpu():
    torch.cuda.synchronize()
    return time.time()

def get_time_gpu_str():
    torch.cuda.synchronize()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def CropPatches(image, len, crop):
    # left [1,an2,h,lw]
    # middles[n,an2,h,mw]
    # right [1,an2,h,rw]
    an, h, w = image.shape[1:4]
    left = image[:, :, :, 0:len + crop]
    num = math.floor((w - len - crop) / len)
    middles = torch.Tensor(num, an, h, len + crop * 2).to(image.device)
    for i in range(num):
        middles[i] = image[0, :, :, (i + 1) * len - crop:(i + 2) * len + crop]
    right = image[:, :, :, -(len + crop):]
    return left, middles, right

def shaveLF(inLF, border=(3, 3)):
    """
    Shave the input light field in terms of a given border.

    :param inLF:   input light field of size: [U, V, H, W]
    :param border: border values

    :return:       shaved light field
    """
    h_border, w_border = border
    if (h_border != 0) and (w_border != 0):
        shavedLF = inLF[:, :, h_border:-h_border, w_border:-w_border]
    elif (h_border != 0) and (w_border == 0):
        shavedLF = inLF[:, :, h_border:-h_border, :]
    elif (h_border == 0) and (w_border != 0):
        shavedLF = inLF[:, :, :, w_border:-w_border]
    else:
        shavedLF = inLF
    return shavedLF

def CropPatches4D(image, len, crop):
    # left [1,1,u,v,h,lw]
    # middles[n,1,u,v,h,mw]
    # right [1,1,u,v,h,rw]
    u, v, h, w = image.shape[2:]
    left = image[:, :, :, :, :, 0:len + crop]
    num = math.floor((w - len - crop) / len)
    middles = torch.Tensor(num, 1, u, v, h, len + crop * 2).to(image.device)
    for i in range(num):
        middles[i] = image[0, :, :, :, :, (i + 1) * len - crop:(i + 2) * len + crop]
    right = image[:, :, :, :, :, -(len + crop):]
    return left, middles, right

def MergePatches2D(left, middles, right, h, w, len, crop):
    out = np.zeros((h, w)).astype(left.dtype)
    out[:, :len] = left[:, :-crop]
    for i in range(middles.shape[0]):
        out[:, len * (i + 1): len*(i + 2)] = middles[i:i+1, :, crop:-crop]
    out[:, -len:] = right[:, crop:]
    return out

def MergePatches(left, middles, right, h, w, len, crop):
    n, a = left.shape[0:2]
    # out = torch.Tensor(n, a, h, w).to(left.device)
    out = np.zeros((n,a,h,w)).astype(left.dtype)
    out[:, :, :, :len] = left[:, :, :, :-crop]
    for i in range(middles.shape[0]):
        out[:, :, :, len * (i + 1):len * (i + 2)] = middles[i:i + 1, :, :, crop:-crop]
    out[:, :, :, -len:] = right[:, :, :, crop:]
    return out

def MergePatches4D(left, middles, right, h, w, len, crop):
    n, u, v = left.shape[0], left.shape[2], left.shape[3]
    out = np.zeros((n,1,u,v,h,w)).astype(left.dtype)
    out[:, :, :, :, :, :len] = left[:, :, :, :, :, :-crop]
    for i in range(middles.shape[0]):
        out[:, :, :, :, :, len * (i + 1):len * (i + 2)] = middles[i:i + 1, :, :, :, :, crop:-crop]
    out[:, :, :, :, :, -len:] = right[:, :, :, :, :, crop:]
    return out

def PSNR(pred, gt, shave_border=0):
# define PSNR function, the input images should be inside the interval of [0,255]
    pred = pred.astype(float)
    gt = gt.astype(float)
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def transfer_img_to_uint8(img):
    # the input image is within (0,1) interval
    img = img * 255.0
    img = np.clip(img, 0.0, 255.0)
    img = np.uint8(np.around(img))
    return img

def colorize(y, ycbcr):
# colorize a grayscale image
# ycbcr means the upscaled YCbCr using Bicubic interpolation
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

def modcrop(imgs, scale):
# modcrop the input image, the input image is a matrix with 1 or 3 channels
    if len(imgs.shape) == 2:
        img_row = imgs.shape[0]
        img_col = imgs.shape[1]
        cropped_row = img_row - img_row % scale
        cropped_col = img_col - img_col % scale
        cropped_img = imgs[:cropped_row, :cropped_col]
    elif len(imgs.shape) == 3:
        img_row = imgs.shape[0]
        img_col = imgs.shape[1]
        cropped_row = img_row - img_row % scale
        cropped_col = img_col - img_col % scale
        cropped_img = imgs[:cropped_row, :cropped_col, :]
    else:
        raise IOError('Img Channel > 3.')

    return cropped_img

def lf_modcrop(lf, scale):
    # modcrop the input light field, the light field shoud be as format as [U,V,X,Y]
    [U, V, X, Y] = lf.shape
    x = X - (X % scale)
    y = Y - (Y % scale)
    output = np.zeros([U, V, x, y])
    for u in range(0, U):
        for v in range(0, V):
            sub_img = lf[u,v]
            output[u,v] = modcrop(sub_img, scale)
    return output


def img_rgb2ycbcr(img):
    # the input image data format should be uint8
    if not len(img.shape) == 3:
        raise IOError('Img channle is not 3')
    if not img.dtype == 'uint8':
        raise IOError('Img should be uint8')
    img = img/255.0
    img_ycbcr = np.zeros(img.shape, 'double')
    img_ycbcr[:, :, 0] = 65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2] + 16
    img_ycbcr[:, :, 1] = -37.797 * img[:, :, 0] - 74.203 * img[:, :, 1] + 112 * img[:, :, 2] + 128
    img_ycbcr[:, :, 2] = 112 * img[:, :, 0] - 93.786 * img[:, :, 1] - 18.214 * img[:, :, 2] + 128
    img_ycbcr = np.round(img_ycbcr)
    img_ycbcr = np.clip(img_ycbcr,0,255)
    img_ycbcr = np.uint8(img_ycbcr)
    return img_ycbcr

def img_ycbcr2rgb(im):
    # the input image data format should be uint8
    if not len(im.shape) == 3:
        raise IOError('Img channle is not 3')
    if not im.dtype == 'uint8':
        raise IOError('Img should be uint8')
    im_YCrCb = np.zeros(im.shape, 'double')
    im_YCrCb = im * 1.0
    tmp = np.zeros(im.shape, 'double')
    tmp[:, :, 0] = im_YCrCb[:, :, 0] - 16.0
    tmp[:, :, 1] = im_YCrCb[:, :, 1] - 128.0
    tmp[:, :, 2] = im_YCrCb[:, :, 2] - 128.0
    im_my = np.zeros(im.shape, 'double')
    im_my[:, :, 0] = 0.00456621 * tmp[:, :, 0] + 0.00625893 * tmp[:, :, 2]
    im_my[:, :, 1] = 0.00456621 * tmp[:, :, 0] - 0.00153632 * tmp[:, :, 1] - 0.00318811 * tmp[:, :, 2]
    im_my[:, :, 2] = 0.00456621 * tmp[:, :, 0] + 0.00791071 * tmp[:, :, 1]
    im_my = im_my * 255
    im_my = np.round(im_my)
    im_my = np.clip(im_my, 0, 255)
    im_my = np.uint8(im_my)
    return im_my
