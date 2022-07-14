# @Time = 2020.12.16
# @Author = Zhen

"""
Dataloader for light field angular super-resolution.
"""

import torch
import torch.utils.data as data
import numpy as np
import random
import h5py
import tables
from utils.utils import *

class ASR_2_to_8_train_data(data.Dataset):
    def __init__(self, args):
        # args: file_path,
        #       patch_size=64,
        #       random_flip_vertical=True,
        #       random_flip_horizontal=True,
        #       random_rotation=True
        super(ASR_2_to_8_train_data, self).__init__()

        self.args = args
        hf = h5py.File(self.args.file_path, 'r')
        # self.lf_ori = hf["lf_gray"][:, 3:11, 3:11, :, :] # [N, 8, 8, H, W]
        self.lf_ori = hf["lf_gray"][:, :8, :8, :, :]  # [N, 8, 8, H, W]
        # self.lf_ori = self.lf_ori.astype(np.float32) / 255.0

    def __getitem__(self, index):
        lf_ori = self.lf_ori[index] # [8, 8, H, W]

        # Get random lf patch (random crop)
        H, W = lf_ori.shape[2:]

        x = random.randrange(0, H - self.args.patch_size)
        y = random.randrange(0, W - self.args.patch_size)

        lf_ori = lf_ori[:, :, x: x + self.args.patch_size, y: y + self.args.patch_size]

        # 4D augmentation
        if self.args.random_flip_vertical and np.random.rand(1) > 0.5:
            lf_ori = np.flip(np.flip(lf_ori, 0), 2)
        if self.args.random_flip_horizontal and np.random.rand(1) > 0.5:
            lf_ori = np.flip(np.flip(lf_ori, 1), 3)
        if self.args.random_rotation:
            r_ang = np.random.randint(1, 5)
            lf_ori = np.rot90(lf_ori, r_ang, (2, 3))
            lf_ori = np.rot90(lf_ori, r_ang, (0, 1))

        # get the lf input
        lf_input = lf_ori[::7, ::7, :, :]# [2, 2, H, W]

        # transfer to tensor
        lf_input = np.expand_dims(lf_input, axis=0)
        lf_ori = np.expand_dims(lf_ori, axis=0)

        lf_input = lf_input.astype(np.float32) / 255.0
        lf_ori = lf_ori.astype(np.float32) / 255.0

        lf_input = torch.Tensor(lf_input)
        lf_ori = torch.Tensor(lf_ori)
        return lf_input, lf_ori

    def __len__(self):
        return self.lf_ori.shape[0]

class ASR_2_to_8_test_data(data.Dataset):
    def __init__(self, file_path):
        super(ASR_2_to_8_test_data, self).__init__()

        hf = h5py.File(file_path, 'r')
        # self.lf_ori = hf["lf_gray"][:, 3:11, 3:11, :, :] # [N, 8, 8, H, W]
        self.lf_ori = hf["lf_gray"][:, :8, :8, :, :]  # [N, 8, 8, H, W]
        # self.lf_ori = self.lf_ori.astype(np.float32) / 255.0

    def __getitem__(self, index):
        lf_ori = self.lf_ori[index] # [8, 8, H, W]
        # get the lf input
        lf_input = lf_ori[::7, ::7, :, :]# [2, 2, H, W]

        # transfer to tensor
        lf_input = np.expand_dims(lf_input, axis=0)
        lf_ori = np.expand_dims(lf_ori, axis=0)

        lf_input = lf_input.astype(np.float32) / 255.0
        lf_ori = lf_ori.astype(np.float32) / 255.0

        lf_input = torch.Tensor(lf_input)
        lf_ori = torch.Tensor(lf_ori)
        return lf_input, lf_ori

    def __len__(self):
        return self.lf_ori.shape[0]

class ASR_2_to_8_extra_train_data(data.Dataset):
    def __init__(self, args, extra_start=1):
        # args: file_path,
        #       patch_size=64,
        #       random_flip_vertical=True,
        #       random_flip_horizontal=True,
        #       random_rotation=True
        super(ASR_2_to_8_extra_train_data, self).__init__()

        self.args = args
        self.extra_start = extra_start
        hf = h5py.File(self.args.file_path, 'r')
        # self.lf_ori = hf["lf_gray"][:, 3:11, 3:11, :, :] # [N, 8, 8, H, W]
        self.lf_ori = hf["lf_gray"][:, :8, :8, :, :]  # [N, 8, 8, H, W]
        # self.lf_ori = self.lf_ori.astype(np.float32) / 255.0

    def __getitem__(self, index):
        lf_ori = self.lf_ori[index] # [8, 8, H, W]

        # Get random lf patch (random crop)
        H, W = lf_ori.shape[2:]

        x = random.randrange(0, H - self.args.patch_size)
        y = random.randrange(0, W - self.args.patch_size)

        lf_ori = lf_ori[:, :, x: x + self.args.patch_size, y: y + self.args.patch_size]

        # 4D augmentation
        if self.args.random_flip_vertical and np.random.rand(1) > 0.5:
            lf_ori = np.flip(np.flip(lf_ori, 0), 2)
        if self.args.random_flip_horizontal and np.random.rand(1) > 0.5:
            lf_ori = np.flip(np.flip(lf_ori, 1), 3)
        if self.args.random_rotation:
            r_ang = np.random.randint(1, 5)
            lf_ori = np.rot90(lf_ori, r_ang, (2, 3))
            lf_ori = np.rot90(lf_ori, r_ang, (0, 1))

        # get the lf input
        lf_input = lf_ori[self.extra_start::(7 - 2*self.extra_start),
                   self.extra_start::(7 - 2*self.extra_start), :, :]# [2, 2, H, W]

        # transfer to tensor
        lf_input = np.expand_dims(lf_input, axis=0)
        lf_ori = np.expand_dims(lf_ori, axis=0)

        lf_input = lf_input.astype(np.float32) / 255.0
        lf_ori = lf_ori.astype(np.float32) / 255.0

        lf_input = torch.Tensor(lf_input)
        lf_ori = torch.Tensor(lf_ori)
        return lf_input, lf_ori

    def __len__(self):
        return self.lf_ori.shape[0]

class ASR_2_to_8_extra_test_data(data.Dataset):
    def __init__(self, file_path, extra_start):
        super(ASR_2_to_8_extra_test_data, self).__init__()

        hf = h5py.File(file_path, 'r')
        self.extra_start = extra_start
        # self.lf_ori = hf["lf_gray"][:, 3:11, 3:11, :, :] # [N, 8, 8, H, W]
        self.lf_ori = hf["lf_gray"][:, :8, :8, :, :]  # [N, 8, 8, H, W]
        # self.lf_ori = self.lf_ori.astype(np.float32) / 255.0

    def __getitem__(self, index):
        lf_ori = self.lf_ori[index] # [8, 8, H, W]
        # get the lf input
        lf_input = lf_ori[self.extra_start::(7 - 2*self.extra_start),
                   self.extra_start::(7 - 2*self.extra_start), :, :]# [2, 2, H, W]

        # transfer to tensor
        lf_input = np.expand_dims(lf_input, axis=0)
        lf_ori = np.expand_dims(lf_ori, axis=0)

        lf_input = lf_input.astype(np.float32) / 255.0
        lf_ori = lf_ori.astype(np.float32) / 255.0

        lf_input = torch.Tensor(lf_input)
        lf_ori = torch.Tensor(lf_ori)
        return lf_input, lf_ori

    def __len__(self):
        return self.lf_ori.shape[0]

class ASR_2_to_7_train_data(data.Dataset):
    def __init__(self, args):
        # args: file_path,
        #       patch_size=64,
        #       random_flip_vertical=True,
        #       random_flip_horizontal=True,
        #       random_rotation=True
        super(ASR_2_to_7_train_data, self).__init__()

        self.args = args
        hf = h5py.File(self.args.file_path, 'r')
        self.lf_ori = hf["lf_gray"][:, 1:8, 1:8, :, :]  # [N, 7, 7, H, W]
        # self.lf_ori = self.lf_ori.astype(np.float32) / 255.0

    def __getitem__(self, index):
        lf_ori = self.lf_ori[index] # [7, 7, H, W]

        # Get random lf patch (random crop)
        H, W = lf_ori.shape[2:]

        x = random.randrange(0, H - self.args.patch_size)
        y = random.randrange(0, W - self.args.patch_size)

        lf_ori = lf_ori[:, :, x: x + self.args.patch_size, y: y + self.args.patch_size]

        # 4D augmentation
        if self.args.random_flip_vertical and np.random.rand(1) > 0.5:
            lf_ori = np.flip(np.flip(lf_ori, 0), 2)
        if self.args.random_flip_horizontal and np.random.rand(1) > 0.5:
            lf_ori = np.flip(np.flip(lf_ori, 1), 3)
        if self.args.random_rotation:
            r_ang = np.random.randint(1, 5)
            lf_ori = np.rot90(lf_ori, r_ang, (2, 3))
            lf_ori = np.rot90(lf_ori, r_ang, (0, 1))

        # get the lf input
        lf_input = lf_ori[::6, ::6, :, :]# [2, 2, H, W]

        # transfer to tensor
        lf_input = np.expand_dims(lf_input, axis=0)
        lf_ori = np.expand_dims(lf_ori, axis=0)

        lf_input = lf_input.astype(np.float32) / 255.0
        lf_ori = lf_ori.astype(np.float32) / 255.0

        lf_input = torch.Tensor(lf_input)
        lf_ori = torch.Tensor(lf_ori)
        return lf_input, lf_ori

    def __len__(self):
        return self.lf_ori.shape[0]

class ASR_2_to_7_test_data(data.Dataset):
    def __init__(self, file_path):
        super(ASR_2_to_7_test_data, self).__init__()

        hf = h5py.File(file_path, 'r')
        self.lf_ori = hf["lf_gray"][:, 1:8, 1:8, :, :]  # [N, 7, 7, H, W]
        # self.lf_ori = self.lf_ori.astype(np.float32) / 255.0

    def __getitem__(self, index):
        lf_ori = self.lf_ori[index] # [7, 7, H, W]
        # get the lf input
        lf_input = lf_ori[::6, ::6, :, :]# [2, 2, H, W]

        # transfer to tensor
        lf_input = np.expand_dims(lf_input, axis=0)
        lf_ori = np.expand_dims(lf_ori, axis=0)

        lf_input = lf_input.astype(np.float32) / 255.0
        lf_ori = lf_ori.astype(np.float32) / 255.0

        lf_input = torch.Tensor(lf_input)
        lf_ori = torch.Tensor(lf_ori)
        return lf_input, lf_ori

    def __len__(self):
        return self.lf_ori.shape[0]
