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
        # args: list_path,
        #       patch_size=64,
        #       random_flip_vertical=True,
        #       random_flip_horizontal=True,
        #       random_rotation=True
        super(ASR_2_to_8_train_data, self).__init__()

        self.args = args

        fd = open(self.args.list_path, 'r')
        self.h5files = [line.strip('\n') for line in fd.readlines()]
        print("Dataset files include {}".format(self.h5files))

        self.lens = []
        self.lf_oris = []

        for i in range(len(self.h5files)):
            # hf = tables.open_file(self.h5files[i], driver="H5FD_CORE")
            # lf_ori = hf.root.lf_gray[:, :8, :8, :, :]

            hf = h5py.File(self.h5files[i])
            lf_ori = hf["lf_gray"][:, :8, :8, :, :]

            self.lf_oris.append(lf_ori)
            self.lens.append(lf_ori.shape[0])

    def __getitem__(self, index):

        file_index = 0
        batch_index = 0
        for i in range(len(self.h5files)):
            if index < self.lens[i]:
                file_index = i
                batch_index = index
                break
            else:
                index -= self.lens[i]

        lf_ori = self.lf_oris[file_index][batch_index, :, :, :, :] # [8, 8, H, W]

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
        total_len = 0
        for i in range(len(self.h5files)):
            total_len += self.lens[i]
        return total_len

class ASR_2_to_8_test_data(data.Dataset):
    def __init__(self, file_path):
        super(ASR_2_to_8_test_data, self).__init__()

        hf = h5py.File(file_path)
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



class ASR_5_to_9_train_data(data.Dataset):
    def __init__(self, args):
        # args: list_path,
        #       patch_size=64,
        #       random_flip_vertical=True,
        #       random_flip_horizontal=True,
        #       random_rotation=True
        super(ASR_5_to_9_train_data, self).__init__()

        self.args = args
        fd = open(self.args.list_path, 'r')
        self.h5files = [line.strip('\n') for line in fd.readlines()]
        print("Dataset files include {}".format(self.h5files))

        self.lens = []
        self.lf_oris = []

        for i in range(len(self.h5files)):
            # hf = tables.open_file(self.h5files[i], driver="H5FD_CORE")
            # lf_ori = hf.root.lf_gray

            hf = h5py.File(self.h5files[i])
            lf_ori = hf["lf_gray"] # [9,9,H,W]
            self.lf_oris.append(lf_ori)
            self.lens.append(lf_ori.shape[0])

    def __getitem__(self, index):

        file_index = 0
        batch_index = 0
        for i in range(len(self.h5files)):
            if index < self.lens[i]:
                file_index = i
                batch_index = index
                break
            else:
                index -= self.lens[i]

        lf_ori = self.lf_oris[file_index][batch_index]  # [9, 9, H, W]

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
        lf_input = lf_ori[::2, ::2, :, :]# [5, 5, H, W]

        # transfer to tensor
        lf_input = np.expand_dims(lf_input, axis=0)
        lf_ori = np.expand_dims(lf_ori, axis=0)

        lf_input = lf_input.astype(np.float32) / 255.0
        lf_ori = lf_ori.astype(np.float32) / 255.0

        lf_input = torch.Tensor(lf_input)
        lf_ori = torch.Tensor(lf_ori)
        return lf_input, lf_ori

    def __len__(self):
        total_len = 0
        for i in range(len(self.h5files)):
            total_len += self.lens[i]
        return total_len

class ASR_5_to_9_test_data(data.Dataset):
    def __init__(self, file_path):
        super(ASR_5_to_9_test_data, self).__init__()

        hf = h5py.File(file_path)
        self.lf_ori = hf["lf_gray"] # [N, 9, 9, H, W]
        # self.lf_ori = self.lf_ori.astype(np.float32) / 255.0

    def __getitem__(self, index):
        lf_ori = self.lf_ori[index] # [9, 9, H, W]
        # get the lf input
        lf_input = lf_ori[::2, ::2, :, :]# [5, 5, H, W]

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

