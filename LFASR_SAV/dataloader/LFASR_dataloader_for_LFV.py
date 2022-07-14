# @Time = 2021.03.10
# @Author = Zhen

"""
Dataloader for LFASR from LFV dataset.
"""

import torch
import torch.utils.data as data
import numpy as np
import random
import h5py
import tables
from utils.utils import *

class ASR_2_8_test_LFV(data.Dataset):
    def __init__(self, file_path):
        super(ASR_2_8_test_LFV, self).__init__()

        hf = h5py.File(file_path, 'r')
        self.lf_ori = hf["lf_gray"]

    def __getitem__(self, index):
        lf_ori = self.lf_ori[index] # [8,8,H,W]
        lf_input = lf_ori[::7, ::7, :, :] # [2,2,H,W]

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

class ASR_2_7_test_LFV(data.Dataset):
    def __init__(self, file_path):
        super(ASR_2_7_test_LFV, self).__init__()

        hf = h5py.File(file_path, 'r')
        self.lf_ori = hf["lf_gray"][:, 1:, 1:, :, :] # [N,7,7,H,W]

    def __getitem__(self, index):

        lf_ori = self.lf_ori[index] # [7,7,H,W]
        lf_input = lf_ori[::6, ::6, :, :] # [2,2,H,W]
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

class ASR_2_8_extra_test_LFV(data.Dataset):
    def __init__(self, file_path, extra_start):
        super(ASR_2_8_extra_test_LFV, self).__init__()

        hf = h5py.File(file_path, 'r')
        self.extra_start = extra_start
        self.lf_ori = hf["lf_gray"]

    def __getitem__(self, index):
        lf_ori = self.lf_ori[index] # [8,8,H,W]
        # get the lf input
        lf_input = lf_ori[self.extra_start::(7 - 2 * self.extra_start),
                   self.extra_start::(7 - 2 * self.extra_start), :, :]  # [2, 2, H, W]

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

