import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
from os.path import join
import utils.utils as utility
from model.model_utils import getNetworkDescription
import scipy.io as sio

import math

import h5py
import matplotlib

matplotlib.use('Agg')

from model.LFSSR import net2x, net4x

import warnings

warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------------------#
# Test settings
parser = argparse.ArgumentParser(description="Testing parameters for LFSSR-SAV")

parser.add_argument("--scale", type=int, default=4, help="SR factor")
parser.add_argument("--save-flag", action="store_true", help="Save the results? --save-flag makes it True")

# opt = parser.parse_args(args=[])
opt = parser.parse_args()
print(opt)

if opt.scale == 2:
    opt.model_dir = './checkpoints/LFSSR_SAV_x2_CityU/model_ckpt.pth'
    opt.layer_num = 16
    opt.angular_num = 7
    opt.fn = 45
    opt.mode = 'parares'
else:
    opt.model_dir = './checkpoints/LFSSR_SAV_x4_CityU/model_ckpt.pth'
    opt.layer_num = 10
    opt.angular_num = 7
    opt.fn = 45
    opt.mode = 'parares'

# -----------------------------------------------------------------------------------#
class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, scale):
        super(DatasetFromHdf5, self).__init__()

        hf = h5py.File(file_path)
        self.GT_y = hf["/GT_y"]  # [N,aw,ah,h,w]
        self.LR_ycbcr = hf["/LR_ycbcr"]  # [N,ah,aw,3,h/s,w/s]

        self.scale = scale

    def __getitem__(self, index):
        h = self.GT_y.shape[3]
        w = self.GT_y.shape[4]

        gt_y = self.GT_y[index]
        gt_y = gt_y.reshape(-1, h, w)
        gt_y = torch.from_numpy(gt_y.astype(np.float32) / 255.0)

        lr_ycbcr = self.LR_ycbcr[index]
        lr_ycbcr = torch.from_numpy(lr_ycbcr.astype(np.float32) / 255.0)

        lr_y = lr_ycbcr[:, :, 0, :, :].clone().view(-1, h // self.scale, w // self.scale)

        lr_ycbcr_up = lr_ycbcr.view(1, -1, h // self.scale, w // self.scale)
        lr_ycbcr_up = torch.nn.functional.interpolate(lr_ycbcr_up, scale_factor=self.scale, mode='bilinear',
                                                      align_corners=False)
        lr_ycbcr_up = lr_ycbcr_up.view(-1, 3, h, w)

        return gt_y, lr_ycbcr_up, lr_y

    def __len__(self):
        return self.GT_y.shape[0]

# -----------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#

model_dir = opt.model_dir

if not os.path.exists(model_dir):
    print('model folder is not found ')

an = opt.angular_num
# ------------------------------------------------------------------------#
# Data loader


def test_dataset(dataset_name):
    print('===> Loading test datasets')

    data_path = join('./TestData/CityU/test_{}_x{}.h5'.format(dataset_name, opt.scale))
    test_set = DatasetFromHdf5(data_path, opt.scale)

    test_list_file = './TestData/CityU/test_lists/test_{}_list.txt'.format(dataset_name)
    fd = open(test_list_file, 'r')
    name_list = [line.strip('\n') for line in fd.readlines()]

    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print('loaded {} LFIs from {}'.format(len(test_loader), data_path))
    # -------------------------------------------------------------------------#
    # Build model
    print("===> building network")
    srnet_name = 'net{}x'.format(opt.scale)
    model = eval(srnet_name)(an, opt.layer_num, opt.mode, opt.fn).to(device)
    s, n = getNetworkDescription(model)
    # print("Network structure:")
    # print(s)
    print("Network parameter number: {}".format(n))

    # ------------------------------------------------------------------------#

    # -------------------------------------------------------------------------#
    # test


    def compt_psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 1.0

        if mse > 1000:
            return -100
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


    def test():
        model.eval()
        lf_list = []
        lf_psnr_y_list = []

        with torch.no_grad():
            for k, batch in enumerate(test_loader):

                lfname = name_list[k]
                print('testing LF {}-{}'.format(dataset_name, lfname))

                save_dir_mat = './LFSSR_results/CityU_7x7/{}xmat/{}'.format(opt.scale, dataset_name)
                if not os.path.exists(save_dir_mat):
                    os.makedirs(save_dir_mat)
                # ----------- SR ---------------#
                gt_y, sr_ycbcr, lr_y = batch[0].numpy(), batch[1].numpy(), batch[2]

                lr_y = lr_y.to(device)

                opt.test_patch = [1, 1]

                if opt.test_patch[0] == 1 and opt.test_patch[1] == 1:
                    try:
                        sr_y = model(lr_y)[0]
                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            print("WARNING: out of memory, clear the cache!")
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                                sr_y = model(lr_y)[0]
                        else:
                            raise exception
                    sr_y = sr_y.cpu().numpy()
                else:
                    Px = opt.test_patch[0]
                    Py = opt.test_patch[1]
                    if opt.scale == 2:
                        pad_size = 32
                    else:
                        pad_size = 16
                    H, W = lr_y.shape[2], lr_y.shape[3]

                    srLF_patches = []

                    for px in range(Px):
                        for py in range(Py):
                            lr_y_patch = utility.getLFPatch(lr_y, Px, Py, H, W, px, py, pad_size)
                            try:
                                sr_y_patch = model(lr_y_patch)[0]
                            except RuntimeError as exception:
                                if "out of memory" in str(exception):
                                    print("WARNING: out of memory, clear the cache!")
                                    if hasattr(torch.cuda, 'empty_cache'):
                                        torch.cuda.empty_cache()
                                        sr_y_patch = model(lr_y_patch)[0]
                                else:
                                    raise exception
                            sr_y_patch = sr_y_patch.cpu().numpy()
                            srLF_patches.append(sr_y_patch)
                    sr_y = utility.mergeLFPatches(srLF_patches, Px, Py, H, W, opt.scale, pad_size)
                if opt.save_flag:
                    sio.savemat('{}/{}_sr_y.mat'.format(save_dir_mat, lfname), {'sr_y': (sr_y * 255.0).astype(np.uint8)})

                sr_ycbcr[:, :, 0] = sr_y
                # ---------compute average PSNR for this LFI----------#

                view_list = []
                view_psnr_y_list = []

                for i in range(an * an):
                    cur_psnr = compt_psnr(gt_y[0, i], sr_y[0, i])
                    view_list.append(i)
                    view_psnr_y_list.append(cur_psnr)


                lf_list.append(k)
                lf_psnr_y_list.append(np.mean(view_psnr_y_list))

                print(
                    'Avg. Y PSNR: {:.2f}'.format(np.mean(view_psnr_y_list)))

        print('Over all {} LFIs on {}: Avg. Y PSNR: {:.2f}'.format(len(test_loader), dataset_name,
                                                                                        np.mean(lf_psnr_y_list)))


    print('===> Test scale {}'.format(opt.scale))
    resume_path = model_dir
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model'])
    print('loaded model {}'.format(resume_path))
    test()


dataset_names = ['Kalantari', 'InriaSynthetic', 'HCI', 'Stanford_General', 'Stanford_Occlusions']

for datasetName in dataset_names:
    test_dataset(datasetName)