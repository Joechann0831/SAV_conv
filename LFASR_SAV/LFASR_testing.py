# @Time = 2022.07.15
# @Author = Zhen

"""
Code for testing LFASR networks.
"""

import argparse
import torch
from torch.utils.data import DataLoader
import scipy.io as sio

import os
import math

from model.LFASR_2_to_7_networks import LFASR_2_to_7_net
from model.LFASR_2_to_8_networks import LFASR_2_to_8_net
from model.LFASR_2_to_8_extra_networks import LFASR_2_to_8_extra_net

from dataloader.LFASR_dataloader import *
from dataloader.LFASR_dataloader_for_LFV import *

import utils.utils as utility
import numpy as np

from utils.logger import make_logs
import time

parser = argparse.ArgumentParser(description="Testing parameters for LFASR-SAV")
# for dataloader
parser.add_argument("--mode", type=str, default="inter28", help="The mode for LFASR, options: inter28, inter27, extra1, extra2")
parser.add_argument("--save-flag", action="store_true", help="Save the results? --save-flag makes it True")

def main():
    opt = parser.parse_args()
    print(opt)

    save_flag = opt.save_flag

    set_names = ["Scene30", "Reflective", "Occlusions", "EPFL", "General", "LFV"]

    if opt.mode == "inter28":

        mode = "inter28"
        block_nums = [16]
        block_modes = ["SAV_para"]
        fns = [45]
        ckpt_names = ["LFASR_SAV_Inter28_refine_16L_fn45"]
    elif opt.mode == 'inter27':
        mode = "inter27"
        block_nums = [16]
        block_modes = ["SAV_para"]
        fns = [45]
        ckpt_names = ["LFASR_SAV_Inter27_refine_16L_fn45"]
    elif opt.mode == 'extra1':
        mode = "extra1"
        block_nums = [8]
        block_modes = ["SAV_para"]
        fns = [45]
        ckpt_names = ["LFASR_SAV_ExtraI_28_refine_8L_fn45"]
    elif opt.mode == 'extra2':
        mode = "extra2"
        block_nums = [8]
        block_modes = ["SAV_para"]
        fns = [45]
        ckpt_names = ["LFASR_SAV_ExtraII_28_refine_8L_fn45"]
    else:
        raise Exception('Wrong mode!')

    for set_name in set_names:
        print("The testing set is: {} now".format(set_name))

        list_file = "./TestData/{}_list.txt".format(set_name)
        fd = open(list_file, 'r')
        name_list = [line.strip('\n') for line in fd.readlines()]

        for block_num, block_mode, fn, ckpt_name in zip(block_nums, block_modes, fns, ckpt_names):
            if mode == "inter28":
                model = LFASR_2_to_8_net(block_num=block_num,
                                         block_mode=block_mode,
                                         fn=fn)
            elif mode == "inter27":
                model = LFASR_2_to_7_net(block_num=block_num,
                                         block_mode=block_mode,
                                         fn=fn)
            elif mode == "extra1":
                model = LFASR_2_to_8_extra_net(block_num=block_num,
                                               block_mode=block_mode,
                                               fn=fn,
                                               extra_start=1)
            elif mode == "extra2":
                model = LFASR_2_to_8_extra_net(block_num=block_num,
                                               block_mode=block_mode,
                                               fn=fn,
                                               extra_start=2)
            else:
                raise Exception("Wrong mode!")

            n = sum(map(lambda x: x.numel(), model.parameters()))
            print("Network parameters: {}".format(n))

            save_dir = "./LFASR_results/{}/{}/{}/".format(mode, block_mode, set_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            model = model.cuda()
            model_path = "./checkpoint/{}/model_ckpt.pth".format(
                ckpt_name)
            model_dict = torch.load(model_path)["model_dict"]
            model.load_state_dict(model_dict)

            if set_name == "Scene30":
                test_file_path = "./TestData/Test_9x9.h5"
            elif set_name == 'LFV':
                test_file_path = "./TestData/Test_LFV_8x8.h5"
            else:
                test_file_path = "./TestData/Test_{}_9x9.h5".format(set_name)

            if mode == "inter28":
                if set_name == 'LFV':
                    test_set = ASR_2_8_test_LFV(file_path=test_file_path)
                else:
                    test_set = ASR_2_to_8_test_data(file_path=test_file_path)
                an_out = 8
            elif mode == "inter27":
                if set_name == 'LFV':
                    test_set = ASR_2_7_test_LFV(file_path=test_file_path)
                else:
                    test_set = ASR_2_to_7_test_data(file_path=test_file_path)
                an_out = 7
            elif mode == "extra1":
                if set_name == 'LFV':
                    test_set = ASR_2_8_extra_test_LFV(file_path=test_file_path, extra_start=1)
                else:
                    test_set = ASR_2_to_8_extra_test_data(file_path=test_file_path, extra_start=1)
                an_out = 8
            elif mode == "extra2":
                if set_name == 'LFV':
                    test_set = ASR_2_8_extra_test_LFV(file_path=test_file_path, extra_start=2)
                else:
                    test_set = ASR_2_to_8_extra_test_data(file_path=test_file_path, extra_start=2)
                an_out = 8
            else:
                raise Exception("Wrong mode!")

            testing_data_loader = DataLoader(dataset=test_set, num_workers=0,
                                             batch_size=1, shuffle=False)

            avg_psnr_coarse, avg_psnr_refine, PSNRs_coarse, PSNRs_refine = test(testing_data_loader,
                                                                                name_list,
                                                                                model,
                                                                                test_crop=0,
                                                                                cuda=True,
                                                                                save_flag=save_flag,
                                                                                save_dir=save_dir,
                                                                                mode=mode,
                                                                                an_out=an_out)
            #### You can use sio.savemat to save the PSNR values.

def test(testing_data_loader, name_list, model, test_crop, cuda, save_flag, save_dir, mode, an_out):
    model.eval()
    avg_psnr_coarse = 0.0
    avg_psnr_refine = 0.0

    PSNRs_coarse = np.zeros([len(testing_data_loader), 1])
    PSNRs_refine = np.zeros([len(testing_data_loader), 1])

    for iteration, (batch, lf_name) in enumerate(zip(testing_data_loader, name_list), 0):
        lf_input, lf_ori = batch[0], batch[1]

        if cuda:
            lf_input = lf_input.cuda()

        with torch.no_grad():
            if not test_crop:
                start_t = time.time()
                lf_recon_coarse, lf_recon_refine = model(lf_input)
                end_t = time.time()
                print("Elapsed time: {}".format(end_t - start_t))
                lf_recon_coarse = lf_recon_coarse.cpu().numpy()
                lf_recon_refine = lf_recon_refine.cpu().numpy()

            else:

                crop = test_crop
                length = 120
                lf_input_l, lf_input_m, lf_input_r = utility.CropPatches4D(lf_input, length, crop)

                lf_recon_coarse_l, lf_recon_refine_l = model(lf_input_l)
                lf_recon_coarse_l = lf_recon_coarse_l.cpu().numpy()
                lf_recon_refine_l = lf_recon_refine_l.cpu().numpy()

                lf_recon_coarse_m = np.zeros((lf_input_m.shape[0], 1, an_out, an_out, lf_input_m.shape[4], lf_input_m.shape[5]),
                                             dtype=np.float32)
                lf_recon_refine_m = np.zeros((lf_input_m.shape[0], 1, an_out, an_out, lf_input_m.shape[4], lf_input_m.shape[5]),
                                             dtype=np.float32)
                for i in range(lf_input_m.shape[0]):
                    lf_recon_coarse_mi, lf_recon_refine_mi = model(lf_input_m[i: i + 1])
                    lf_recon_coarse_m[i: i + 1] = lf_recon_coarse_mi.cpu().numpy()
                    lf_recon_refine_m[i: i + 1] = lf_recon_refine_mi.cpu().numpy()

                lf_recon_coarse_r, lf_recon_refine_r = model(lf_input_r)
                lf_recon_coarse_r = lf_recon_coarse_r.cpu().numpy()
                lf_recon_refine_r = lf_recon_refine_r.cpu().numpy()

                lf_recon_coarse = utility.MergePatches4D(lf_recon_coarse_l, lf_recon_coarse_m,
                                                         lf_recon_coarse_r, lf_input.shape[4], lf_input.shape[5],
                                                         length, crop)
                lf_recon_refine = utility.MergePatches4D(lf_recon_refine_l, lf_recon_refine_m,
                                                         lf_recon_refine_r, lf_input.shape[4], lf_input.shape[5],
                                                         length, crop)

        lf_ori = lf_ori.squeeze().numpy()
        lf_recon_coarse = lf_recon_coarse[0, 0]
        lf_recon_refine = lf_recon_refine[0, 0]

        # get to uint8
        lf_ori = utility.transfer_img_to_uint8(lf_ori)
        lf_recon_coarse = utility.transfer_img_to_uint8(lf_recon_coarse)
        lf_recon_refine = utility.transfer_img_to_uint8(lf_recon_refine)

        psnr_coarse_value = 0.0
        psnr_refine_value = 0.0

        if mode == "inter27":
            total_view_num = 45
            for u in range(7):
                for v in range(7):
                    if not ((u, v) == (0, 6) or (u, v) == (0, 0) or (u, v) == (6, 0) or (u, v) == (6, 6)):
                        psnr_coarse_value += utility.PSNR(lf_ori[u, v], lf_recon_coarse[u, v])
                        psnr_refine_value += utility.PSNR(lf_ori[u, v], lf_recon_refine[u, v])
        elif mode == "inter28":
            total_view_num = 60
            for u in range(8):
                for v in range(8):
                    if not ((u, v) == (0, 7) or (u, v) == (0, 0) or (u, v) == (7, 0) or (u, v) == (7, 7)):
                        psnr_coarse_value += utility.PSNR(lf_ori[u, v], lf_recon_coarse[u, v])
                        psnr_refine_value += utility.PSNR(lf_ori[u, v], lf_recon_refine[u, v])
        elif mode == "extra1":
            total_view_num = 60
            for u in range(8):
                for v in range(8):
                    if not ((u, v) == (1, 1) or (u, v) == (1, 6) or (u, v) == (6, 1) or (u, v) == (6, 6)):
                        psnr_coarse_value += utility.PSNR(lf_ori[u, v], lf_recon_coarse[u, v])
                        psnr_refine_value += utility.PSNR(lf_ori[u, v], lf_recon_refine[u, v])
        elif mode == "extra2":
            total_view_num = 60
            for u in range(8):
                for v in range(8):
                    if not ((u, v) == (2, 2) or (u, v) == (2, 5) or (u, v) == (5, 2) or (u, v) == (5, 5)):
                        psnr_coarse_value += utility.PSNR(lf_ori[u, v], lf_recon_coarse[u, v])
                        psnr_refine_value += utility.PSNR(lf_ori[u, v], lf_recon_refine[u, v])
        else:
            raise Exception("Wrong mode!")

        psnr_coarse_value /= total_view_num
        psnr_refine_value /= total_view_num

        avg_psnr_coarse += psnr_coarse_value
        avg_psnr_refine += psnr_refine_value

        PSNRs_coarse[iteration] = psnr_coarse_value
        PSNRs_refine[iteration] = psnr_refine_value

        print("{}: Coarse PSNR: {:.4f}, Refine PSNR: {:.4f}".format(lf_name, psnr_coarse_value, psnr_refine_value))
        if save_flag:
            sio.savemat("{}{}_res.mat".format(save_dir, lf_name), {"lf_recon_refine": lf_recon_refine,
                                                                   "lf_gt": lf_ori})
    avg_psnr_coarse /= len(testing_data_loader)
    avg_psnr_refine /= len(testing_data_loader)
    print("Avarege testing PSNR: Coarse {:.4f}, Refine {:.4f}".format(avg_psnr_coarse, avg_psnr_refine))

    return avg_psnr_coarse, avg_psnr_refine, PSNRs_coarse, PSNRs_refine

if __name__ == '__main__':
    main()