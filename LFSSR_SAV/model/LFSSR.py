# @Time = 2021.12.23
# @Author = Zhen

"""
Spatial SR for light fields.
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from utils.convNd import convNd
# from model.model_utils import *
from model.model_utils import *

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)

    return torch.from_numpy(filter).float()


class net2x(nn.Module):
    def __init__(self, an, layer, mode="catres", fn=64):

        super(net2x, self).__init__()

        self.an = an
        self.an2 = an * an
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=fn, kernel_size=3, stride=1, padding=1)

        #
        if mode == "catres":
            sas_para = SAS_para()
            sac_para = SAC_para()
            sas_para.fn = fn
            sac_para.fn = fn
            alt_blocks = [SAV_concat(SAS_para=sas_para, SAC_para=sac_para) for _ in range(layer)]
        elif mode == 'cat':
            sas_para = SAS_para()
            sac_para = SAC_para()
            sas_para.fn = fn
            sac_para.fn = fn
            alt_blocks = [SAV_concat(SAS_para=sas_para, SAC_para=sac_para, residual_connection=False) for _ in range(layer)]
        elif mode == 'Dserial':
            sas_para = SAS_para()
            sac_para = SAC_para()
            sas_para.fn = fn
            sac_para.fn = fn
            alt_blocks = [SAV_double_serial(SAS_para=sas_para, SAC_para=sac_para) for _ in range(layer)]
        elif mode == "parares":
            sas_para = SAS_para()
            sac_para = SAC_para()
            sas_para.fn = fn
            sac_para.fn = fn
            alt_blocks = [SAV_parallel(SAS_para=sas_para, SAC_para=sac_para, feature_concat=False) for _ in
                          range(layer)]
        elif mode == "SAS":
            alt_blocks = [SAS_conv(act='lrelu', fn=fn) for _ in range(layer)]
        elif mode == "SAC":
            alt_blocks = [SAC_conv(act='lrelu', fn=fn) for _ in range(layer)]
        else:
            raise Exception("Wrong mode!")
        self.refine_sas = nn.Sequential(*alt_blocks)

        self.fup1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=fn, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.res1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_in', 'leaky_relu')
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, lr):

        N, dimsize, h, w = lr.shape  # lr [N,81,h,w]
        lr = lr.view(N * self.an2, 1, h, w)  # [N*81,1,h,w]

        x = self.lrelu(self.conv0(lr))  # [N*81,64,h,w]

        ## CS-MOD
        lf_out = x.view(N, self.an2, -1, h, w)  # [N, an2, C, H, W]
        lf_out = lf_out.permute([0, 2, 1, 3, 4]).contiguous()  # [N, C, an2, H, W]
        lf_out = lf_out.view(N, lf_out.shape[1], self.an, self.an, h, w)  # [N, C, an, an, H, W]
        lf_out = self.refine_sas(lf_out)  # [N*81,64,h,w]
        lf_out = lf_out.view(N, -1, self.an2, h, w)  # [N, C, an2, H, W]
        lf_out = lf_out.permute([0, 2, 1, 3, 4]).contiguous()  # [N, an2, C, H, W]
        lf_out = lf_out.view(N * self.an2, lf_out.shape[2], h, w)

        fup_1 = self.fup1(lf_out)  # [N*81,64,2h,2w]
        res_1 = self.res1(fup_1)  # [N*81,1,2h,2w]
        iup_1 = self.iup1(lr)  # [N*81,1,2h,2w]

        sr_2x = res_1 + iup_1  # [N*81,1,2h,2w]
        sr_2x = sr_2x.view(N, self.an2, h * 2, w * 2)
        return [sr_2x]

class net4x(nn.Module):
    def __init__(self, an, layer, mode="catres", fn=64):
        super(net4x, self).__init__()

        self.an = an
        self.an2 = an * an
        self.fn = fn
        self.mode = mode
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=fn, kernel_size=3, stride=1, padding=1)

        self.altblock1 = self.make_layer(layer_num=layer, fn=fn)

        self.fup1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=fn, out_channels=fn, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.res1 = nn.Conv2d(in_channels=fn, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.altblock2 = self.make_layer(layer_num=layer, fn=fn)

        self.fup2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=fn, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.res2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.iup2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_in', 'leaky_relu')
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, layer_num=6, fn=64):

        layers = []
        sas_para = SAS_para()
        sac_para = SAC_para()
        sas_para.fn = fn
        sac_para.fn = fn
        if self.mode == "catres":
            for i in range(layer_num):
                layers.append(SAV_concat(SAS_para=sas_para, SAC_para=sac_para))
        elif self.mode == "parares":
            for i in range(layer_num):
                layers.append(SAV_parallel(SAS_para=sas_para, SAC_para=sac_para, feature_concat=False))
        elif self.mode == "Dserial":
            for i in range(layer_num):
                layers.append(SAV_double_serial(SAS_para=sas_para, SAC_para=sac_para))
        elif self.mode == "SAS":
            for i in range(layer_num):
                layers.append(SAS_conv(act='lrelu', fn=fn))
        elif self.mode == "SAC":
            for i in range(layer_num):
                layers.append(SAC_conv(act='lrelu', fn=fn))
        return nn.Sequential(*layers)

    def forward(self, lr):

        N, _, h, w = lr.shape  # lr [N,81,h,w]
        lr = lr.view(N * self.an2, 1, h, w)  # [N*81,1,h,w]

        x = self.lrelu(self.conv0(lr))  # [N*81,64,h,w]
        #########
        lf_out = x.view(N, self.an2, -1, h, w)  # [N, an2, C, H, W]
        lf_out = lf_out.permute([0, 2, 1, 3, 4]).contiguous()  # [N, C, an2, H, W]
        lf_out = lf_out.view(N, lf_out.shape[1], self.an, self.an, h, w)  # [N, C, an, an, H, W]
        lf_out = self.altblock1(lf_out)  # [N*81,64,h,w]
        lf_out = lf_out.view(N, -1, self.an2, h, w)  # [N, C, an2, H, W]
        lf_out = lf_out.permute([0, 2, 1, 3, 4]).contiguous()  # [N, an2, C, H, W]
        lf_out = lf_out.view(N * self.an2, lf_out.shape[2], h, w)
        #######
        fup_1 = self.fup1(lf_out)  # [N*81,64,2h,2w]
        res_1 = self.res1(fup_1)  # [N*81,1,2h,2w]
        iup_1 = self.iup1(lr)  # [N*81,1,2h,2w]

        sr_2x = res_1 + iup_1  # [N*81,1,2h,2w]
        ##########
        f_2 = fup_1.view(N, self.an2, -1, 2 * h, 2 * w)  # [N, an2, C, H, W]
        f_2 = f_2.permute([0, 2, 1, 3, 4]).contiguous()  # [N, C, an2, H, W]
        f_2 = f_2.view(N, f_2.shape[1], self.an, self.an, 2 * h, 2 * w)  # [N, C, an, an, H, W]
        f_2 = self.altblock2(f_2)  # [N*81,64,2h,2w]
        f_2 = f_2.view(N, -1, self.an2, 2 * h, 2 * w)  # [N, C, an2, H, W]
        f_2 = f_2.permute([0, 2, 1, 3, 4]).contiguous()  # [N, an2, C, H, W]
        f_2 = f_2.view(N * self.an2, f_2.shape[2], 2 * h, 2 * w)
        ##########
        fup_2 = self.fup2(f_2)  # [N*81,64,4h,4w]
        res_2 = self.res2(fup_2)  # [N*81,1,4h,4w]
        iup_2 = self.iup2(sr_2x)  # [N*81,1,4h,4w]
        sr_4x = res_2 + iup_2  # [N*81,1,4h,4w]

        sr_2x = sr_2x.view(N, self.an2, h * 2, w * 2)
        sr_4x = sr_4x.view(N, self.an2, h * 4, w * 4)

        return sr_4x, sr_2x


# class net4x_new(nn.Module):
#     def __init__(self, an, layer, mode="catres", fn=64):
#         super(net4x_new, self).__init__()
#
#         self.an = an
#         self.an2 = an * an
#         self.fn = fn
#         self.mode = mode
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#         self.conv0 = nn.Conv2d(in_channels=1, out_channels=fn, kernel_size=3, stride=1, padding=1)
#
#         self.altblock1 = self.make_layer(layer_num=layer, fn=fn)
#
#         self.fup1 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=fn, out_channels=fn, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         )
#         self.res1 = nn.Conv2d(in_channels=fn, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.iup1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)
#
#         self.altblock2 = self.make_layer(layer_num=layer, fn=fn)
#
#         self.fup2 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=fn, out_channels=64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         )
#         self.res2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.iup2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1)
#         for m in self.modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 c1, c2, h, w = m.weight.data.size()
#                 weight = get_upsample_filter(h)
#                 m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, 0.2, 'fan_in', 'leaky_relu')
#                 m.weight.data *= 0.1
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def make_layer(self, layer_num=6, fn=64):
#
#         layers = []
#         sas_para = SAS_para()
#         sac_para = SAC_para()
#         sas_para.fn = fn
#         sac_para.fn = fn
#         if self.mode == "catres":
#             for i in range(layer_num):
#                 layers.append(SAV_concat(SAS_para=sas_para, SAC_para=sac_para))
#         elif self.mode == "parares":
#             for i in range(layer_num):
#                 layers.append(SAV_parallel(SAS_para=sas_para, SAC_para=sac_para, feature_concat=False))
#         elif self.mode == "SAS":
#             for i in range(layer_num):
#                 layers.append(SAS_conv(act='lrelu', fn=fn))
#         elif self.mode == "SAC":
#             for i in range(layer_num):
#                 layers.append(SAC_conv(act='lrelu', fn=fn))
#         return nn.Sequential(*layers)
#
#     def forward(self, lr):
#
#         N, _, h, w = lr.shape  # lr [N,81,h,w]
#         lr = lr.view(N * self.an2, 1, h, w)  # [N*81,1,h,w]
#
#         x = self.lrelu(self.conv0(lr))  # [N*81,64,h,w]
#         #########
#         lf_out = x.view(N, self.an2, -1, h, w)  # [N, an2, C, H, W]
#         lf_out = lf_out.permute([0, 2, 1, 3, 4]).contiguous()  # [N, C, an2, H, W]
#         lf_out = lf_out.view(N, lf_out.shape[1], self.an, self.an, h, w)  # [N, C, an, an, H, W]
#         lf_out = self.altblock1(lf_out)  # [N*81,64,h,w]
#         lf_out = lf_out.view(N, -1, self.an2, h, w)  # [N, C, an2, H, W]
#         lf_out = lf_out.permute([0, 2, 1, 3, 4]).contiguous()  # [N, an2, C, H, W]
#         lf_out = lf_out.view(N * self.an2, lf_out.shape[2], h, w)
#         #######
#         fup_1 = self.fup1(lf_out)  # [N*81,64,2h,2w]
#         res_1 = self.res1(fup_1)  # [N*81,1,2h,2w]
#         iup_1 = self.iup1(lr)  # [N*81,1,2h,2w]
#
#         sr_2x = res_1 + iup_1  # [N*81,1,2h,2w]
#         ##########
#         f_2 = fup_1.view(N, self.an2, -1, 2 * h, 2 * w)  # [N, an2, C, H, W]
#         f_2 = f_2.permute([0, 2, 1, 3, 4]).contiguous()  # [N, C, an2, H, W]
#         f_2 = f_2.view(N, f_2.shape[1], self.an, self.an, 2 * h, 2 * w)  # [N, C, an, an, H, W]
#         f_2 = self.altblock2(f_2)  # [N*81,64,2h,2w]
#         f_2 = f_2.view(N, -1, self.an2, 2 * h, 2 * w)  # [N, C, an2, H, W]
#         f_2 = f_2.permute([0, 2, 1, 3, 4]).contiguous()  # [N, an2, C, H, W]
#         f_2 = f_2.view(N * self.an2, f_2.shape[2], 2 * h, 2 * w)
#         ##########
#         fup_2 = self.fup2(f_2)  # [N*81,64,4h,4w]
#         res_2 = self.res2(fup_2)  # [N*81,1,4h,4w]
#         iup_2 = self.iup2(sr_2x)  # [N*81,1,4h,4w]
#         sr_4x = res_2 + iup_2  # [N*81,1,4h,4w]
#
#         sr_2x = sr_2x.view(N, self.an2, h * 2, w * 2)
#         sr_4x = sr_4x.view(N, self.an2, h * 4, w * 4)
#
#         return sr_4x, sr_2x