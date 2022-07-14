# @Time = 2020.12.16
# @Author = Zhen

"""
Model utilities.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from utils.convNd import convNd

class indexes:

    indexes_for_2_to_8 = [60, 0, 1, 2, 3, 4, 5, 61,
                          6, 7, 8, 9, 10, 11, 12, 13,
                          14, 15, 16, 17, 18, 19, 20, 21,
                          22, 23, 24, 25, 26, 27, 28, 29,
                          30, 31, 32, 33, 34, 35, 36, 37,
                          38, 39, 40, 41, 42, 43, 44, 45,
                          46, 47, 48, 49, 50, 51, 52, 53,
                          62, 54, 55, 56, 57, 58, 59, 63]
    indexes_for_2_to_7 = [45, 0, 1, 2, 3, 4, 46,
                          5, 6, 7, 8, 9, 10, 11,
                          12, 13, 14, 15, 16, 17, 18,
                          19, 20, 21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30, 31, 32,
                          33, 34, 35, 36, 37, 38, 39,
                          47, 40, 41, 42, 43, 44, 48]
    indexes_for_2_to_8_extra1 = [0, 1, 2, 3, 4, 5, 6, 7,
                                 8, 60, 9, 10, 11, 12, 61, 13,
                                 14, 15, 16, 17, 18, 19, 20, 21,
                                 22, 23, 24, 25, 26, 27, 28, 29,
                                 30, 31, 32, 33, 34, 35, 36, 37,
                                 38, 39, 40, 41, 42, 43, 44, 45,
                                 46, 62, 47, 48, 49, 50, 63, 51,
                                 52, 53, 54, 55, 56, 57, 58, 59]
    indexes_for_2_to_8_extra2 = [0, 1, 2, 3, 4, 5, 6, 7,
                                 8, 9, 10, 11, 12, 13, 14, 15,
                                 16, 17, 60, 18, 19, 61, 20, 21,
                                 22, 23, 24, 25, 26, 27, 28, 29,
                                 30, 31, 32, 33, 34, 35, 36, 37,
                                 38, 39, 62, 40, 41, 63, 42, 43,
                                 44, 45, 46, 47, 48, 49, 50, 51,
                                 52, 53, 54, 55, 56, 57, 58, 59]


class SAS_conv(nn.Module):
    def __init__(self, act='relu', fn=64):
        super(SAS_conv, self).__init__()

        # self.an = an
        self.init_indicator = 'relu'
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise Exception("Wrong activation function!")

        self.spaconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.spaconv.weight, a, 'fan_in', self.init_indicator)
        init.constant_(self.spaconv.bias, 0.0)

        self.angconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1)
        init.kaiming_normal_(self.angconv.weight, a, 'fan_in', self.init_indicator)
        init.constant_(self.angconv.bias, 0.0)


    def forward(self, x):
        N, c, U, V, h, w = x.shape  # [N,c,U,V,h,w]
        # N = N // (self.an * self.an)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(N*U*V, c, h, w)

        out = self.act(self.spaconv(x))  # [N*U*V,c,h,w]
        out = out.view(N, U*V, c, h * w)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * h * w, c, U, V)  # [N*h*w,c,U,V]

        out = self.act(self.angconv(out))  # [N*h*w,c,U,V]
        out = out.view(N, h * w, c, U*V)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, U, V, c, h, w)  # [N,U,V,c,h,w]
        out = out.permute(0, 3, 1, 2, 4, 5).contiguous() # [N,c,U,V,h,w]
        return out


class SAC_conv(nn.Module):
    def __init__(self, act='relu', symmetry=True, max_k_size=3, fn=64):
        super(SAC_conv, self).__init__()

        # self.an = an
        self.init_indicator = 'relu'
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise Exception("Wrong activation function!")

        if symmetry:
            k_size_ang = max_k_size
            k_size_spa = max_k_size
        else:
            k_size_ang = max_k_size - 2
            k_size_spa = max_k_size

        self.verconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(k_size_ang, k_size_spa),
                                 stride=(1,1), padding=(k_size_ang // 2, k_size_spa // 2))
        init.kaiming_normal_(self.verconv.weight, a, 'fan_in', self.init_indicator)
        init.constant_(self.verconv.bias, 0.0)

        self.horconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(k_size_ang, k_size_spa),
                                 stride=(1,1), padding=(k_size_ang // 2, k_size_spa // 2))
        init.kaiming_normal_(self.horconv.weight, a, 'fan_in', self.init_indicator)
        init.constant_(self.horconv.bias, 0.0)


    def forward(self, x):
        N, c, U, V, h, w = x.shape  # [N,c,U,V,h,w]
        # N = N // (self.an * self.an)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N*V*w, c, U, h)

        out = self.act(self.verconv(x))  # [N*V*w,c,U,h]
        out = out.view(N, V * w, c, U * h)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * U * h, c, V, w)  # [N*U*h,c,V,w]

        out = self.act(self.horconv(out))  # [N*U*h,c,V,w]
        out = out.view(N, U * h, c, V * w)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, V, w, c, U, h)  # [N,V,w,c,U,h]
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous() # [N,c,U,V,h,w]
        return out

class SAV_concat(nn.Module):
    def __init__(self, SAS_para, SAC_para, residual_connection=True):
        """
        parameters for building SAS-SAC block
        :param SAS_para: {act, fn}
        :param SAC_para: {act, symmetry, max_k_size, fn}
        :param residual_connection: True or False for residual connection
        """
        super(SAV_concat, self).__init__()
        self.res_connect = residual_connection
        self.SAS_conv = SAS_conv(act=SAS_para.act, fn=SAS_para.fn)
        self.SAC_conv = SAC_conv(act=SAC_para.act, symmetry=SAC_para.symmetry, max_k_size=SAC_para.max_k_size, fn=SAC_para.fn)
        
    def forward(self, lf_input):
        feat = self.SAS_conv(lf_input)
        res = self.SAC_conv(feat)
        if self.res_connect:
            res += lf_input
        return res

class SAV_parallel(nn.Module):
    def __init__(self, SAS_para, SAC_para, feature_concat=True):
        super(SAV_parallel, self).__init__()
        self.feature_concat = feature_concat
        self.SAS_conv = SAS_conv(act=SAS_para.act, fn=SAS_para.fn)
        self.SAC_conv = SAC_conv(act=SAC_para.act, symmetry=SAC_para.symmetry, max_k_size=SAC_para.max_k_size, fn=SAC_para.fn)
        if self.feature_concat:
            self.channel_reduce = convNd(in_channels=2 * SAS_para.fn,
                                   out_channels=SAS_para.fn,
                                   num_dims=4,
                                   kernel_size=(1, 1, 1, 1),
                                   stride=(1, 1, 1, 1),
                                   padding=(0, 0, 0, 0),
                                   kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in', 'leaky_relu'),
                                   bias_initializer=lambda x: nn.init.constant_(x, 0.0))
    def forward(self, lf_input):
        sas_feat = self.SAS_conv(lf_input)
        sac_feat = self.SAC_conv(lf_input)# [N,c,U,V,h,w]

        if self.feature_concat:
            concat_feat = torch.cat((sas_feat, sac_feat), dim=1)  # [N,2c,U,V,h,w]
            res = self.channel_reduce(concat_feat)
            res += lf_input
        else:
            res = sas_feat + sac_feat + lf_input
        return res

def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

def weights_init_kaiming_small(m, act, a, scale):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose3d') != -1:
        nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

def weights_init_kaiming_for_all_small(m, act, scale, a=0):
    # if act="relu", a use the default value
    # if act="leakyrelu", a use the negative slope of leakyrelu
    nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
    m.weight.data *= scale
    if m.bias is not None:
        m.bias.data.zero_()

def weights_init_kaiming_for_all(m, act, a=0):
    nn.init.kaiming_normal_(m.weight, a, 'fan_in', act)
    if m.bias is not None:
        m.bias.data.zero_()


class SAS_para:
    def __init__(self):
        self.act = 'lrelu'
        self.fn = 64

class SAC_para:
    def __init__(self):
        self.act = 'lrelu'
        self.symmetry = True
        self.max_k_size = 3
        self.fn = 64