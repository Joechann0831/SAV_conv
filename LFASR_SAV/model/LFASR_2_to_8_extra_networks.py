# @Time = 2021.03.07
# @Author = Zhen

"""
LFASR with network using SAS convolution for 2x2 to 8x8 with extrapolation.
"""

import torch
import torch.nn as nn
from model.model_utils import *
from utils.convNd import convNd

class LFASR_2_to_8_extra_net(nn.Module):
    def __init__(self, block_num, block_mode="SAS", fn=64, init_gain=1.0, extra_start=1):
        super(LFASR_2_to_8_extra_net, self).__init__()

        self.new_ind = indexes()
        self.extra_start = extra_start

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.feature_conv = convNd(in_channels=1,
                                   out_channels=fn,
                                   num_dims=4,
                                   kernel_size=(3, 3, 3, 3),
                                   stride=(1, 1, 1, 1),
                                   padding=(1, 1, 1, 1),
                                   kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in', 'leaky_relu'),
                                   bias_initializer=lambda x: nn.init.constant_(x, 0.0))

        sas_para = SAS_para()
        sac_para = SAC_para()
        sas_para.act = 'lrelu'
        sas_para.fn = fn
        sac_para.act = 'lrelu'
        sac_para.symmetry = True
        sac_para.max_k_size = 3
        sac_para.fn = fn

        if block_mode == "SAS":
            alt_blocks = [SAS_conv(act='lrelu', fn=fn) for _ in range(block_num)]
        elif block_mode == "SAC":
            alt_blocks = [SAC_conv(act='lrelu', fn=fn) for _ in range(block_num)]
        elif block_mode == "SAV_catres":
            alt_blocks = [SAV_concat(SAS_para=sas_para, SAC_para=sac_para, residual_connection=True) for _ in
                          range(block_num)]
        elif block_mode == "SAV_cat":
            alt_blocks = [SAV_concat(SAS_para=sas_para, SAC_para=sac_para, residual_connection=False) for _ in
                          range(block_num)]
        elif block_mode == "SAV_para":
            alt_blocks = [SAV_parallel(SAS_para=sas_para, SAC_para=sac_para, feature_concat=False) for _ in
                          range(block_num)]
        else:
            raise Exception("Not implemented block mode!")

        self.alt_blocks = nn.Sequential(*alt_blocks)

        self.synthesis_layer = convNd(in_channels=fn,
                                      out_channels=60,
                                      num_dims=4,
                                      kernel_size=(2, 2, 3, 3),
                                      stride=(1, 1, 1, 1),
                                      padding=(0, 0, 1, 1),
                                      kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in', 'leaky_relu'),
                                      bias_initializer=lambda x: nn.init.constant_(x, 0.0))

        self.ang_dim_reduction1 = convNd(in_channels=1,
                                         out_channels=16,
                                         num_dims=4,
                                         kernel_size=(2, 2, 3, 3),
                                         stride=(2, 2, 1, 1),
                                         padding=(0, 0, 1, 1),
                                         kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                              'leaky_relu'),
                                         bias_initializer=lambda x: nn.init.constant_(x, 0.0))
        self.ang_dim_reduction2 = convNd(in_channels=16,
                                         out_channels=64,
                                         num_dims=4,
                                         kernel_size=(2, 2, 3, 3),
                                         stride=(2, 2, 1, 1),
                                         padding=(0, 0, 1, 1),
                                         kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                              'leaky_relu'),
                                         bias_initializer=lambda x: nn.init.constant_(x, 0.0))
        self.residual_predict = convNd(in_channels=64,
                                       out_channels=60,
                                       num_dims=4,
                                       kernel_size=(2, 2, 3, 3),
                                       stride=(1, 1, 1, 1),
                                       padding=(0, 0, 1, 1),
                                       kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                            'leaky_relu'),
                                       bias_initializer=lambda x: nn.init.constant_(x, 0.0))
        for m in self.modules():
            weights_init_kaiming_small(m, 'leaky_relu', a=0.2, scale=init_gain)

    def freeze_coarse_network(self):
        print("Freeze the coarse network!")
        for m in self.feature_conv.parameters():
            m.requires_grad = False
        for m in self.alt_blocks.parameters():
            m.requires_grad = False
        for m in self.synthesis_layer.parameters():
            m.requires_grad = False

    def forward(self, lf_input):
        # lf_input: [B, 1, 2, 2, H, W]

        B, H, W = lf_input.shape[0], lf_input.shape[4], lf_input.shape[5]
        feat = self.relu(self.feature_conv(lf_input)) # [B, 64, 2, 2, H, W]
        feat_syn = self.alt_blocks(feat) # [B, 64, 2, 2, H, W]
        new_views = self.synthesis_layer(feat_syn) # [B, 60, 1, 1, H, W]

        # concat, re-organization and reshape
        lf_input = lf_input.view(B, 4, 1, 1, H, W) # [B, 4, 1, 1, H, W]
        lf_recon = torch.cat((new_views, lf_input), dim=1) # [B, 64, 1, 1, H, W]

        if self.extra_start == 1:
            new_ind = torch.LongTensor(self.new_ind.indexes_for_2_to_8_extra1).to(lf_input.device)
        elif self.extra_start == 2:
            new_ind = torch.LongTensor(self.new_ind.indexes_for_2_to_8_extra2).to(lf_input.device)
        else:
            raise Exception("Please import right extra start!")

        lf_recon = lf_recon.index_select(1, new_ind) # [B, 64, 1, 1, H, W]
        lf_recon = lf_recon.view(B, 1, 8, 8, H, W) # [B, 1, 8, 8, H, W]

        ## refinement
        feat_ang_reduce1 = self.relu(self.ang_dim_reduction1(lf_recon)) # [B, 16, 4, 4, H, W]
        feat_ang_reduce2 = self.relu(self.ang_dim_reduction2(feat_ang_reduce1)) # [B, 64, 2, 2, H, W]
        residual = self.residual_predict(feat_ang_reduce2) # [B, 60, 1, 1, H, W]

        ## final reconstruction
        new_views_refine = residual + new_views # [B, 60, 1, 1, H, W]
        lf_recon_refine = torch.cat((new_views_refine, lf_input), dim=1) # [B, 64, 1, 1, H, W]
        lf_recon_refine = lf_recon_refine.index_select(1, new_ind)
        lf_recon_refine = lf_recon_refine.view(B, 1, 8, 8, H, W) # [B, 1, 8, 8, H, W]
        return lf_recon, lf_recon_refine