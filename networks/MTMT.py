import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
import pdb
from typing import List

from .resnext.resnext101_5out import ResNeXt101

import pdb


config_vgg = {'convert': [[128, 256, 512, 512, 512], [64, 128, 256, 512, 512]],
              'merge1': [[128, 256, 128, 3, 1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],
                         [512, 0, 512, 7, 3]], 'merge2': [[128], [256, 512, 512, 512]]}  # no convert layer, no conv6

config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [128, 256, 512, 512, 512]],
                 'edgeinfo': [[16, 16, 16, 16], 128, [16, 8, 4, 2]], 'edgeinfoc': [64, 128],
                 'block': [[512, [16]], [256, [16]], [256, [16]], [128, [16]]], 'fuse': [[16, 16, 16, 16], True],
                 'fuse_ratio': [[16, 1], [8, 1], [4, 1], [2, 1]],
                 'merge1': [[128, 256, 128, 3, 1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],
                            [512, 0, 512, 7, 3]], 'merge2': [[128], [256, 512, 512, 512]]}

config_resnext101 = {'convert': [[64, 256, 512, 1024, 2048], [32, 64, 64, 64, 64]],
                 'merge1': [[32, 0, 32, 3, 1], [64, 0, 64, 3, 1], [64, 0, 64, 5, 2], [64, 0, 64, 5, 2], [64, 0, 64, 7, 3]], 'merge2': [[32], [64, 64, 64, 64]]}


class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up0, up1, up2 = [], [], []
        for i in range(len(list_k[0])):
            up0.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.BatchNorm2d(list_k[1][i]), nn.ReLU(inplace=True)))

        self.convert0 = nn.ModuleList(up0)
        # ModuleList(
        #   (0): Sequential(
        #     (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU(inplace=True)
        #   )
        #   (1): Sequential(
        #     (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU(inplace=True)
        #   )
        #   (2): Sequential(
        #     (0): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU(inplace=True)
        #   )
        #   (3): Sequential(
        #     (0): Conv2d(1024, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU(inplace=True)
        #   )
        #   (4): Sequential(
        #     (0): Conv2d(2048, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU(inplace=True)
        #   )
        # )

    def forward(self, list_x: List[torch.Tensor]):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl

class ResidualBlockLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockLayer, self).__init__()
        self.residual_block1 = nn.Sequential(nn.Conv2d(in_channels, in_channels//4, 1, 1,bias=False),
                      nn.Conv2d(in_channels//4, in_channels//4, 3, 1, 1, bias=False),
                      nn.Conv2d(in_channels//4, in_channels, 1, 1, bias=False))
        self.residual_block2 = nn.Sequential(nn.Conv2d(in_channels, in_channels//4, 1, 1,bias=False),
                      nn.Conv2d(in_channels//4, in_channels//4, 3, 1, 1, bias=False),
                      nn.Conv2d(in_channels//4, in_channels, 1, 1, bias=False))
        self.out = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        self.relu = nn.ReLU()

        pdb.set_trace()

    def forward(self, x):
        x_tmp = self.relu(x + self.residual_block1(x))
        x_tmp = self.relu(x_tmp + self.residual_block2(x_tmp))
        x_tmp = self.out(x_tmp)
        return x_tmp

#DSS merge
class MergeLayer1(nn.Module):
    def __init__(self):
        super(MergeLayer1, self).__init__()
        list_k = [[32, 0, 32, 3, 1],
         [64, 0, 64, 3, 1],
         [64, 0, 64, 5, 2],
         [64, 0, 64, 5, 2],
         [64, 0, 64, 7, 3]]

        trans, up, DSS = [], [], []
        for i, ik in enumerate(list_k):
            up.append(nn.Sequential(nn.Conv2d(ik[0], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True)))
            if i > 0 and i < len(list_k)-1: # i represent number
                DSS.append(nn.Sequential(nn.Conv2d(ik[0]*(i+1), ik[0], 1, 1, 1),
                           nn.BatchNorm2d(ik[0]), nn.ReLU(inplace=True)))

        trans.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.ReLU(inplace=True)))


        # self.shadow_score = nn.Conv2d(list_k[0][2], 1, 3, 1, 1)
        self.shadow_score = nn.Sequential(
            nn.Conv2d(list_k[1][2], list_k[1][2]//4, 3, 1, 1), 
            nn.BatchNorm2d(list_k[1][2]//4), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[1][2]//4, 1, 1)
        )
        # Sequential(
        #   (0): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (2): ReLU(inplace=True)
        #   (3): Dropout(p=0.1, inplace=False)
        #   (4): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1))
        # )

        # self.edge_score = nn.Conv2d(list_k[0][2], 1, 3, 1, 1)
        self.edge_score = nn.Sequential(
            nn.Conv2d(list_k[0][2], list_k[0][2]//4, 3, 1, 1), 
            nn.BatchNorm2d(list_k[0][2]//4), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), 
            nn.Conv2d(list_k[0][2]//4, 1, 1)
        )

        self.up = nn.ModuleList(up)
        self.relu = nn.ReLU()
        self.trans = nn.ModuleList(trans)
        # (Pdb) self.trans
        # ModuleList(
        #   (0): Sequential(
        #     (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (1): ReLU(inplace=True)
        #   )
        # )
        self.DSS = nn.ModuleList(DSS)

        # subitizing section
        self.number_per_fc = nn.Linear(list_k[1][2], 1) #64->1
        # self.number_per_fc -- Linear(in_features=64, out_features=1, bias=True)

        torch.nn.init.constant_(self.number_per_fc.weight, 0)


    def forward(self, list_x: List[torch.Tensor], x_size: List[int]):
        # list_x -- resnext_feature
        # (Pdb) len(list_x), list_x[0].size(), list_x[1].size(), list_x[2].size(), list_x[3].size(), list_x[4].size()
        # (5, [1, 32, 208, 208], [1, 64, 104, 104], [1, 64, 52, 52], [1, 64, 26, 26], [1, 64, 13, 13])

        fpn_edge_score, fpn_shadow_score, fpn_edge_feature, fpn_shadow_feature, U_tmp = [], [], [], [], []

        # layer5
        num_f = len(list_x)
        tmp = self.up[num_f - 1](list_x[num_f - 1]) # [1, 64, 13, 13]
        fpn_shadow_feature.append(tmp)
        U_tmp.append(tmp)
        # self.shadow_score(tmp).size() -- [1, 1, 13, 13]
        fpn_shadow_score.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))


        # layer4
        up_tmp_x2 = F.interpolate(U_tmp[0], list_x[3].size()[2:], mode='bilinear', align_corners=True)
        U_tmp.append(self.DSS[0](torch.cat([up_tmp_x2, list_x[3]], dim=1)))
        tmp = self.up[3](U_tmp[-1])
        fpn_shadow_feature.append(tmp)
        fpn_shadow_score.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))

        # layer3
        up_tmp_x2 = F.interpolate(U_tmp[1], list_x[2].size()[2:], mode='bilinear', align_corners=True)
        up_tmp_x4 = F.interpolate(U_tmp[0], list_x[2].size()[2:], mode='bilinear', align_corners=True)
        U_tmp.append(self.DSS[1](torch.cat([up_tmp_x4, up_tmp_x2, list_x[2]], dim=1)))
        tmp = self.up[2](U_tmp[-1])
        fpn_shadow_feature.append(tmp)
        fpn_shadow_score.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))

        # layer2
        up_tmp_x2 = F.interpolate(U_tmp[2], list_x[1].size()[2:], mode='bilinear', align_corners=True)
        up_tmp_x4 = F.interpolate(U_tmp[1], list_x[1].size()[2:], mode='bilinear', align_corners=True)
        up_tmp_x8 = F.interpolate(U_tmp[0], list_x[1].size()[2:], mode='bilinear', align_corners=True)
        U_tmp.append(self.DSS[2](torch.cat([up_tmp_x8, up_tmp_x4, up_tmp_x2, list_x[1]], dim=1)))
        tmp = self.up[1](U_tmp[-1])
        fpn_shadow_feature.append(tmp)
        # fpn_shadow_score.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))

        # vector = F.adaptive_avg_pool2d(fpn_shadow_feature[0], output_size=1)
        # vector = vector.view(vector.size(0), -1)
        # fc_score = self.number_per_fc(vector)

        # edge layer fuse
        U_tmp = list_x[0] + F.interpolate((self.trans[-1](fpn_shadow_feature[0])), list_x[0].size()[2:], mode='bilinear', align_corners=True)
        tmp = self.up[0](U_tmp)
        fpn_edge_feature = tmp

        # fpn_edge_score = F.interpolate(self.edge_score(tmp), x_size, mode='bilinear', align_corners=True)

        # fpn_edge_score.size() -- [1, 1, 416, 416]
        # fpn_edge_feature.size() -- [1, 32, 208, 208]
        # len(fpn_shadow_score) -- 4
        # len(fpn_shadow_feature) -- 4

        # return fpn_edge_score, fpn_edge_feature, fpn_shadow_score, fpn_shadow_feature, fc_score
        return fpn_edge_feature, fpn_shadow_feature

class MergeLayer1_FPN(nn.Module):
    def __init__(self, list_k):
        super(MergeLayer1_FPN, self).__init__()
        pdb.set_trace()

        self.list_k = list_k
        trans, up = [], []
        for ik in list_k:
            up.append(nn.Sequential(nn.Conv2d(ik[0], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True)))
        trans.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.ReLU(inplace=True)))
        # self.shadow_score = nn.Conv2d(list_k[0][2], 1, 3, 1, 1)
        self.shadow_score = nn.Sequential(
            nn.Conv2d(list_k[1][2], list_k[1][2]//4, 3, 1, 1), nn.BatchNorm2d(list_k[1][2]//4), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[1][2]//4, 1, 1)
        )
        # self.edge_score = nn.Conv2d(list_k[0][2], 1, 3, 1, 1)
        self.edge_score = nn.Sequential(
            nn.Conv2d(list_k[0][2], list_k[0][2]//4, 3, 1, 1), nn.BatchNorm2d(list_k[0][2]//4), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[0][2]//4, 1, 1)
        )
        self.up = nn.ModuleList(up)
        self.relu = nn.ReLU()
        self.trans = nn.ModuleList(trans)
        pdb.set_trace()

    def forward(self, list_x, x_size):
        fpn_edge_score, fpn_shadow_score, fpn_edge_feature, fpn_shadow_feature = [], [], [], []

        num_f = len(list_x)
        tmp = self.up[num_f - 1](list_x[num_f - 1])
        fpn_shadow_feature.append(tmp)
        U_tmp = tmp
        fpn_shadow_score.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))

        for j in range(2, num_f):
            i = num_f - j
            U_tmp = list_x[i] + F.interpolate((U_tmp), list_x[i].size()[2:], mode='bilinear', align_corners=True)

            tmp = self.up[i](U_tmp)
            U_tmp = tmp
            fpn_shadow_feature.append(tmp)
            fpn_shadow_score.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))

        # edge layer fuse
        U_tmp = list_x[0] + F.interpolate((self.trans[-1](fpn_shadow_feature[0])), list_x[0].size()[2:], mode='bilinear', align_corners=True)
        tmp = self.up[0](U_tmp)
        fpn_edge_feature.append(tmp)

        fpn_edge_score.append(F.interpolate(self.edge_score(tmp), x_size, mode='bilinear', align_corners=True))
        return fpn_edge_score, fpn_edge_feature, fpn_shadow_score, fpn_shadow_feature


class MergeLayer2(nn.Module):
    def __init__(self):
        super(MergeLayer2, self).__init__()
        list_k = [[32], [64, 64, 64, 64]]

        self.list_k = list_k
        trans, up, score = [], [], []
        for i in list_k[0]:
            tmp = []
            tmp_up = []
            tmp_score = []
            feature_k = [[3, 1], [5, 2], [5, 2], [7, 3]]
            for idx, j in enumerate(list_k[1]):
                tmp.append(nn.Sequential(nn.Conv2d(j, i, 1, 1, bias=False), nn.BatchNorm2d(i), nn.ReLU(inplace=True)))
                tmp_up.append(
                    nn.Sequential(nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(i), nn.ReLU(inplace=True),
                                  nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(i), nn.ReLU(inplace=True),
                                  nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(i), nn.ReLU(inplace=True)))
            trans.append(nn.ModuleList(tmp))
            up.append(nn.ModuleList(tmp_up))
        self.sub_score = nn.Sequential(
            nn.Conv2d(list_k[0][0], list_k[0][0]//4, 3, 1, 1), nn.BatchNorm2d(list_k[0][0]//4), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[0][0]//4, 1, 1)
        )

        self.trans = nn.ModuleList(trans)
        self.up = nn.ModuleList(up)

        self.relu = nn.ReLU()


    def forward(self, list_x, list_y, x_size):
        # fpn_edge_feature, fpn_shadow_feature, x_size
        up_score, tmp_feature = [], []
        list_y = list_y[::-1] # Reverse

        for j, j_x in enumerate(list_y):
            tmp = F.interpolate(self.trans[0][j](j_x), list_x.size()[2:], mode='bilinear', align_corners=True) + list_x
            tmp_f = self.up[0][j](tmp)
            up_score.append(F.interpolate(self.sub_score(tmp_f), x_size, mode='bilinear', align_corners=True))
            tmp_feature.append(tmp_f)


        tmp_fea = tmp_feature[0]
        for i_fea in range(len(tmp_feature) - 1):
            tmp_fea = self.relu(torch.add(tmp_fea, F.interpolate((tmp_feature[i_fea + 1]), tmp_feature[0].size()[2:],
                                                                 mode='bilinear', align_corners=True)))
        up_score.append(F.interpolate(self.sub_score(tmp_fea), x_size, mode='bilinear', align_corners=True))
        # up_score.append(F.interpolate(self.final_score(tmp_fea), x_size, mode='bilinear', align_corners=True))

        # len(up_score), up_score[0].size(),up_score[1].size(),up_score[2].size(),up_score[3].size(),up_score[4].size()
        # (5, [1, 1, 416, 416], [1, 1, 416, 416], [1, 1, 416, 416], [1, 1, 416, 416], [1, 1, 416, 416])
        return up_score


# extra part
def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    elif base_model_cfg == 'resnext101':
        config = config_resnext101

    #  config['merge1']
    # [[32, 0, 32, 3, 1], 
    # [64, 0, 64, 3, 1], 
    # [64, 0, 64, 5, 2], 
    # [64, 0, 64, 5, 2], 
    # [64, 0, 64, 7, 3]]

    # config['merge2']
    # [[32], [64, 64, 64, 64]]

    merge1_layers = MergeLayer1()
    merge2_layers = MergeLayer2()

    return vgg, merge1_layers, merge2_layers


# TUN network
class TUN_bone(nn.Module):
    def __init__(self, base_model_cfg, base, merge1_layers, merge2_layers):
        super(TUN_bone, self).__init__()
        self.base_model_cfg = base_model_cfg

        # self.base_model_cfg -- 'resnext101'
        # base -- ResNeXt101()

        if self.base_model_cfg == 'vgg':
            self.base = base
            # self.base_ex = nn.ModuleList(base_ex)
            self.merge1 = merge1_layers
            self.merge2 = merge2_layers

        elif self.base_model_cfg == 'resnext101':
            # config_resnext101['convert'] -- [[64, 256, 512, 1024, 2048], [32, 64, 64, 64, 64]]
            self.convert = ConvertLayer(config_resnext101['convert'])
            self.base = base
            self.merge1 = merge1_layers
            self.merge2 = merge2_layers

    def forward(self, x):
        # x.size() -- [1, 3, 416, 416]
        x_size = x.size()[2:] # [416, 416]
        resnext_feature = self.base(x)
        # (Pdb) len(resnext_feature) -- 5
        # resnext_feature[0].size(),
        # resnext_feature[1].size(), 
        # resnext_feature[2].size(),
        # resnext_feature[3].size(), 
        # resnext_feature[4].size()
        # [1, 64, 208, 208], 
        # [1, 256, 104, 104], 
        # [1, 512, 52, 52], 
        # [1, 1024, 26, 26], 
        # [1, 2048, 13, 13]
        if self.base_model_cfg == 'resnext101':
            resnext_feature = self.convert(resnext_feature)
        # resnext_feature[0].size(),resnext_feature[1].size(), resnext_feature[2].size(),resnext_feature[3].size(), resnext_feature[4].size()
        # [1, 32, 208, 208], [1, 64, 104, 104], [1, 64, 52, 52], [1, 64, 26, 26], [1, 64, 13, 13]

        # fpn_edge_score, fpn_edge_feature, fpn_shadow_score, fpn_shadow_feature, fc_score = self.merge1(resnext_feature, x_size)

        fpn_edge_feature, fpn_shadow_feature = self.merge1(resnext_feature, x_size)
        fpn_final_score = self.merge2(fpn_edge_feature, fpn_shadow_feature, x_size)
        mask = torch.sigmoid(fpn_final_score[-1])
        return mask


# build the whole network
def build_model(base_model_cfg='resnext101', ema=False):
    if not ema:
        return TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, ResNeXt101()))
    else:
        return TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, ResNeXt101()))


# weight init
def xavier(param):
    # init.xavier_uniform(param)
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
