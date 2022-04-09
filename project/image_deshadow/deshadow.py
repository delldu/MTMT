"""Data loader."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2020-2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:46:28 CST
# ***
# ************************************************************************************/
#

import os
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from resnext import ResNeXt101

import pdb
from typing import List


class DeshadowModel(nn.Module):
    def __init__(self):
        super(DeshadowModel, self).__init__()

        self.convert = ConvertLayer()
        self.base = ResNeXt101()
        self.merge1 = MergeLayer1()
        self.merge2 = MergeLayer2()

    def forward(self, x):
        x_size = x.size()[2:]
        conv2merge = self.base(x)
        conv2merge = self.convert(conv2merge)
        up_edge, edge_feature, up_sal, sal_feature, up_subitizing = self.merge1(conv2merge, x_size)
        up_sal_final = self.merge2(edge_feature, sal_feature, x_size)
        return up_edge, up_sal, up_subitizing, up_sal_final


class ConvertLayer(nn.Module):
    def __init__(self):
        super(ConvertLayer, self).__init__()
        list_k = [[64, 256, 512, 1024, 2048], [32, 64, 64, 64, 64]]
        up0, up1, up2 = [], [], []
        for i in range(len(list_k[0])):
            up0.append(
                nn.Sequential(
                    nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False),
                    nn.BatchNorm2d(list_k[1][i]),
                    nn.ReLU(inplace=True),
                )
            )

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


class MergeLayer1(nn.Module):
    def __init__(self):
        super(MergeLayer1, self).__init__()
        self.list_k = [[32, 0, 32, 3, 1], [64, 0, 64, 3, 1], [64, 0, 64, 5, 2], [64, 0, 64, 5, 2], [64, 0, 64, 7, 3]]

        trans, up, DSS = [], [], []

        for i, ik in enumerate(self.list_k):
            up.append(
                nn.Sequential(
                    nn.Conv2d(ik[0], ik[2], ik[3], 1, ik[4]),
                    nn.BatchNorm2d(ik[2]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]),
                    nn.BatchNorm2d(ik[2]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]),
                    nn.BatchNorm2d(ik[2]),
                    nn.ReLU(inplace=True),
                )
            )
            if i > 0 and i < len(self.list_k) - 1:  # i represent number
                DSS.append(
                    nn.Sequential(
                        nn.Conv2d(ik[0] * (i + 1), ik[0], 1, 1, 1), 
                        nn.BatchNorm2d(ik[0]), 
                        nn.ReLU(inplace=True)
                    )
                )

        trans.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.ReLU(inplace=True)))

        # self.shadow_score = nn.Conv2d(list_k[0][2], 1, 3, 1, 1)
        self.shadow_score = nn.Sequential(
            nn.Conv2d(self.list_k[1][2], self.list_k[1][2] // 4, 3, 1, 1),
            nn.BatchNorm2d(self.list_k[1][2] // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(self.list_k[1][2] // 4, 1, 1),
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
            nn.Conv2d(self.list_k[0][2], self.list_k[0][2] // 4, 3, 1, 1),
            nn.BatchNorm2d(self.list_k[0][2] // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(self.list_k[0][2] // 4, 1, 1),
        )
        # (Pdb) self.edge_score
        # Sequential(
        #   (0): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (2): ReLU(inplace=True)
        #   (3): Dropout(p=0.1, inplace=False)
        #   (4): Conv2d(8, 1, kernel_size=(1, 1), stride=(1, 1))
        # )

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
        self.number_per_fc = nn.Linear(self.list_k[1][2], 1)  # 64->1
        # self.number_per_fc -- Linear(in_features=64, out_features=1, bias=True)

        torch.nn.init.constant_(self.number_per_fc.weight, 0)

    def forward(self, list_x, x_size: List[int]):
        pdb.set_trace()

        up_edge, up_sal, edge_feature, sal_feature, U_tmp = [], [], [], [], []

        num_f = len(list_x)
        tmp = self.up[num_f - 1](list_x[num_f - 1])
        sal_feature.append(tmp)
        U_tmp.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode="bilinear", align_corners=True))

        pdb.set_trace()

        # layer5
        up_tmp_x2 = F.interpolate(U_tmp[0], list_x[3].size()[2:], mode="bilinear", align_corners=True)
        U_tmp.append(self.DSS[0](torch.cat([up_tmp_x2, list_x[3]], dim=1)))
        tmp = self.up[3](U_tmp[-1])
        sal_feature.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode="bilinear", align_corners=True))

        # layer4
        up_tmp_x2 = F.interpolate(U_tmp[1], list_x[2].size()[2:], mode="bilinear", align_corners=True)
        up_tmp_x4 = F.interpolate(U_tmp[0], list_x[2].size()[2:], mode="bilinear", align_corners=True)
        U_tmp.append(self.DSS[1](torch.cat([up_tmp_x4, up_tmp_x2, list_x[2]], dim=1)))
        tmp = self.up[2](U_tmp[-1])
        sal_feature.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode="bilinear", align_corners=True))

        # layer3
        up_tmp_x2 = F.interpolate(U_tmp[2], list_x[1].size()[2:], mode="bilinear", align_corners=True)
        up_tmp_x4 = F.interpolate(U_tmp[1], list_x[1].size()[2:], mode="bilinear", align_corners=True)
        up_tmp_x8 = F.interpolate(U_tmp[0], list_x[1].size()[2:], mode="bilinear", align_corners=True)
        U_tmp.append(self.DSS[2](torch.cat([up_tmp_x8, up_tmp_x4, up_tmp_x2, list_x[1]], dim=1)))
        tmp = self.up[1](U_tmp[-1])
        sal_feature.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode="bilinear", align_corners=True))

        vector = F.adaptive_avg_pool2d(sal_feature[0], output_size=1)
        vector = vector.view(vector.size(0), -1)
        up_subitizing = self.number_per_fc(vector)

        # edge layer fuse
        U_tmp = list_x[0] + F.interpolate(
            (self.trans[-1](sal_feature[0])), list_x[0].size()[2:], mode="bilinear", align_corners=True
        )
        tmp = self.up[0](U_tmp)
        edge_feature.append(tmp)

        up_edge.append(F.interpolate(self.edge_score(tmp), x_size, mode="bilinear", align_corners=True))

        pdb.set_trace()

        return up_edge, edge_feature, up_sal, sal_feature, up_subitizing


class MergeLayer2(nn.Module):
    def __init__(self):
        super(MergeLayer2, self).__init__()
        self.list_k = [[32], [64, 64, 64, 64]]

        trans, up, score = [], [], []
        for i in self.list_k[0]:
            tmp = []
            tmp_up = []
            tmp_score = []
            feature_k = [[3, 1], [5, 2], [5, 2], [7, 3]]
            for idx, j in enumerate(self.list_k[1]):
                tmp.append(nn.Sequential(nn.Conv2d(j, i, 1, 1, bias=False), 
                        nn.BatchNorm2d(i), nn.ReLU(inplace=True)))
                tmp_up.append(
                    nn.Sequential(
                        nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]),
                        nn.BatchNorm2d(i),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]),
                        nn.BatchNorm2d(i),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]),
                        nn.BatchNorm2d(i),
                        nn.ReLU(inplace=True),
                    )
                )
            trans.append(nn.ModuleList(tmp))
            up.append(nn.ModuleList(tmp_up))

        self.sub_score = nn.Sequential(
            nn.Conv2d(self.list_k[0][0], self.list_k[0][0] // 4, 3, 1, 1),
            nn.BatchNorm2d(self.list_k[0][0] // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(self.list_k[0][0] // 4, 1, 1),
        )

        self.trans = nn.ModuleList(trans)
        self.up = nn.ModuleList(up)

        self.relu = nn.ReLU()

    def forward(self, list_x, list_y, x_size):
        up_score, tmp_feature = [], []
        list_y = list_y[::-1]

        for i, i_x in enumerate(list_x):
            for j, j_x in enumerate(list_y):
                tmp = F.interpolate(self.trans[i][j](j_x), i_x.size()[2:], mode="bilinear", align_corners=True) + i_x
                tmp_f = self.up[i][j](tmp)
                up_score.append(F.interpolate(self.sub_score(tmp_f), x_size, mode="bilinear", align_corners=True))
                tmp_feature.append(tmp_f)

        tmp_fea = tmp_feature[0]
        for i_fea in range(len(tmp_feature) - 1):
            tmp_fea = self.relu(
                torch.add(
                    tmp_fea,
                    F.interpolate(
                        (tmp_feature[i_fea + 1]), tmp_feature[0].size()[2:], mode="bilinear", align_corners=True
                    ),
                )
            )
        up_score.append(F.interpolate(self.sub_score(tmp_fea), x_size, mode="bilinear", align_corners=True))
        # up_score.append(F.interpolate(self.final_score(tmp_fea), x_size, mode='bilinear', align_corners=True))

        return up_score


def load_weight(model, path):
    """Load model."""

    if not os.path.exists(path):
        raise IOError(f"Model checkpoint '{path}' doesn't exist.")

    # state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = torch.load(path, map_location=torch.device("cpu"))

    target_state_dict = model.state_dict()
    for n in target_state_dict.keys():
        n2 = n.replace("seqlist", "")
        if n2 in state_dict.keys():
            p2 = state_dict[n2]
            target_state_dict[n].copy_(p2)
        else:
            raise KeyError(n)


def get_model(device):
    """Create model."""

    model_path = "models/image_deshadow.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = DeshadowModel()

    load_weight(model, checkpoint)
    model = model.to(device)
    model.eval()

    # model = torch.jit.script(model)

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_deshadow.torch"):
    #     model.save("output/image_deshadow.torch")

    return model


if __name__ == "__main__":
    model = get_model(torch.device("cpu"))
    pdb.set_trace()
