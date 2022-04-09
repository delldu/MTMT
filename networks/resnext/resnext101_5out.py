import torch
from torch import nn

from . import resnext_101_32x4d_
from .config import resnext_101_32_path
import pdb

class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = resnext_101_32x4d_.get_resnext_101_32x4d()
        # xxxx8888 net.load_state_dict(torch.load(resnext_101_32_path))
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]
        # (Pdb) self.layer0
        # Sequential(
        #   (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (2): ReLU()
        # )


    def forward(self, x):
        layers = []
        layer0 = self.layer0(x)
        layers.append(layer0)
        layer1 = self.layer1(layer0)
        layers.append(layer1)
        layer2 = self.layer2(layer1)
        layers.append(layer2)
        layer3 = self.layer3(layer2)
        layers.append(layer3)
        layer4 = self.layer4(layer3)
        layers.append(layer4)
        return layers
