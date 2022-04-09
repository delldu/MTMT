from torch import nn
import pdb


class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = get_resnext_101_32x4d()
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3:5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

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


class SeqMapReduce(nn.Module):
    def __init__(self, seqlist0, seqlist1):
        super(SeqMapReduce, self).__init__()
        self.seqlist0 = seqlist0
        self.seqlist1 = seqlist1

    def forward(self, x):
        return self.seqlist0(x) + self.seqlist1(x)


def get_resnext_101_32x4d():
    resnext_101_32x4d = nn.Sequential(
        nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
        nn.Sequential(
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(64, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),
                            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(256),
                    ),
                    nn.Sequential(
                        nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(256),
                    ),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),
                            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(256),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),
                            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(256),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
        ),
        nn.Sequential(
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(512),
                    ),
                    nn.Sequential(
                        nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(512),
                    ),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(512),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(512),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(512),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
        ),
        nn.Sequential(
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Sequential(
                        nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(1024),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
        ),
        nn.Sequential(
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(),
                            nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(2048),
                    ),
                    nn.Sequential(
                        nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(2048),
                    ),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(),
                            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(2048),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(
                SeqMapReduce(
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(),
                            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(),
                        ),
                        nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(2048),
                    ),
                    nn.Identity(),
                ),
                nn.ReLU(),
            ),
        ),
        nn.AvgPool2d((7, 7), (1, 1)),
    )
    return resnext_101_32x4d
