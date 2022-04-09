# from functools import reduce
# import torch
import torch.nn as nn

class SeqMapReduce(nn.Module):
    def __init__(self, seqlist0, seqlist1):
        super(SeqMapReduce, self).__init__()
        self.seqlist0 = seqlist0
        self.seqlist1 = seqlist1

    def forward(self, x):
        return self.seqlist0(x) + self.seqlist1(x)

def get_resnext_101_32x4d():
    resnext_101_32x4d = nn.Sequential(  # Sequential,
        nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                    nn.Sequential(  # Sequential,
                        nn.Sequential(  # Sequential,
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
                    nn.Sequential(  # Sequential,
                        nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                        nn.BatchNorm2d(256),
                    ),
                ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                    nn.Sequential(  # Sequential,
                        nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                    nn.Sequential(  # Sequential,
                        nn.Sequential(  # Sequential,
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
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(512),
                          ),
                          ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(1024),
                          ),
                          ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
        nn.Sequential(  # Sequential,
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
                          nn.Sequential(  # Sequential,
                              nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                              nn.BatchNorm2d(2048),
                          ),
                          ),
                nn.ReLU(),
            ),
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
            nn.Sequential(  # Sequential,
                SeqMapReduce(
                          nn.Sequential(  # Sequential,
                              nn.Sequential(  # Sequential,
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
