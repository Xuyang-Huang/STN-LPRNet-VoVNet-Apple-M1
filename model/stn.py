#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：STN-LPRNet-VovNet
@File    ：stn.py
@Author  ：Xuyang Huang
@Date    ：2023/1/23 14:16 
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, hw_out):
        super(STN, self).__init__()
        # Spatial transformer localization-network 24x96
        self.localization = nn.Sequential(
            nn.Conv2d(ch_in, (ch_in+ch_out)//2, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d((ch_in+ch_out)//2, ch_out, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(ch_out*hw_out[0]*hw_out[1], 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )

    def init_weights(self):
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, inputs):
        x = self.localization(inputs)

        # x = torch.cat([x1, x2], 1)
        x = torch.flatten(x, 1)

        theta = self.fc_loc(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta.cpu(), inputs.size(), align_corners=True)
        outputs = F.grid_sample(inputs.cpu(), grid.cpu(), align_corners=True)

        return outputs.to(inputs.device)


def build_stn(phase="train"):
    Net = STN()

    if phase == "train":
        return Net.train()
    else:
        return Net.eval()