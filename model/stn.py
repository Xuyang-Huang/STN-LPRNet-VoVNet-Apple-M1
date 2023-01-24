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
    def __init__(self):
        super(STN, self).__init__()
        # Spatial transformer localization-network 24x96
        self.loc_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # 12x48
        self.loc_conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=3, padding=1)  # 4x16
        self.loc_conv3 = nn.Conv2d(3, 32, kernel_size=6, stride=6)  # 4x16

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 2 * 32, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2),
        )

    def init_weights(self):
        # Initialize the weights/bias with identity transformation
        self.fc_loc[0].weight.data.zero_()
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, inputs):
        x1 = self.loc_conv1(inputs)
        x1 = self.loc_conv2(x1)
        x2 = self.loc_conv3(inputs)

        x = torch.cat([x1, x2], 1)
        x = torch.flatten(x, 1)

        theta = self.fc_loc(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta.cpu(), inputs.size())
        outputs = F.grid_sample(inputs.cpu(), grid.cpu())

        return outputs.to(inputs.device)


def build_stn(phase="train"):
    Net = STN()

    if phase == "train":
        return Net.train()
    else:
        return Net.eval()