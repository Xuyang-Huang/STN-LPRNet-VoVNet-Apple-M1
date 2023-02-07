#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：LPRNet
@File    ：lprnet.py
@Author  ：Xuyang Huang
@Date    ：2023/1/23 12:42
"""
import torch.nn as nn
import torch
from model.stn import STN


class OSA(nn.Module):
    def __init__(self, ch_in, ch_m, ch_out, conv_num):
        super(OSA, self).__init__()
        convs = [nn.Conv2d(ch_in, ch_m, kernel_size=(3, 3), padding=1)]
        for _ in range(conv_num - 1):
            convs.append(nn.Conv2d(ch_m, ch_m, kernel_size=(3, 3), padding=1))
        self.list = nn.ModuleList(convs)
        self.down_features = nn.Conv2d(ch_m * conv_num, ch_out, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(ch_out)

    def conv_and_append(self, x, conv_func):
        x = conv_func(x)
        x = nn.functional.relu(x)
        return x

    def forward(self, x):
        features = []
        for layer in self.list:
            x = layer(x)
            x = nn.functional.relu(x)
            features.append(x)

        x = torch.cat(features, 1)
        x = self.down_features(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        return x


class LPRNet(nn.Module):
    def __init__(self, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.freeze_stn_flag = False
        self.freeze_lpr_flag = False
        self.phase = phase
        self.class_num = class_num
        self.stn_switch = False
        self.stn = nn.Sequential(STN(3, 16, 5, [6, 24]),
                                 STN(64, 16, 3, [3, 12]),
                                 STN(128, 32, 3, [3, 12]))
        self.backbone = nn.Sequential(
            # input: 24x96
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),  # 12x48
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 3 12x48
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            OSA(ch_in=64, ch_m=64, ch_out=128, conv_num=3),  # 6 12x48
            nn.MaxPool2d(kernel_size=2, stride=2),  # 6x24
            OSA(ch_in=128, ch_m=96, ch_out=256, conv_num=3),  # 6x24
        )

        self.container = nn.Sequential(
            # input: 6x24
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(6, 6), stride=1),  # 1x19
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU()
        )

        # self.freeze_stn()

    @staticmethod
    def freeze(model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze(model):
        for param in model.parameters():
            param.requires_grad = True

    def freeze_stn(self):
        self.freeze_stn_flag = True
        self.freeze(self.stn)

    def freeze_lpr(self):
        self.freeze_lpr_flag = True
        self.freeze(self.backbone)
        self.freeze(self.container)

    def unfreeze_stn(self):
        self.freeze_stn_flag = False
        self.unfreeze(self.stn)

    def unfreeze_lpr(self):
        self.freeze_lpr_flag = False
        self.unfreeze(self.backbone)
        self.unfreeze(self.container)

    def forward(self, x):
        if self.stn_switch:
            x = self.stn[0](x)
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)

            if i == 3 and self.stn_switch:
                x = self.stn[1](x)

            if i == 6 and self.stn_switch:
                x = self.stn[2](x)
        # x = self.backbone(x)
        logits = self.container(x)
        return logits.reshape([-1, self.class_num, 19])


def build_lprnet(phase="train", class_num=66, dropout_rate=0.5):
    Net = LPRNet(phase, class_num, dropout_rate)

    if phase == "train":
        return Net.train()
    else:
        Net.stn_switch = True
        return Net.eval()


if __name__ == "__main__":
    from torchsummary import summary
    import numpy as np

    model = build_lprnet("train", 66, 0.5)
    model.stn_switch = True
    summary(model.stn, (3, 24, 96))
    summary(model.backbone, (3, 24, 96))
    test_data = model(torch.Tensor(np.random.random([1, 3, 24, 96])))
