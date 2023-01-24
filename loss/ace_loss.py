#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：STN-LPRNet-VoVNet 
@File    ：ace_loss.py
@Author  ：Xuyang Huang
@Date    ：2023/1/24 15:02 
"""

import torch
import torch.nn as nn


class ACELossJS(nn.Module):
    def __init__(self, blank, class_num, reduction='mean'):
        super(ACELossJS, self).__init__()
        self.class_num = class_num
        self.reduction = reduction
        self.blank = blank

    def KL(self, p, q):
        return torch.sum(p * torch.log(p / q))

    def forward(self, x, y, target_lengths):
        predicts = torch.argmax(x, dim=1)
        lpr_length = x.shape[-1]
        loss = []
        y_start = 0
        for i in range(len(target_lengths)):
            n_k = torch.bincount(predicts[i], minlength=self.class_num)
            y_k = torch.bincount(y[y_start: y_start+target_lengths[i]], minlength=self.class_num)
            # blank_num = lpr_length - torch.sum(y_k)
            # y_k[self.blank] = blank_num
            non_zero_mask = y_k != 0
            n_p = torch.clip(n_k[non_zero_mask]/(torch.sum(n_k[non_zero_mask]) - n_k[self.blank]), 1e-5)
            y_p = y_k[non_zero_mask]/(torch.sum(y_k)-y_k[self.blank])
            loss.append(self.KL(n_p, (n_p + y_p)/2) + self.KL(y_p, (n_p + y_p)/2))

            y_start += target_lengths[i]
        loss = torch.stack(loss, dim=0)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class ACELoss(nn.Module):
    def __init__(self, blank, class_num, reduction='mean'):
        super(ACELoss, self).__init__()
        self.class_num = class_num
        self.reduction = reduction
        self.blank = blank

    def forward(self, x, y, target_lengths):
        predicts = torch.argmax(x, dim=1)
        lpr_length = x.shape[-1]
        loss = []
        y_start = 0
        for i in range(len(target_lengths)):
            n_k = torch.bincount(predicts[i], minlength=self.class_num)
            y_k = torch.bincount(y[y_start: y_start+target_lengths[i]], minlength=self.class_num)
            non_zero_mask = y_k != 0
            n_p = torch.clip(n_k[non_zero_mask]/(torch.sum(n_k[non_zero_mask]) - n_k[self.blank]), 1e-5)
            y_p = y_k[non_zero_mask]/(torch.sum(y_k)-y_k[self.blank])
            loss.append(torch.sum(-n_p * torch.log(y_p)))

            y_start += target_lengths[i]
        loss = torch.stack(loss, dim=0)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss