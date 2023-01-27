#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：STN-LPRNet-VoVNet 
@File    ：inference.py
@Author  ：Xuyang Huang
@Date    ：2023/1/26 00:30 
"""

from model.lprnet import build_lprnet
from datasets.label_basic import *
import torch
import cv2
import numpy as np


def load_model(weights_fp):
    lprnet = build_lprnet("val", len(ALL_CHARS_DICT), 0)
    lprnet.load_state_dict(torch.load(weights_fp))
    return lprnet


def inference(net, img):
    net = net.eval()
    logits = net(torch.Tensor(img[np.newaxis, :, :, :]))
    logits = logits.squeeze().detach().numpy()
    predicts = np.argmax(logits, 0)

    res = []
    last_c = ""
    for item in predicts:
        if item == len(ALL_CHARS_DICT) - 1:
            continue
        if item == last_c:
            continue

        last_c = item
        if item != ALL_CHARS_DICT["-"]:
            res.append(ALL_CHARS_DICT_INVERT[item])
    return " ".join(res)

if __name__ == "__main__":
    lprnet = load_model("./weights/LPRNet_Alternate_Train_BEST.pth")
    raw_image = cv2.imread("./test_img/ccpd_rotate_1.jpg")
    image = cv2.resize(raw_image, (96, 24))
    image = np.transpose(image, (2, 0, 1))
    result = inference(lprnet, image)
    print(result)

