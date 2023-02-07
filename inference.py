#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：STN-LPRNet-VoVNet 
@File    ：inference.py
@Author  ：Xuyang Huang
@Date    ：2023/1/26 00:30 
"""

from model.lprnet import build_lprnet
from model.stn import sampling
from datasets.label_basic import *
import torch
import cv2
import numpy as np


def load_model(weights_fp):
    lprnet = build_lprnet("val", len(ALL_CHARS_DICT), 0)
    lprnet.load_state_dict(torch.load(weights_fp))
    return lprnet


def visualize_stn(net, img):
    img = torch.Tensor(img[np.newaxis, :, :, :])
    net = net.eval()
    theta = net.stn[0].get_theta(img)
    img_trans = img
    img_trans = sampling(theta, img_trans)
    x = net.stn[0](img)
    for i, layer in enumerate(net.backbone.children()):
        x = layer(x)

        if i == 3:
            theta = net.stn[1].get_theta(x)
            img_trans = sampling(theta, img_trans)
            x = net.stn[1](x)

        if i == 6:
            theta = net.stn[2].get_theta(x)
            img_trans = sampling(theta, img_trans)
            x = net.stn[2](x)
    img_trans = img_trans.squeeze().detach().numpy().transpose([1, 2, 0]).astype(np.uint8)
    cv2.namedWindow("img")
    cv2.imshow("img", img_trans)

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
    lprnet = load_model("./weights/LPRNet_Train_BEST.pth")
    raw_image = cv2.imread("./test_img/ccpd_fn_1.jpg")
    image = cv2.resize(raw_image, (96, 24))

    cv2.namedWindow("og img")
    cv2.imshow("og img", image)

    image = np.transpose(image, (2, 0, 1))
    result = inference(lprnet, image)
    visualize_stn(lprnet, image)
    print(result)
    cv2.waitKey()

