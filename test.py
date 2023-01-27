#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：STN-LPRNet-VoVNet
@File    ：test.py
@Author  ：Xuyang Huang
@Date    ：2023/1/25 11:30
"""

from datasets.label_basic import *
from datasets.data_loader import LPRDataLoader
from args import get_parser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.lprnet import build_lprnet
import time
import numpy as np
import torch
import os

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(int)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def Greedy_Decode_Eval(Net, datasets, args):
    Net = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size + 1
    batch_iterator = iter(
        DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start + length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])

        if args.mps:
            images = Variable(images.to(torch.device("mps")))
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(ALL_CHARS_DICT) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(ALL_CHARS_DICT) - 1):
                    if c == len(ALL_CHARS_DICT) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp + Tn_1 + Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))
    return Acc

def load_model(weights_fp):
    lprnet = build_lprnet("val", len(ALL_CHARS_DICT), 0)
    lprnet.load_state_dict(torch.load(weights_fp))
    return lprnet


def test(img_dir, img_size, weights_fp):
    datasets = LPRDataLoader(img_dir, img_size, False)
    all_type = datasets.all_type
    img_fps = datasets.img_paths
    args = get_parser()

    lprnet = load_model(weights_fp)
    if args.mps:
        lprnet.to(torch.device("mps"))

    for item in all_type:
        print(f"Now {item}")
        datasets.img_paths = [fp for fp in img_fps if os.path.split(fp)[1].split("_")[1] == item]
        datasets.img_paths = datasets.img_paths[:1000]
        Greedy_Decode_Eval(lprnet, datasets, args)


if __name__ == "__main__":
    test("/Users/xuyanghuang/Downloads/CCPD数据集/cropped_datasets/train", (96, 24), "./weights/LPRNet_Alternate_Train_BEST.pth")