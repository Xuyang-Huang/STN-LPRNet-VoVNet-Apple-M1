#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：STN-LPRNet-VovNet
@File    ：data_loader.py
@Author  ：Xuyang Huang
@Date    ：2023/1/23 14:38 
"""

from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import os, glob

from .label_basic import *
from .aug_data import ImageAugmentation
class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, aug=True):

        self.aug = aug
        augment = ImageAugmentation()
        self.aug_func = augment.augment
        self.img_paths = glob.glob(os.path.join(img_dir, "*"))
        random.shuffle(self.img_paths)
        self.img_size = imgSize

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        img = cv2.imread(filename)
        if self.aug and np.random.rand() > 0.3:
            img = self.aug_func(img)
        img = cv2.resize(img, self.img_size)

        _, file_name = os.path.split(filename)
        label_str = file_name.split("_")[2][:-4].split("-")
        label = list()
        last_c = "="
        for c in label_str:
            if c == last_c:
                label.append(ALL_CHARS_DICT["-"])
            label.append(int(c))
            last_c = c

        img = np.transpose(img, (2, 0, 1))
        return img.astype(np.float32), label, len(label)