#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：STN-LPRNet-VovNet
@File    ：crop_ccpd.py
@Author  ：Xuyang Huang
@Date    ：2023/1/23 11:48
"""

import glob, os, random
import cv2
from label_basic import *


def proc_img(fp, save_folder):
    folder, name = os.path.split(fp)
    name_split = name.split("-")

    bbox = get_bbox(name_split[2])

    label = get_label(name_split[4])

    img = cv2.imread(fp)
    crop_img = img[bbox[0][1]: bbox[1][1], bbox[0][0]: bbox[1][0], :]
    file_name = os.path.basename(folder) + "_" + "-".join([str(item) for item in label]) + ".jpg"
    cv2.imwrite(os.path.join(save_folder, file_name), crop_img)


def get_bbox(bbox_str):
    bbox = bbox_str.split("_")
    bbox = [[int(item.split("&")[0]), int(item.split("&")[1])] for item in bbox]
    return bbox


def get_label(label_str):
    label_split = label_str.split("_")
    label = [ALL_CHARS_DICT[PROVINCE_CHAR[int(label_str[0])]]]
    label.extend([ALL_CHARS_DICT[WORD_CHAR[int(item)]] for item in label_split[1:]])
    return label


if __name__ == "__main__":
    random.seed(1)
    ccpd_folder = "/Users/xuyanghuang/Downloads/CCPD数据集/ccpd2019/"
    save_folder = "/Users/xuyanghuang/Downloads/CCPD数据集/cropped_datasets/"
    skip_folder_name = ["ccpd_blur", "ccpd_np"]
    max_num_per_type = 50e3
    all_type_folders = glob.glob(os.path.join(ccpd_folder, "ccpd*"))
    for type_folder in all_type_folders:
        if any([skip_folder in type_folder for skip_folder in skip_folder_name]):
            continue
        all_file_paths = glob.glob(os.path.join(type_folder, "*"))
        random.shuffle(all_file_paths)
        for i, file_path in enumerate(all_file_paths):
            if i > max_num_per_type:
                print(f"Finish {type_folder}")
                break
            if i % 6 == 0:
                save_type = "val"
            else:
                save_type = "train"
            if not os.path.exists(os.path.join(save_folder, save_type)):
                os.makedirs(os.path.join(save_folder, save_type))
            proc_img(file_path, os.path.join(save_folder, save_type))