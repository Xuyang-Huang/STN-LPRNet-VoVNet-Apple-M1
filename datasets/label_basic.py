#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：STN-LPRNet-VovNet
@File    ：label_basic.py
@Author  ：Xuyang Huang
@Date    ：2023/1/23 14:39 
"""

PROVINCE_CHAR = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

WORD_CHAR = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "-", "#"]

ALL_CHARS = PROVINCE_CHAR
ALL_CHARS.extend(WORD_CHAR)
ALL_CHARS_DICT = {item: i for i, item in enumerate(ALL_CHARS)}
ALL_CHARS_DICT_INVERT = {i: item for i, item in enumerate(ALL_CHARS)}
