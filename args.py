#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：STN-LPRNet-VoVNet 
@File    ：args.py
@Author  ：Xuyang Huang
@Date    ：2023/1/25 11:50 
"""
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=300, help='epoch to train the network')
    parser.add_argument('--img_size', default=[96, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default="/Users/xuyanghuang/Downloads/CCPD数据集/cropped_datasets/train",
                        help='the train images path')
    parser.add_argument('--test_img_dirs', default="/Users/xuyanghuang/Downloads/CCPD数据集/cropped_datasets/val",
                        help='the test images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.003, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=19, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=64, help='training batch size.')
    parser.add_argument('--test_batch_size', default=32, help='testing batch size.')
    parser.add_argument('--phase_train', default="train", type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--mps', default=True, type=bool, help='Use mps to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=1000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[3, 50, 301], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')
    parser.add_argument('--stn_epoch', default=10, help='pretrained base model')
    parser.add_argument('--stn_acc', default=0.7, help='pretrained base model')
    parser.add_argument('--stn_weight_decay', default=2e-5)
    parser.add_argument('--alternate_training', default=False)
    parser.add_argument('--alternate_training_interval', default=12000)
    parser.add_argument('--stn_warm_iter', default=5000)

    args = parser.parse_args()

    return args