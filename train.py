#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：STN-LPRNet-VovNet
@File    ：train.py
@Author  ：Xuyang Huang
@Date    ：2023/1/23 14:30
"""
import random

from model.lprnet import build_lprnet
from args import get_parser
from test import Greedy_Decode_Eval, collate_fn
from datasets.data_loader import LPRDataLoader
from datasets.label_basic import *
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os
import math

device = None


def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


def alternate_training(iters, net, interval):
    if (iters // interval) % 2 == 0:
        if not net.freeze_lpr_flag or net.freeze_stn_flag:
            net.unfreeze_stn()
            net.freeze_lpr()
            print("Freeze lpr")
        return

    if not net.freeze_stn_flag or net.freeze_lpr_flag:
        net.freeze_stn()
        net.unfreeze_lpr()
        print("Freeze stn")


def train():
    args = get_parser()
    seed_everything(1)

    T_length = args.lpr_max_len  # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    lprnet = build_lprnet(phase="train", class_num=len(ALL_CHARS_DICT),
                          dropout_rate=args.dropout_rate)
    global device
    device = torch.device("mps" if args.mps else "cpu")

    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("load pretrained model successful!")
    else:
        def xavier(tensor):
            shape = tensor.shape
            tensor = tensor.reshape([1, -1])
            return nn.init.xavier_uniform_(tensor).reshape(shape)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(m.state_dict()[key][...])
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.stn.apply(weights_init)
        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")

    optimizer_params = [
        {'params': lprnet.stn.parameters(), 'weight_decay': args.stn_weight_decay, 'lr': 1e-3},
        {'params': lprnet.backbone.parameters(), 'weight_decay': args.weight_decay},
        {'params': lprnet.container.parameters(), 'weight_decay': args.weight_decay}
    ]
    optimizer = optim.AdamW(optimizer_params, lr=args.learning_rate, betas=(args.momentum, 0.99))
    # lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    train_dataset = LPRDataLoader(train_img_dirs, args.img_size, aug=True)
    test_dataset = LPRDataLoader(test_img_dirs, args.img_size, aug=False)

    epoch_size = len(train_dataset) // args.train_batch_size
    max_iter = args.max_epoch * epoch_size

    ctc_loss = nn.CTCLoss(blank=len(ALL_CHARS) - 1, reduction='mean')  # reduction: 'none' | 'mean' | 'sum'

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    lprnet = lprnet.to(device)

    best_acc = 0
    start_stn_iter = 0
    for iteration in range(start_iter, max_iter):
        if (epoch >= args.stn_epoch or best_acc > args.stn_acc) and not lprnet.stn_switch:
            lprnet.stn_switch = True
            for _stn in lprnet.stn.children():
                _stn.init_weights()
            start_stn_iter = iteration
            print("start STN")
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(
                DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers,
                           collate_fn=collate_fn))
            loss_val = 0
            epoch += 1

        if iteration != 0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + TRAIN_NAME + '_iteration_' + repr(iteration) + '.pth')

        if (iteration + 1) % args.test_interval == 0:
            acc = Greedy_Decode_Eval(lprnet, test_dataset, args)
            if acc > best_acc:
                best_acc = acc
                torch.save(lprnet.state_dict(), args.save_folder + TRAIN_NAME + '_BEST.pth')
            lprnet = lprnet.train()  # should be switch to train mode
            # scheduler.step(acc)

        start_time = time.time()
        # load train data
        images, labels, lengths = next(batch_iterator)
        # labels = np.array([el.numpy() for el in labels]).T
        # print(labels)
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        # update lr
        # lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        if args.mps:
            images = Variable(images, requires_grad=False).to(device)
            labels = Variable(labels, requires_grad=False).to(device)
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # forward
        logits = lprnet(images)
        log_probs = logits.permute([2, 0, 1])  # for ctc loss: T x N x C
        # print(labels.shape)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # log_probs = log_probs.detach().requires_grad_()
        # print(log_probs.shape)
        # backprop
        optimizer.zero_grad()

        if args.mps:
            log_probs = log_probs.cpu()
            labels = labels.cpu()

        c_loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)

        loss = c_loss

        if loss.item() == np.inf:
            continue
        loss.to(device)
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        end_time = time.time()
        if iteration % 100 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) +
                  ' || CTC Loss: %.4f||' % (c_loss.item()) + 'Batch time: %.4f sec. ||' % (end_time - start_time) +
                  'LR1: %.8f' % (optimizer.state_dict()['param_groups'][0]['lr']) +
                  'LR2: %.8f' % (optimizer.state_dict()['param_groups'][1]['lr']))
        if args.alternate_training and lprnet.stn_switch:
            alternate_training(iteration - start_stn_iter, lprnet, args.alternate_training_interval)
        if iteration - start_stn_iter < args.stn_warm_iter and lprnet.stn_switch:
            optimizer.param_groups[0]['lr'] = warm_up(1e-3, iteration - start_stn_iter, args.stn_warm_iter, epoch_size)
    # final test
    print("Final test Accuracy:")
    Greedy_Decode_Eval(lprnet, test_dataset, args)

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + f'Final_{TRAIN_NAME}_model.pth')


def warm_up(lr, step, iter_warm, epoch_size):
    epoch = (step+1) / epoch_size
    epoch_warm = iter_warm / epoch_size
    return lr * min(epoch ** (-0.5), epoch * epoch_warm ** (-1.5))


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


if __name__ == "__main__":
    global TRAIN_NAME
    TRAIN_NAME = "LPRNet_Alternate_Train"
    train()
