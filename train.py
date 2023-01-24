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

device = None

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=300, help='epoch to train the network')
    parser.add_argument('--img_size', default=[96, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default="/Users/xuyanghuang/Downloads/CCPD数据集/cropped_datasets/train", help='the train images path')
    parser.add_argument('--test_img_dirs', default="/Users/xuyanghuang/Downloads/CCPD数据集/cropped_datasets/val", help='the test images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.001, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=19, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=32, help='training batch size.')
    parser.add_argument('--test_batch_size', default=32, help='testing batch size.')
    parser.add_argument('--phase_train', default="train", type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--mps', default=True, type=bool, help='Use mps to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=1000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[5, 50, 100, 200, 301], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')
    parser.add_argument('--stn_epoch', default=10, help='pretrained base model')
    parser.add_argument('--stn_acc', default=0.6, help='pretrained base model')

    args = parser.parse_args()

    return args


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


def train():
    args = get_parser()

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

    optimizer = optim.AdamW(lprnet.parameters(), lr=args.learning_rate)
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
    for iteration in range(start_iter, max_iter):
        if (epoch >= args.stn_epoch or best_acc > args.stn_acc) and lprnet.stn_switch == False:
            lprnet.stn_switch = True
            lprnet.stn.init_weights()
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
            # lprnet.train() # should be switch to train mode

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

        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
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
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (optimizer.state_dict()['param_groups'][0]['lr']))
    # final test
    print("Final test Accuracy:")
    Greedy_Decode_Eval(lprnet, test_dataset, args)

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + f'Final_{TRAIN_NAME}_model.pth')


def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
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
            images = Variable(images.to(device))
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


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


if __name__ == "__main__":
    global TRAIN_NAME
    TRAIN_NAME = "LPRNet"
    seed_everything(1)
    train()
