#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：STN-LPRNet-VovNet
@File    ：aug_data.py
@Author  ：Xuyang Huang
@Date    ：2023/1/23 14:45 
"""

# -*- coding: utf-8 -*-
"""
Created on 2019/4/17
File augment_image
@author:ZhengYuwei
@ref: https://github.com/huxiaoman7/mxnet-cnn-plate-recognition, @author Huxiaoman
功能：
进行图像增强：透视变换、HSV变化、添加背景、高斯噪声、添加污渍
"""
import cv2
import numpy as np
import math
import os
import random


class ImageAugmentation(object):
    """ 一些图像增强操作操作: 透视变换、HSV变化 """

    horizontal_sight_directions = ('left', 'mid', 'right')
    vertical_sight_directions = ('up', 'mid', 'down')

    def __init__(self):
        """ 一些图像增强参数的默认初值初始化
        """
        # 透视变换
        self.angle_horizontal = 7
        self.angle_vertical = 7
        self.angle_up_down = 5
        self.angle_left_right = 5
        self.factor = 3
        # 色调，饱和度，亮度
        self.hue_keep = 0.9
        self.saturation_keep = 0.8
        self.value_keep = 0.8

    def sight_transfer(self, images, horizontal_sight_direction, vertical_sight_direction):
        """ 对图片进行视角变换
        :param images: 图片列表
        :param horizontal_sight_direction: 水平视角变换方向
        :param vertical_sight_direction: 垂直视角变换方向
        :return:
        """
        flag = 0
        img_num = len(images)
        # 左右视角
        if horizontal_sight_direction == 'left':
            flag += 1
            images[0], matrix, size = self.left_right_transfer(images[0], is_left=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        elif horizontal_sight_direction == 'right':
            flag -= 1
            images[0], matrix, size = self.left_right_transfer(images[0], is_left=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        else:
            pass
        # 上下视角
        if vertical_sight_direction == 'down':
            flag += 1
            images[0], matrix, size = self.up_down_transfer(images[0], is_down=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        elif vertical_sight_direction == 'up':
            flag -= 1
            images[0], matrix, size = self.up_down_transfer(images[0], is_down=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        else:
            pass

        # 左下视角 或 右上视角
        if abs(flag) == 2:
            images[0], matrix, size = self.vertical_tilt_transfer(images[0], is_left_high=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)

            images[0], matrix, size = self.horizontal_tilt_transfer(images[0], is_right_tilt=True)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        # 左上视角 或 右下视角
        elif abs(flag) == 1:
            images[0], matrix, size = self.vertical_tilt_transfer(images[0], is_left_high=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)

            images[0], matrix, size = self.horizontal_tilt_transfer(images[0], is_right_tilt=False)
            for i in range(1, img_num):
                images[i] = cv2.warpPerspective(images[i], matrix, size)
        else:
            pass

        return images

    def up_down_transfer(self, img, is_down=True, angle=None):
        """ 上下视角，默认下视角
        :param img: 正面视角原始图片
        :param is_down: 是否下视角
        :param angle: 角度
        :return:
        """
        if angle is None:
            angle = self.rand_reduce(self.angle_up_down)

        shape = img.shape
        size_src = (shape[1], shape[0])
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
        # 计算图片进行投影倾斜后的位置
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_down:
            pts2 = np.float32([[interval, 0], [0, size_src[1]],
                               [size_src[0] - interval, 0], [size_src[0], size_src[1]]])
        else:
            pts2 = np.float32([[0, 0], [interval, size_src[1]],
                               [size_src[0], 0], [size_src[0] - interval, size_src[1]]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_src)
        return dst, matrix, size_src

    def left_right_transfer(self, img, is_left=True, angle=None):
        """ 左右视角，默认左视角
        :param img: 正面视角原始图片
        :param is_left: 是否左视角
        :param angle: 角度
        :return:
        """
        if angle is None:
            angle = self.angle_left_right  # self.rand_reduce(self.angle_left_right)

        shape = img.shape
        size_src = (shape[1], shape[0])
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])
        # 计算图片进行投影倾斜后的位置
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_left:
            pts2 = np.float32([[0, 0], [0, size_src[1]],
                               [size_src[0], interval], [size_src[0], size_src[1] - interval]])
        else:
            pts2 = np.float32([[0, interval], [0, size_src[1] - interval],
                               [size_src[0], 0], [size_src[0], size_src[1]]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_src)
        return dst, matrix, size_src

    def vertical_tilt_transfer(self, img, is_left_high=True):
        """ 添加按照指定角度进行垂直倾斜(上倾斜或下倾斜，最大倾斜角度self.angle_vertical一半）
        :param img: 输入图像的numpy
        :param is_left_high: 图片投影的倾斜角度，左边是否相对右边高
        """
        angle = self.rand_reduce(self.angle_vertical)

        shape = img.shape
        size_src = [shape[1], shape[0]]
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])

        # 计算图片进行上下倾斜后的距离，及形状
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[1]))
        size_target = (int(math.cos((float(angle) / 180) * math.pi) * shape[1]), shape[0] + interval)
        # 目标图像上四个顶点的坐标
        if is_left_high:
            pts2 = np.float32([[0, 0], [0, size_target[1] - interval],
                               [size_target[0], interval], [size_target[0], size_target[1]]])
        else:
            pts2 = np.float32([[0, interval], [0, size_target[1]],
                               [size_target[0], 0], [size_target[0], size_target[1] - interval]])

        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_target)
        return dst, matrix, size_target

    def horizontal_tilt_transfer(self, img, is_right_tilt=True):
        """ 添加按照指定角度进行水平倾斜(右倾斜或左倾斜，最大倾斜角度self.angle_horizontal一半）
        :param img: 输入图像的numpy
        :param is_right_tilt: 图片投影的倾斜方向（右倾，左倾）
        """
        angle = self.rand_reduce(self.angle_horizontal)

        shape = img.shape
        size_src = [shape[1], shape[0]]
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, size_src[1]], [size_src[0], 0], [size_src[0], size_src[1]]])

        # 计算图片进行左右倾斜后的距离，及形状
        interval = abs(int(math.sin((float(angle) / 180) * math.pi) * shape[0]))
        size_target = (shape[1] + interval, int(math.cos((float(angle) / 180) * math.pi) * shape[0]))
        # 目标图像上四个顶点的坐标
        if is_right_tilt:
            pts2 = np.float32([[interval, 0], [0, size_target[1]],
                               [size_target[0], 0], [size_target[0] - interval, size_target[1]]])
        else:
            pts2 = np.float32([[0, 0], [interval, size_target[1]],
                               [size_target[0] - interval, 0], [size_target[0], size_target[1]]])

        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, matrix, size_target)
        return dst, matrix, size_target

    @staticmethod
    def rand_reduce(val):
        return int(np.random.random() * val)

    def rand_perspective_transfer(self, img, factor=None, size=None):
        """ 添加投影映射畸变
        :param img: 输入图像的numpy
        :param factor: 畸变的参数
        :param size: 图片的目标尺寸，默认维持不变
        """
        if factor is None:
            factor = self.factor
        if size is None:
            size = (img.shape[1], img.shape[0])
        shape = size
        # 源图像四个顶点坐标
        pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
        # 目标图像上四个顶点的坐标
        pts2 = np.float32([[self.rand_reduce(factor), self.rand_reduce(factor)],
                           [self.rand_reduce(factor), shape[0] - self.rand_reduce(factor)],
                           [shape[1] - self.rand_reduce(factor), self.rand_reduce(factor)],
                           [shape[1] - self.rand_reduce(factor), shape[0] - self.rand_reduce(factor)]])
        # 获取 3x3的投影映射/透视变换 矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # 利用投影映射矩阵，进行透视变换
        dst = cv2.warpPerspective(img, matrix, size)
        return dst, matrix, size

    def rand_hsv(self, img):
        """ 添加饱和度光照的噪声
        :param img: BGR格式的图片
        :return 加了饱和度、光照噪声的BGR图片
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 色调，饱和度，亮度
        hsv[:, :, 0] = hsv[:, :, 0] * (self.hue_keep + np.random.random() * (1 - self.hue_keep))
        hsv[:, :, 1] = hsv[:, :, 1] * (self.saturation_keep + np.random.random() * (1 - self.saturation_keep))
        hsv[:, :, 2] = hsv[:, :, 2] * (self.value_keep + np.random.random() * (1 - self.value_keep))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    @staticmethod
    def search_file(search_path, file_format='.jpg'):
        """在指定目录search_path下，递归目录搜索指定尾缀的文件
        :param search_path: 指定的搜索目录，如：./2018年收集的素材并已校正
        :param file_format: 文件尾缀，如‘.jpg’
        :return: 该目录下所有指定尾缀文件的路径组成的list
        """
        file_path_list = []
        # 获取：1.父目录绝对路径 2.所有文件夹名字（不含路径） 3.所有文件名字
        for root_path, dir_names, file_names in os.walk(search_path):
            # 收集符合条件的文件名
            for filename in file_names:
                if filename.endswith(file_format):
                    file_path_list.append(os.path.join(root_path, filename))
        return file_path_list

    def augment(self, img, horizontal_sight_direction=None, vertical_sight_direction=None):
        """ 综合上面的加载操作，进行全流程加噪
        :param img: 待加噪图片，传进来的图片是白底黑字
        :param horizontal_sight_direction: 水平视角方向
        :param vertical_sight_direction: 垂直视角方向
        :return: 加噪后的图片，numpy数组
        """
        if horizontal_sight_direction is None:
            horizontal_sight_direction = ImageAugmentation.horizontal_sight_directions[random.randint(0, 2)]
        if vertical_sight_direction is None:
            vertical_sight_direction = ImageAugmentation.vertical_sight_directions[random.randint(0, 2)]

        # 基于视角的变换
        img = self.sight_transfer([img], horizontal_sight_direction, vertical_sight_direction)
        img = img[0]
        # 加上随机透视变换，这个其实可以不用
        img, _, _ = self.rand_perspective_transfer(img)
        img = self.rand_hsv(img)
        return img