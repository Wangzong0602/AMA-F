from __future__ import absolute_import

from torchvision.transforms import *

#from PIL import Image
import random
import math
#import numpy as np
#import torch


class ChannelAdap(object):
    """ Adaptive selects a channel or two channels. GRAY
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image. 擦除区域与输入图像的最小比例。
         sh: Maximum proportion of erased area against input image. 擦除区域相对于输入图像的最大比例。
         r1: Minimum aspect ratio of erased area. 擦除区域的最小纵横比。
         mean: Erasing value.
         RRR、GGG、BBB、原图各为1/4概率
    """
    
    def __init__(self, probability = 0.5):
        self.probability = probability

       
    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
            # return img

        idx = random.randint(0, 3)  # 0 1 2 3随机取值
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            img = img

        return img
        
        
class ChannelAdapGray(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
         RRR、GGG、BBB各为1/4概率，原图、灰度各为1/8概率

    """
    
    def __init__(self, probability = 0.5):
        self.probability = probability

       
    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
            # return img

        idx = random.randint(0, 3)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            if random.uniform(0, 1) > self.probability:
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:] # 生成灰度图片
                img[0,:,:] = tmp_img
                img[1,:,:] = tmp_img
                img[2,:,:] = tmp_img
        return img

class ChannelRandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image. 擦除区域与输入图像的最小比例。
         sh: Maximum proportion of erased area against input image. 擦除区域相对于输入图像的最大比例。
         r1: Minimum aspect ratio of erased area. 擦除区域的最小纵横比。
         mean: Erasing value.
         原图、ChannelRandomErasing概率各为1/2，ChannelRandomErasing每个通道擦除区域相同，使用imagenet不同通道均值填充。代码与RandomErasing相同。
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)  # 图片纵横比0.3-10/3

            h = int(round(math.sqrt(target_area * aspect_ratio)))  # 获得擦除区域的高
            w = int(round(math.sqrt(target_area / aspect_ratio)))  # 获得擦除区域的宽

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img