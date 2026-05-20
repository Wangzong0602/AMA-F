import numpy as np
from PIL import Image
import torch.utils.data as data
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision.transforms as transforms
import random
import math
import torch
import torch.nn as nn
class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.


    """
    
    def __init__(self, gray = 2):
        self.gray = gray

    def __call__(self, img):
    
        idx = random.randint(0, self.gray)
        
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
            tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
            img[0,:,:] = tmp_img
            img[1,:,:] = tmp_img
            img[2,:,:] = tmp_img
        return img


class SpatialAttention(nn.Module):
    """CBAM空间注意力模块（轻量级CPU版本）"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入x: [C, H, W]
        max_out, _ = torch.max(x, dim=0, keepdim=True)  # [1, H, W]
        avg_out = torch.mean(x, dim=0, keepdim=True)  # [1, H, W]
        concat = torch.cat([max_out, avg_out], dim=0)  # [2, H, W]
        sa_map = self.conv(concat.unsqueeze(0))  # [1, 1, H, W]
        return self.sigmoid(sa_map.squeeze(0))  # [1, H, W]


class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):

        data_dir = '/home/xiegengsheng/wangguangzhou/datasets/SYSU-MM01/'
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(hue=0.5),
            transforms.RandomErasing(p=0.5, value=(0.4914, 0.4822, 0.4465)),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, value=(0.4914, 0.4822, 0.4465)),
            normalize,
            ChannelRandomErasing(probability=0.5)])

        self.transform_color1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, value=(0.4914, 0.4822, 0.4465)),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)])

        self.spatial_att = SpatialAttention(kernel_size=7)
        self.to_tensor = transforms.ToTensor()
    # def __getitem__(self, index):
    #
    #     img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
    #     img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
    #
    #     x10 = self.transform_color(img1)
    #     x2 = self.transform_thermal(img2)
    #
    #     patch_size = (16, 16)
    #     keep_prob = 0.1
    #
    #     x11 = x2
    #     # 遍历新图片的每个patch
    #     for i in range(0, x10.shape[1], patch_size[0]):
    #         for j in range(0, x10.shape[2], patch_size[1]):
    #             # 从 x1 或 x2 中选择patch
    #             if np.random.rand() <= keep_prob:
    #                 x11[:, i:i + patch_size[0], j:j + patch_size[1]] = x10[:, i:i + patch_size[0], j:j + patch_size[1]]
    #
    #     return x10, x11, x2, target1, target2

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        x10 = self.transform_color(img1)
        x11 = self.transform_color1(img1)
        x2 = self.transform_thermal(img2)

        # ============== 核心修改：基于空间注意力的自适应条纹替换 ==============
        # 将原始图像转换为Tensor计算注意力
        # 注意：这里使用原始图像img1而不是变换后的x11，因为变换会改变图像内容
        img_pil = Image.fromarray(img1.astype('uint8'))
        vis_tensor = self.to_tensor(img_pil).squeeze(0)  # [C, H, W]

        # 计算空间注意力图
        with torch.no_grad():
            sa_map = self.spatial_att(vis_tensor)  # [1, H, W]

        # 获取图像尺寸
        H, W = img_pil.size[1], img_pil.size[0]
        stripe_h = H // 4

        # 计算每个条纹的注意力权重
        stripe_weights = []
        for j in range(4):
            start = j * stripe_h
            end = (j + 1) * stripe_h if j < 3 else H
            stripe_region = sa_map[:, start:end, :]
            stripe_weights.append(torch.mean(stripe_region).item())

        # 归一化权重 (0~1范围)
        min_w = min(stripe_weights)
        max_w = max(stripe_weights)
        if max_w - min_w > 1e-5:
            norm_weights = [(w - min_w) / (max_w - min_w) for w in stripe_weights]
        else:
            norm_weights = [0.5] * 4  # 所有区域权重相等

        # 应用自适应条纹替换
        for j in range(4):
            start = j * stripe_h
            end = (j + 1) * stripe_h if j < 3 else H

            # 动态替换概率：基础概率 * (1 - 注意力权重)
            # 高注意力区域(重要区域)替换概率降低
            # 上半身区域(j<3)基础概率0.3，下半身(j>=3)基础概率0.7
            base_prob = 0.3 if j < 2 else 0.7
            replace_prob = base_prob * (1.0 - norm_weights[j])

            if random.uniform(0, 1) < replace_prob:
                x11[:, j * stripe_h: (j + 1) * stripe_h, :] = x2[:, j * stripe_h: (j + 1) * stripe_h, :]
        # ============== 修改结束 ==============

        return x10, x11, x2, target1, target2
        # stripe_h = int(x11.size(1) / 6)
        # for j in range(6):
        #     # 上半身区域(前3个条纹)有更高概率保留原始模态
        #     if j < 3:  # 上半身区域
        #         prob = 0.3  # 30%概率替换为另一模态特征
        #     else:  # 下半身区域
        #         prob = 0.7  # 70%概率替换为另一模态特征
        #
        #     if random.uniform(0, 1) < prob:
        #         x11[:, j * stripe_h: (j + 1) * stripe_h, :] = x2[:, j * stripe_h: (j + 1) * stripe_h, :]
        #
        # return x10, x11, x2, target1, target2


    def __len__(self):
        return len(self.train_color_label)


class LLCMData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_vis.txt'
        train_thermal_list = data_dir + 'idx/train_nir.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            # print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),  # 本质为RandomErasing
            ChannelAdapGray(probability=0.5)])  # 1/8灰度

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5)])

        self.transform_color1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)])  # GRAY RRR、GGG、BBB各为1/3概率，单通道RandomErasing扩展到三通道

        self.spatial_att = SpatialAttention(kernel_size=7)
        self.spatial_att.eval()  # 设置为评估模式，不更新参数

        self.to_tensor = transforms.ToTensor()
    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(img1)
        img2_trans = self.transform_thermal(img2)

        # ============== 核心修改：基于空间注意力的自适应条纹替换 ==============
        # 使用原始可见光图像计算空间注意力
        img_pil = Image.fromarray(img1.astype('uint8'))
        vis_tensor = self.to_tensor(img_pil)  # [C, H, W]

        # 计算空间注意力图
        with torch.no_grad():
            sa_map = self.spatial_att(vis_tensor)  # [1, H, W]

        # 获取图像尺寸
        H, W = img_pil.size[1], img_pil.size[0]
        stripe_h = H // 6  # 分为6个水平条带

        # 计算每个条带的注意力权重
        stripe_weights = []
        for j in range(6):
            start = j * stripe_h
            end = (j + 1) * stripe_h if j < 5 else H
            stripe_region = sa_map[:, start:end, :]
            stripe_weights.append(torch.mean(stripe_region).item())

        # 归一化权重 (0~1范围)
        min_w = min(stripe_weights)
        max_w = max(stripe_weights)
        if max_w - min_w > 1e-5:
            norm_weights = [(w - min_w) / (max_w - min_w) for w in stripe_weights]
        else:
            norm_weights = [0.5] * 6  # 所有区域权重相等

        # 应用自适应条纹替换
        for j in range(6):
            start = j * stripe_h
            end = (j + 1) * stripe_h if j < 5 else H

            # 动态替换概率公式：基础概率 * (1 - 注意力权重)
            # 上半身区域(j<3)基础概率0.3，下半身(j>=3)基础概率0.7
            base_prob = 0.3 if j < 3 else 0.7
            replace_prob = base_prob * (1.0 - norm_weights[j])

            # 根据概率决定是否替换当前条带
            if random.uniform(0, 1) < replace_prob:
                img1_1[:, start:end, :] = img2_trans[:, start:end, :]
        # ============== 修改结束 ==============

        return img1_0, img1_1, img2_trans, target1, target2

    def __len__(self):
        return len(self.train_color_label)

        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        data_dir = '/home/xiegengsheng/wangguangzhou/datasets/RegDB/'
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])
            
        self.transform_color = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5)])
            
        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2)])
        self.spatial_att = SpatialAttention(kernel_size=7)
        self.spatial_att.eval()  # 设置为评估模式

        self.to_tensor = transforms.ToTensor()
    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(img1)
        img2_trans = self.transform_thermal(img2)

        img_pil = Image.fromarray(img1.astype('uint8'))
        vis_tensor = self.to_tensor(img_pil)  # [C, H, W]
        with torch.no_grad():
            sa_map = self.spatial_att(vis_tensor)  # [1, H, W]

        # 获取图像尺寸
        H, W = img_pil.size[1], img_pil.size[0]
        stripe_h = H // 6

        # 计算每个条纹的注意力权重
        stripe_weights = []
        for j in range(6):
            start = j * stripe_h
            end = (j + 1) * stripe_h if j < 5 else H
            stripe_region = sa_map[:, start:end, :]
            stripe_weights.append(torch.mean(stripe_region).item())

        # 归一化权重 (0~1范围)
        min_w = min(stripe_weights)
        max_w = max(stripe_weights)
        if max_w - min_w > 1e-5:
            norm_weights = [(w - min_w) / (max_w - min_w) for w in stripe_weights]
        else:
            norm_weights = [0.5] * 6  # 所有区域权重相等

        # 应用自适应条纹替换
        for j in range(6):
            start = j * stripe_h
            end = (j + 1) * stripe_h if j < 5 else H

            # 动态替换概率公式：基础概率 * (1 - 注意力权重)
            # 上半身区域(j<3)基础概率0.3，下半身(j>=3)基础概率0.7
            base_prob = 0.3 if j < 3 else 0.7
            replace_prob = base_prob * (1.0 - norm_weights[j])

            if random.uniform(0, 1) < replace_prob:
                img1_1[:, j * stripe_h: (j + 1) * stripe_h, :] = img2_trans[:, j * stripe_h: (j + 1) * stripe_h, :]
        # ============== 修改结束 ==============

        return img1_0, img1_1, img2_trans, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class Dataloader_MEM(data.Dataset):
    def __init__(self, data_dir, dataset=None, size=(288, 144)):
        self.train_color_label = dataset.train_color_label
        self.train_thermal_label = dataset.train_thermal_label
        self.train_color_image = dataset.train_color_image
        self.train_thermal_image = dataset.train_thermal_image
        self.choose = 0

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5)])

    def __getitem__(self, index):
        if self.choose == 0:
            img1, target1 = self.train_color_image[index], self.train_color_label[index]
            img1_0 = self.transform_color(img1)
            return img1_0, target1
        elif self.choose == 1:
            img2, target2 = self.train_thermal_image[index], self.train_thermal_label[index]
            img2 = self.transform_thermal(img2)
            return img2, target2

    def __len__(self):
        if self.choose == 0:
            return len(self.train_color_label)
        elif self.choose == 1:
            return len(self.train_thermal_label)
        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label