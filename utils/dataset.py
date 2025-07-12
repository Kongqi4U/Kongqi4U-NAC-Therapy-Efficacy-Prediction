import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import numpy as np


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'Training_Images/*.jpg'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 生成标签路径，使用os.path来处理路径
        label_path = os.path.join(
            os.path.dirname(image_path).replace('Training_Images', 'Training_Labels'),
            os.path.basename(image_path).replace('.jpg', '.png')
        )
        # 确保文件存在
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        if not os.path.exists(label_path):
            raise ValueError(f"Label file not found: {label_path}")

        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图

        if image is None or label is None:
            raise ValueError(f"Failed to load image or label: {image_path}, {label_path}")

        # 调整图片大小
       #  image = cv2.resize(image, (256, 256))
        # label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255

        # 随机进行数据增强
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)

        # 调整为单通道的图片
        # image = image.reshape(3, image.shape[0], image.shape[1])
        # label = label.reshape(3, label.shape[0], label.shape[1])
        # 转换为灰度图像（如果是 RGB 图像）
        if len(image.shape) == 3:  # 如果是 RGB 图像
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (256, 256))  # 调整为 256x256
        image = image.reshape(1, image.shape[0], image.shape[1])  # 变为 (1, 256, 256)

        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        label = label.reshape(1, label.shape[0], label.shape[1])  # 标签处理为单通道 (1, 256, 256)

        # if len(image.shape) == 3:  # 如果图像是 RGB（3 通道）
           #  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


# 用于测试
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("../images/cracks")
    img, label = isbi_dataset[1]
    print(label.shape)

    plt.imshow(np.transpose(label, (1, 2, 0)), interpolation='nearest', cmap='gray')
    plt.show()

    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=8,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
