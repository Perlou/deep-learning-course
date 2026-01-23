import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_dim=784):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # 第一层：升维
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            # 第二层
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            # 第三层
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            # 输出层：映射到图像维度
            nn.Linear(1024, img_dim),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # 第一层：降维
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            # 第二层
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            # 输出层
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 输出概率
        )

    def forward(self, img):
        return self.model(img)


def basic_implementation():
    latent_dim = 100


def main():
    basic_implementation()
    # advanced_examples()
    # training_tips()
    # exercises()


if __name__ == "__main__":
    main()
