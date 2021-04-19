# -*- encoding: utf-8 -*-
'''
Date: 2021-04-14 16:16:37
LastEditors: Jervint
LastEditTime: 2021-04-14 17:12:11
Description: 
FilePath: /kubeflow-pytorch-test/model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ConvBlock(1, 16, 3, 2, 1)
        self.conv2 = ConvBlock(16, 32, 3, 1, 1)
        self.conv3 = ConvBlock(32, 64, 1, 1, 0)
        self.max_pool2d = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool2d(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)