# -*- encoding: utf-8 -*-
'''
Date: 2021-04-14 11:10:22
LastEditors: Jervint
LastEditTime: 2021-04-14 14:11:09
Description: 
FilePath: /kubeflow-pytorch-test/mnist.py
'''
from __future__ import print_function

import argparse
import os

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
BATCH_SIZE = 64
EPOCHS = 10

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
WRITER = SummaryWriter("./logs")

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
KWARGS = {'num_workers': 1, 'pin_memory': True} if USE_CUDA else {}

TRAIN_LOADER = torch.utils.data.DataLoader(datasets.FashionMNIST(
    '../data',
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])),
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           **KWARGS)
TEST_LOADER = torch.utils.data.DataLoader(datasets.FashionMNIST(
    '../data',
    train=False,
    download=False,
    transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])),
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          **KWARGS)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


MODEL = Net().to(DEVICE)
if dist.is_available() and WORLD_SIZE > 1:
    dist.init_process_group(backend=dist.Backend.GLOO)
if dist.is_available() and dist.is_initialized():
    Distributor = nn.parallel.DistributedDataParallel if USE_CUDA else nn.parallel.DistributedDataParallelCPU
    MODEL = Distributor(MODEL)

OPTIMIZER = optim.SGD(MODEL.parameters(), lr=1e-2, momentum=0.5)


def train(epoch):
    MODEL.train()
    for batch_idx, (data, target) in enumerate(TRAIN_LOADER):
        data, target = data.to(DEVICE), target.to(DEVICE)
        OPTIMIZER.zero_grad()
        output = MODEL(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        OPTIMIZER.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
            epoch, batch_idx * len(data), len(TRAIN_LOADER.dataset),
            100. * batch_idx / len(TRAIN_LOADER), loss.item()))
        niter = epoch * len(TRAIN_LOADER) + batch_idx
        WRITER.add_scalar('loss', loss.item(), niter)


def test(epoch):
    MODEL.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in TEST_LOADER:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = MODEL(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(TEST_LOADER.dataset)
    print('\naccuracy={:.4f}\n'.format(
        float(correct) / len(TEST_LOADER.dataset)))
    WRITER.add_scalar('accuracy',
                      float(correct) / len(TEST_LOADER.dataset), epoch)


if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)

    torch.save(MODEL.state_dict(), "mnist_cnn.pt")
