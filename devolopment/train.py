# -*- encoding: utf-8 -*-
'''
Date: 2021-04-14 16:16:26
LastEditors: Jervint
LastEditTime: 2021-04-14 17:45:37
Description: 
FilePath: /kubeflow-pytorch-test/train.py
'''

import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
from tensorboardX import SummaryWriter
# import torch.scheduler as sched
from model import Net


def train_epoch(model, train_loader, optimizer, epoch, device, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(data)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        niter = epoch * len(train_loader) + batch_idx
        writer.add_scalar('loss', loss.item(), niter)


def test_epoch(model, test_loader, epoch, device, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\naccuracy={:.4f}\n'.format(
        float(correct) / len(test_loader.dataset)))
    writer.add_scalar('accuracy',
                      float(correct) / len(test_loader.dataset), epoch)


def train(train_loader_path,
          test_loader_path,
          epochs,
          lr=1e-2,
          momentum=0.5,
          seed=1):
    writer = SummaryWriter("./logs/{}_{}".format(epochs, lr))
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(train_loader_path, 'rb') as f:
        train_dataloader = pickle.loads(f.read())
    with open(test_loader_path, 'rb') as f:
        test_dataloader = pickle.loads(f.read())

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_dataloader, optimizer, epoch, device, writer)
        test_epoch(model, test_dataloader, epoch, device, writer)

    torch.save(model.cpu().state_dict(), "mnist.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Kubeflow MNIST training-training PyTorch-script ')
    parser.add_argument('--dataset_root_dir',
                        help='dataset\'s root dir.',
                        default="./datasets",
                        type=str)
    parser.add_argument("--dataset_name",
                        help="dataset\'s name",
                        default="fashion-mnist",
                        type=str)
    parser.add_argument("--seed", help="dataset\'s name", default=1, type=int)
    parser.add_argument("--lr",
                        help="dataset\'s name",
                        default=1e-2,
                        type=float)
    parser.add_argument("--momentum",
                        help="dataset\'s name",
                        default=0.5,
                        type=float)
    parser.add_argument("--epochs",
                        help="dataset\'s name",
                        default=20,
                        type=int)
    args = parser.parse_args()

    train_loader_path = os.path.join(args.dataset_root_dir, args.dataset_name,
                                     "train.pickle")
    test_loader_path = os.path.join(args.dataset_root_dir, args.dataset_name,
                                    "test.pickle")
    train(train_loader_path, test_loader_path, args.epochs, args.lr,
          args.momentum, args.seed)
