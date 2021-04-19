# -*- encoding: utf-8 -*-
'''
Date: 2021-04-14 15:17:49
LastEditors: Jervint
LastEditTime: 2021-04-14 16:11:37
Description: 
FilePath: /kubeflow-pytorch-test/preprocess.py
'''
import os
import argparse
import pickle
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def preprocess(dataset_root_dir,
               train_batch_size,
               test_batch_size,
               num_workers,
               dataset_name="fashion-mnist"):
    dataset_path = os.path.join(dataset_root_dir, dataset_name)
    if not os.path.exists(dataset_path):
        os.system("mkdir -p {}".format(dataset_path))
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])
    train_dataset = datasets.FashionMNIST(dataset_path,
                                          train=True,
                                          download=True,
                                          transform=transform)
    test_dataset = datasets.FashionMNIST(dataset_path,
                                         train=False,
                                         download=True,
                                         transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)

    with open(os.path.join(dataset_path, 'train.pickle'), 'wb') as f:
        pickle.dump(train_loader, f)
    with open(os.path.join(dataset_path, 'test.pickle'), 'wb') as f:
        pickle.dump(test_loader, f)
    print("Write dataloader done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Kubeflow MNIST training-preprocessing PyTorch-script ')
    parser.add_argument('--dataset_root_dir',
                        help='dataset\'s root dir.',
                        default="./datasets",
                        type=str)
    parser.add_argument('--train_batch_size',
                        help='dataset\'s root dir.',
                        default=64,
                        type=int)
    parser.add_argument('--test_batch_size',
                        help='dataset\'s root dir.',
                        default=64,
                        type=int)
    parser.add_argument('--num_workers',
                        help='dataset\'s root dir.',
                        default=4,
                        type=int)
    parser.add_argument("--dataset_name",
                        help="dataset\'s name",
                        default="fashion-mnist",
                        type=str)
    args = parser.parse_args()

    preprocess(args.dataset_root_dir, args.train_batch_size,
               args.test_batch_size, args.num_workers, args.dataset_name)
