<!--
 * @Date: 2021-04-15 18:16:36
 * @LastEditors: Jervint
 * @LastEditTime: 2021-04-15 18:47:30
 * @Description: 
 * @FilePath: /kubeflow-pytorch-test/readme.md
-->

### Kubeflow-Pipelines使用记录
#### 1. Train
有两种方式实现pipeline，分别为`自定义k8s-schema的yaml配置文件`和`使用kfp（Kubeflow Pipelines SDK）生成yaml配置文件`
> 推荐使用kfp，使用简单   `pip install kfp`

* 第一步：制作基础镜像`devolopment.Dockerfile`
```Dockerfile
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN pip install tensorboardX==1.6.0 torchvision==0.9.0
RUN mkdir -p /workspace
ADD . /workspace # or "git clone ${REPO} /workspace"
WORKDIR /workspace

RUN chgrp -R 0 /workspace && chmod -R g+rwX /workspace
```
* 第二步：生成数据集`preprocess.py`
```python
# -*- encoding: utf-8 -*-
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
```
* 第三步：定义网络结构`model.py`
```
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
```
* 第四步：训练脚本`train.py`
```python
# -*- encoding: utf-8 -*-
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

    train_loader_path = os.path.join(args.dataset_root_dir, args.dataset_name, "train.pickle")
    test_loader_path = os.path.join(args.dataset_root_dir, args.dataset_name, "test.pickle")
    train(train_loader_path, test_loader_path, args.epochs, args.lr, args.momentum, args.seed)
```
* 第五步：定义pipeline`devolop_pipeline.py`
```python
# -*- encoding: utf-8 -*-
import kfp.dsl as dsl
from kfp.dsl import PipelineVolume

# 定义preprocess_op
def preprocess_op(docker_image_path: str, dataset_root_dir: str,
                  dataset_name: str):
    def volume_op():
        return dsl.VolumeOp(name="create pipeline volume",
                            resource_name="pipeline-pvc",
                            modes=["ReadWriteOnce"],
                            size="3Gi")

    pvolume = volume_op()
    return dsl.ContainerOp(
        name="preprocessing",
        image=docker_image_path,
        command=["python", f"preprocess.py"],
        arguments=[
            "--dataset_root_dir", dataset_root_dir, "--dataset_name",
            dataset_name
        ],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume.volume})

# 定义train_op
def train_op(docker_image_path: str, pvolume: PipelineVolume,
             dataset_root_dir: str, dataset_name: str):
    return dsl.ContainerOp(
        name="train",
        image=docker_image_path,
        command=["python", f"train.py"],
        arguments=[
            "--dataset_root_dir", dataset_root_dir, "--dataset_name",
            dataset_name
        ],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume})

# 定义pipeline
@dsl.pipeline(
    name='Fashion MNIST Training Pytorch Pipeline',
    description=
    'Fashion MNIST Training Pipeline to be executed on KubeFlow-Pytorch.')
def train_pipeline(
        image: str = "harbor.qunhequnhe.com/koolab/kubeflow-mnist:train",
        dataset_root_dir: str = "/workspace/datasets",
        dataset_name: str = "fashion-mnist"):

    preprocess_data = preprocess_op(image,
                                    dataset_root_dir=dataset_root_dir,
                                    dataset_name=dataset_name)
    train_and_eval = train_op(image,
                              pvolume=preprocess_data.pvolume,
                              dataset_root_dir=dataset_root_dir,
                              dataset_name=dataset_name)


if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(training_pipeline, __file__ + '.tar.gz')
```
* 第六步：生成配置文件`train_pipeline.tar.gz`
```bash
dsl-compile --py devolop_pipeline.py --output train_pipeline.tar.gz
```

* 第七步：上传`train_pipeline.tar.gz`创建pipeline，并创建Experiment运行（通过graph-node查看log）
    * web-ui
        1. upload pipeline
        2. create experiment
        3. create run
    * 命令行：`tar -zxf train_pipeline.tar.gz && kubectl apply -f train_pipeline.yaml`

