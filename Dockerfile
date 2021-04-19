FROM continuumio/miniconda3
RUN pip install --no-cache torch==1.8.0 tensorboardX==1.6.0 torchvision==0.9.0 kubernetes kfp kfserving -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com/pypi/simple/ 


