FROM continuumio/miniconda3
RUN pip install --no-cache torch==1.8.0 tensorboardX==1.6.0 torchvision==0.9.0 kubernetes kfp kfserving -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com/pypi/simple/ 
RUN mkdir -p /workspace
WORKDIR /workspace
RUN chgrp -R 0 /workspace && chmod -R g+rwX /workspace

ADD deployment /workspace/deployment