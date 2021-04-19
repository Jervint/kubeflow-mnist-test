FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

# FROM continuumio/miniconda3
RUN pip install --no-cache torch==1.8.0 tensorboardX==1.6.0 torchvision==0.9.0 kubernetes kfp kfserving -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com/pypi/simple/ 
# RUN mkdir -p /workspace
# WORKDIR /workspace
# RUN chgrp -R 0 /workspace && chmod -R g+rwX /workspace

# ADD devolopment /workspace/devolopment

SHELL ["/bin/bash", "-c"]
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4