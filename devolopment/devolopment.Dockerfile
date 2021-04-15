FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN pip install tensorboardX==1.6.0 torchvision==0.9.0
RUN mkdir -p /workspace
ADD . /workspace
WORKDIR /workspace

RUN chgrp -R 0 /workspace && chmod -R g+rwX /workspace