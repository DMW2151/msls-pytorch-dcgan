---
title: Model Training Notes
author: Dustin Wilson
date: January 29, 2022
---

This post is included along with my [main post](./trained-a-gan.html) to serve as a user guide for those interested in training their own GANs with the code presented [here](https://github.com/DMW2151/msls-pytorch-dcgan). 

## Infrastructure

```bash
```

## Data Preparation

```bash

# Please Note: https://github.com/mapillary/mapillary_sls/issues/23
curl -k -XGET ${A_SIGNED_DOWNLOAD_URL} --output /data/msls_${BATCH_NUM}.zip &&\
    sudo unzip -qq /data/msls_${BATCH_NUM}.zip -d /data/imgs
```

## Model Training

I designed model training to be as simple as possible. Once on an instance running a Deep-Learning AMI, the following command kicks off the full model training cycle on a directory of images. I activated `pytorch_p38`, installed a few additional dependencies, and let my model train for a few hours.

```bash
    source activate pytorch_p38

# Extra installs to drive tensorboard <-> Pytorch traces
pip3 install \
    tensorboard \
    torch_tb_profiler

# Train model using all images in `/msls/data/images/**` (or start with a smaller sample...)
 python3 ./model/run_gaudi_dcgan.py \
    --name p3_2xl \
    --data /data/imgs/ \
    --batch 1024 \
    --profile True \
    --logging True
```

In general, If you're just interested in generating a GAN (and not the intermediate training or hardware metrics), cloning the model [repo](https://github.com/DMW2151/msls-pytorch-dcgan) onto a deep-learning AMI instance is the fastest way to get started training. To my knowledge, `DL1`, `P`, and `G` type instances have access to AWS' Deep Learning AMI.

## Miscellaneous, Useful Commands

```bash
    # Quick Model Profiling
python -m torch.utils.bottleneck model/run_gaudi_dcgan.py  --dataroot /data/imgs/test/ --n_epochs 1
```

```bash
    # Try out SIMD Pillow...
pip uninstall pillow &&\
    CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```
