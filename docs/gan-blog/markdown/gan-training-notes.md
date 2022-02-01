---
title: Model Training Notes - Generative Street Level Imagery on DL1 Instances
author: Dustin Wilson
date: January 29, 2022
---

This post is included along with my [main post](./trained-a-gan.html) to serve as a user guide for those interested in training their own GANs with the code presented here. 

## Data Preparation

```bash

curl -k -XGET ${A_SIGNED_DOWNLOAD_URL} --output /data/msls_${BATCH_NUM}.zip &&\
    sudo unzip -qq /data/msls_${BATCH_NUM}.zip -d /data/imgs
```

## Model Training

I designed model training to be as simple as possible. Once on an instance running a Deep-Learning AMI, the following command kicks off the full model training cycle on a directory of images. I activated `pytorch_p38`, installed a few additional dependencies, and let my model train for a few hours.

```bash
    source activate pytorch_p38

# Train model using all images in `/msls/data/images/**`
python3 ~/msls-pytorch-dcgan/model/run_gaudi_dcgan.py \
    --dataroot /msls/data/images/ \
    --name msls_dl1_global \
    --s_epoch 0 \
    --n_epoch 16

```

In general, If you're just interested in generating a GAN (and not the intermediate training or hardware metrics), cloning the model [repo](https://github.com/DMW2151/msls-pytorch-dcgan) onto a deep-learning AMI instance is the fastest way to get started training. To my knowledge, `DL1`, `P`, and `G` type instances have access to AWS' Deep Learning AMI.




## Interpolated Images

