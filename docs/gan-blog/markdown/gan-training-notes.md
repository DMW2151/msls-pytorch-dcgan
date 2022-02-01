---
title: Model Training Notes - Generative Street Level Imagery on DL1 Instances
author: Dustin Wilson
date: January 29, 2022
---

This post is included along with my [main post](./trained-a-gan.html) to serve as a user guide for those interested in training their own GANs with the code presented here

## Data Preparation

```bash
curl -k -XGET ${A_SIGNED_DOWNLOAD_URL} --output /data/msls_${BATCH_NUM}.zip &&\
    sudo unzip -qq /data/msls_${BATCH_NUM}.zip -d /data/imgs
```

## Model Training

```bash
# Run Model
source activate pytorch_p38

# Test on A smaller sample first!
python3 /home/ubuntu/msls-pytorch-dcgan/model/run_gaudi_dcgan.py \
    --dataroot "/data/imgs/" \
    --name msls_test_001 \
    --s_epoch 0 \
    --n_epoch 16
```

```bash
python3 -m cProfile -o profile_name.prof \
    /home/ubuntu/msls-pytorch-dcgan/model/run_gaudi_dcgan.py \
    --dataroot "/data/imgs/train_val/london/database/" \
    --name msls_test_001 \
    --s_epoch 0 \
    --n_epoch 2
```
