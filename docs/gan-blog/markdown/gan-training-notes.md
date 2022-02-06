---
title: Model Training Notes
author: Dustin Wilson
date: January 29, 2022
---

This post is included along with my [main post](./trained-a-gan.html) to serve as a user guide for those interested in training their own GANs with the code presented [here](https://github.com/DMW2151/msls-pytorch-dcgan). 

## Infrastructure

**NOTICE:**

- This module deploys live resources into your AWS account, please be mindful of the costs associated with those resources.

- This module does not download (nor provision pipelines to download) MSLS data, you may access it from [Mapillary](https://www.mapillary.com/datasets).

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

### Training on a DL1 - Deep Learning Base / Habana Deep Learning AMI 

These instances have the habana drivers pre-installed, the only additional level of complication is running the correct docker container on top of the machine.

```bash
## Copy to good drive - 10 min?
cp -r /ebs /data

docker run -ti --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice \
    --net=host \
    -v /data:/data \
    -v /efs:/efs \
    -v /root/msls-pytorch-dcgan:/root/msls-pytorch-dcgan/ \
    --ipc=host \
    vault.habana.ai/gaudi-docker/1.2.0/ubuntu18.04/habanalabs/pytorch-installer-1.10.0:1.2.0-585

## Once Attached to Container Instance
pip3 install \
    tensorboard \
    torch_tb_profiler
    
 python3 /root/msls-pytorch-dcgan/model/run_gaudi_dcgan.py \
    --name dl24-w-noise \
    --data /data/ebs/imgs/train_val/phoenix \
    --batch 512 \
    --profile True \
    --logging True
```


## Model Training - GPU Instance


```bash
    source activate pytorch_p38

# Extra installs to drive tensorboard <-> Pytorch traces
pip3 install \
    tensorboard \
    torch_tb_profiler

# Train model using all images in `/msls/data/images/**` (or start with a smaller sample...)
 python3 ~/msls-pytorch-dcgan/model/run_dcgan.py \
    --name p3-w-noise \
    --data /data/imgs/ \
    --batch 512 \
    --profile True \
    --logging True
```

In general, If you're just interested in generating a GAN (and not the intermediate training or hardware metrics), cloning the model [repo](https://github.com/DMW2151/msls-pytorch-dcgan) onto a deep-learning AMI instance is the fastest way to get started training. To my knowledge, `DL1`, `P`, and `G` type instances have access to AWS' Deep Learning AMI.
