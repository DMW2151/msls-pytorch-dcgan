---
title: Model Training Notes
author: Dustin Wilson
date: January 29, 2022
---

This post is included along with my [main post](./trained-a-gan.html) to serve as a user guide for those interested in training their own GANs with the code presented [here](https://github.com/DMW2151/msls-pytorch-dcgan).

-------

## Infrastructure

### The Full Infrastructure

**NOTICE:** <mark> This module deploys live resources into your AWS account, please be mindful of the costs associated with those resources. This module does not download (nor provision pipelines to download) MSLS data, you may access it from [Mapillary](https://www.mapillary.com/datasets).</mark>

All infrastructure for the project is available [here](https://github.com/DMW2151/msls-dcgan-infra) and should be readily accessible if you're familiar with Terraform. If you're not comfortable with Terraform, I would recommend reading the following resources to make sure you're **safely** deploying resources to your account.

- [Terraform on AWS](https://learn.hashicorp.com/collections/terraform/aws-get-started)

If you are familiar with Terraform, I'd ask that you still be quite cautious, as this module deploys resources that can easily top a few hundred dollars / day if left unattended (mostly the `DL1` instance, but SageMaker and EBS can add up, too). Keep the following in mind:

- Remember to change the state-file's S3-backend to a bucket you own!
- Remember to change regions to one with `DL1` instances offered!

### Model Training Only

If you're just interested in training a GAN (and not the intermediate training or hardware metrics), I would recommend provisioning your own infrastructure, cloning the model [repo](https://github.com/DMW2151/msls-pytorch-dcgan) onto a suitable EC2 instance, and pip installing the training package with `pip -e install /path/to/msls/package/model/`.

### Note on Disk Performance

Because of the image size I'm using in the network, this problem is a bit inbalanced. On a high-performance GPU or the HPU, disk becomes a bottleneck rather quickly. The relative size of the training images (`3 x 64 x 64`) vs. the size of the images pulled off disk (~`3 x 480 x 480`) leads to constant stress on the disk. In this case, adding extra resources to train the model has a sharply diminished effect.

My Terraform provisions a GP3 volume w. 8000 IOPs to (partially) handle for this, but if you're using a modern GPU or HPU, I would suggest training on the best disk available to your instance. On the `DL1`, I saw a meaningful performance improvement from using ephemeral NVME storage over the GP3 volume.

-------

## Data Preparation

The MSLS data is available for download [here](https://www.mapillary.com/dataset/places). Please note that since at least December they've had [sporadic performance](https://github.com/mapillary/mapillary_sls/issues/23) on their download site. The data is available in batches, and at the very least Batch 6 (1.6 GB compressed) should download without issue.

The data should not be accessed programmatically, once you agree to the Mapillary TOS you'll recieve a signed URL for each batch. I used the following to download and unzip the data.

```bash
    #! /bin/bash
curl -k -XGET ${A_SIGNED_DOWNLOAD_URL} --output /data/msls_${BATCH_NUM}.zip &&\
    sudo unzip -qq /data/msls_${BATCH_NUM}.zip -d /data/imgs
```

-------

## Model Training

I designed model training to be as simple as possible. On a GPU instance, the following should be adequate to begin training. All flags for `python3 -m msls.run_dcgan` are shown below. Please refer to the full documentation to understand what each flag and value does.

```bash
    #! /bin/bash

python3 -m pip install --upgrade pip &&\
    sudo -H pip3 install /path/to/msls/package/model

python3 -m msls.run_dcgan \
    -c '{"name": "msls-dcgan-128", "root": "/efs/trained_model/", "log_frequency": 50, "save_frequency": 1,}' \
    -t '{"nc": 3, "nz": 256, "ngf": 256, "ndf": 64, "lr": 0.0002, "beta1": 0.5, "beta2": 0.999, "batch_size": 256, "img_size": 128, "weight_decay": 0.05}'\
    --s_epoch 0 \
    --n_epoch 16 \
    --dataroot /data/imgs/train_val/helsinki \
    --logging True \
    --profile True  \
    --s3_bucket 'dmw2151-habana-model-outputs'
```

In the case above, the `msls` package will fall back to training on the GPU because no HPU is available. When a call to `load_habana_module()` succeeds, the model will prefer to train on the HPU. Once in a container (or on host) with the Habana modules properly installed, the run same commands as above.

```bash
    #! /bin/bash
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
```

When using a container on a DLAMI with Habana drivers, remember to mount data and metadata folders with Docker's `-v` flag. Finally, If you want a truly guided experience, you can use the notebooks in `./model/notebooks` to train in a SageMaker notebook (as of writing, `DL1` is not a supported SageMaker instance type).
