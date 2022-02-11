---
title: Model Training Notes
author: Dustin Wilson
date: February 8, 2022
---

This post is included along with my [main post](./trained-a-gan.html) to serve as a user guide for those interested in training their own GANs with the code presented [here](https://github.com/DMW2151/msls-pytorch-dcgan).

## Infrastructure

### Setting Up The Full Infrastructure

**NOTICE:** &mdash; <mark>This module deploys live resources into your AWS account, please be mindful of the costs associated with those resources. This module does not download (nor provision pipelines to download) MSLS data, but you may access it from [Mapillary](https://www.mapillary.com/datasets).</mark>

All infrastructure for the project is available [here](https://github.com/DMW2151/msls-dcgan-infra) and should be readily accessible if you're familiar with Terraform. If you're not comfortable with Terraform, I would recommend reading the following resources to make sure you're **safely** deploying resources to your account.

- [Terraform on AWS](https://learn.hashicorp.com/collections/terraform/aws-get-started)

If you are familiar with Terraform, I'd ask that you still be quite cautious, as this module deploys resources that can easily top a few hundred dollars / day if left unattended (mostly the `DL1` instance, but SageMaker and EBS can add up, too). Keep the following in mind:

- **Remember to change the state-file's S3-backend to a bucket you own!**
- **Remember to change regions to one with `DL1` instances offered!**

### Model Training Only

If you're just interested in training a GAN (and not the intermediate training or hardware metrics), I would recommend spinning up a single `DL1` instance, cloning the model [repo](https://github.com/DMW2151/msls-pytorch-dcgan), and pip installing the training package with `pip install -e /path/to/msls/package/model/`. From there, you should be ready to train on any data local to that instance.

**NOTE:** &mdash; If you're using a modern GPU or HPU, I would suggest training on the best disk available to your instance. On the `DL1`, I saw a meaningful performance improvement from using ephemeral NVME storage over the GP3 volume.

-------

## Data Preparation

The MSLS data is available for download [here](https://www.mapillary.com/dataset/places). Please note that since at least December they've had [sporadic performance](https://github.com/mapillary/mapillary_sls/issues/23) on their download site. The data is available in batches, and at the very least Batch 6 (1.6 GB compressed) should download without issue. The data should **not** be accessed programmatically, and once you agree to the Mapillary TOS you'll receive a signed URL for each batch. I used the following to download and unzip the data to local storage.

```bash
    #! /bin/bash
curl -k -XGET ${A_SIGNED_DOWNLOAD_URL} --output /data/msls_${BATCH_NUM}.zip &&\
    sudo unzip -qq /data/msls_${BATCH_NUM}.zip -d /data/imgs
```

-------

## Model Training

I designed model training to be as simple as possible. On a GPU instance, the following should be adequate to begin training. All flags for `python3 -m msls.run_dcgan` are shown below. Please see the CLI  `--help` to understand what each flag and value does in the example below.

```bash

python3 -m pip install --upgrade pip &&\
    sudo -H pip3 install /path/to/msls/package/model

# In short, the below does a 16 epoch run on a model with id: `global-dcgan-128-1`
# using data at `/data/imgs/train_val/`, saving checkpoints to `/efs/trained_model/`
# and uploading an artifact to s3://dmw2151-habana-model-outputs
python3 -m msls.run_dcgan \
    -c '{"name": "global-dcgan-128-1", "root": "/efs/trained_model/", "log_frequency": 50, "save_frequency": 1}' \
    -t '{"nc": 3, "nz": 128, "ngf": 128, "ndf": 32, "lr": 0.0002, "beta1": 0.5, "beta2": 0.999, "batch_size": 256, "img_size": 128, "weight_decay": 0.05}'\
    --s_epoch 0 \
    --n_epoch 16 \
    --dataroot /data/imgs/train_val/ \
    --logging True \
    --profile True  \
    --s3_bucket 'dmw2151-habana-model-outputs'
```

In the case above, the `msls` package will fall back to training on the GPU because no HPU is available. However, when a call to `load_habana_module()` succeeds, the model will prefer to train on the HPU. For example, rather than running `python3 -m msls.run_dcgan` directly on the host, I can spin up a Habana PyTorch container, mount the proper volumes, and then run the same training script as above.

```bash

# From the host, launch a PyTorch Container w. the proper Volumes...
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

# Run the same command(s), but in the container
root@xyz45645abzy:/\# python3 -m msls.run_dcgan ....
```

Finally, If you want a truly guided experience, you can use the notebooks in `./model/notebooks` to train in a SageMaker notebook. As of writing, `DL1` is not a supported SageMaker instance type, so you'll be restricted to GPU instances.
