---
title: Model Training Notes
author: Dustin Wilson
date: February 8, 2022
---

This section is included along with my [main post](./trained-a-gan.html) to serve as a user guide for those interested in training their own GANs with the code presented [here](https://github.com/DMW2151/msls-pytorch-dcgan).

## Infrastructure

### Setting Up The Full Infrastructure

**NOTICE:** &mdash; This module deploys live resources into your AWS account, please be mindful of the costs associated with those resources. This module does not download (nor provision pipelines to download) MSLS data, but you may access it from [Mapillary](https://www.mapillary.com/datasets).

All infrastructure for the project is available [here](https://github.com/DMW2151/msls-dcgan-infra) and should be readily accessible if you're familiar with Terraform. If you're not comfortable with Terraform, I would recommend reading Hashicorp's intro to [Terraform on AWS](https://learn.hashicorp.com/collections/terraform/aws-get-started) to make sure you're **safely** deploying resources to your account.

If you are familiar with Terraform, I'd ask that you still be quite cautious. This module deploys resources that can easily top a few hundred dollars / day if left unattended (mostly the `DL1` instance, but SageMaker and EBS can add up, too). Keep the following in mind:

Remember to:

- Change the Terraform statefile's S3-backend to a bucket you own!
- Change regions and AZs to launch the training instance to one with `DL1` instances available!
- Remove the API module, I strongly doubt you'll want or need it.

### Model Training Infra Only

If you're just interested in training a GAN, I would recommend spinning up a single `DL1` instance, cloning the model [repo](https://github.com/DMW2151/msls-pytorch-dcgan), and pip installing the training package with `pip install -e /path/to/msls/package/model/`. From there, you should be ready to train on any data local to that instance.

**NOTE:** &mdash; If you're using a modern GPU or HPU, I would suggest training on the best disk available to your instance. On the `DL1`, I saw a meaningful performance improvement from using ephemeral NVME storage over the GP3 volume.

-------

## Data Preparation

The MSLS data is available for download [here](https://www.mapillary.com/dataset/places). Please note that since December they've had [sporadic performance](https://github.com/mapillary/mapillary_sls/issues/23) on their download site. The data is available in batches, and at the very least Batch 6 (1.6 GB compressed) should download without issue. You should not attempt to access the data programmatically. Once you agree to the Mapillary TOS you'll receive a signed URL for each batch. I used the following to download and unzip each batch to instance storage.

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
    pip3 install /root/msls-pytorch-dcgan/model/ # Possibly with `sudo -H`

# In short, the below does a 16 epoch run on a model with:
#
# Id: `global-dcgan-128-1`
# Image data data from: `/data/imgs/train_val/`
# Saving checkpoints to: `/efs/trained_model/`
# Uploading an artifact to: `s3://dmw2151-habana-model-outputs`
#
python3 -m msls.run_dcgan \
    -c '{"name": "global-dcgan-128-1", "root": "/efs/trained_model/", "log_frequency": 50, "save_frequency": 1}' \
    -t '{"nc": 3, "nz": 128, "ngf": 128, "ndf": 32, "lr": 0.0002, "beta1": 0.5, "beta2": 0.999, "batch_size": 256, "img_size": 128, "weight_decay": 0.05}'\
    --s_epoch 0 \
    --n_epoch 16 \
    --dataroot /data/imgs/train_val/ \
    --logging True \
    --profile True  \
    --s3_bucket 'dmw2151-habana-model-outputs'

# BUG: Mysteriously -> This works while the above fails in the container; build the MSLS package without
# any references to habana, then just run habana dcgan as a standalone py file. oof.
pip3 install boto3

cd /root/msls-pytorch-dcgan/model/msls &&\
    python3 run_dcgan.py \
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
    -v /ebs:/data \
    -v /efs:/efs \
    -v /home/ubuntu/msls-pytorch-dcgan:/root/msls-pytorch-dcgan/ \
    --ipc=host \
    vault.habana.ai/gaudi-docker/1.2.0/ubuntu18.04/habanalabs/pytorch-installer-1.10.0:1.2.0-585

# Run the same command(s), but in the container
root@xyzxyzxyz:/\# python3 -m msls.run_dcgan ....
```

Finally, If you want a truly guided experience, you can use the notebooks in `./model/notebooks` to train in a SageMaker notebook. As of writing, `DL1` is not a supported SageMaker instance type, so you'll be restricted to GPU instances.

## Inference & New Image Generation

In my opinion, this is the cool part. Individual images and Gifs can be generated from a CPU instance by restoring a trained model and passing any noise vector through the Generator.

Static images are produced quite easily with the following, where `G` is a pre-trained model:

```python
    imgs = vutils.make_grid(
    G(Z).detach().to(DEVICE), padding=4, normalize=True, nrow=4
).cpu()

tmp_img_hash = uuid.uuid4().__str__()
vutils.save_image(imgs, f"/tmp/{tmp_img_hash}.png")
```

Generating GIFs is a bit more complicated. The GIFs are created with a method called `SLERP`. spherical linear interpolation (`SLERP`) involves taking two key-frames (e.g. `Z` vectors) and creating smooth intermediate locations (input `Z`) between them. Rather than re-implementing `SLERP`, you can just use the following to create and save a new GIF.

```python
    from msls.dcgan_utils import (
    gen_img_sequence_array
)

tmp_gif_location = gen_img_sequence_array(
    MODEL_CFG, G, n_frames=10, Z_size=TRAIN_CFG.nz
)
```
