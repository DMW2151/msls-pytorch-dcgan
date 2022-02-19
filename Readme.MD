# MSLS-DCGAN

## About

This repository contains code for training an implementation of Ian Goodfellow's original `DCGAN` model<sup>1</sup> on data sourced from Mapillary's Street Level Sequences dataset. Although this paper is quite old, I felt that it's worth taking a look at because it's a foundational paper in the field and can be implemented in just a few hundred lines of PyTorch. The code in this repo has been instrumented to run on either Gaudi accelerated or Nvidia GPU instances.

I would encourage you to refer to my full write-up [here](https://gan.dmw2151.com/trained-a-gan.html), which contains details on:

- Theory Behind DCGAN
- Discussion of MSLS Dataset & Data Preparation
- Discussion of AWS Architecture
- Modifications for Training on Gaudi (`DL1`) and Nvidia GPU (`P`)
- DCGAN Results
- Comparative Results w.r.t Cost and Hardware Performance (`DL1` v. `P`)

Please see the Habana HPU repo for this model [here](https://github.com/DMW2151/Model-References)


## Usage

```bash
python3 -m pip install --upgrade pip &&\
    sudo -H pip3 install /home/ubuntu/msls-pytorch-dcgan/model
```

```bash
# In short, the below does a 16 epoch run on a model with:
#
# Id: `global-dcgan-128-1`
# Image data data from: `/data/imgs/train_val/`
# Saving checkpoints to: `/efs/trained_model/`
# Uploading an artifact to: `s3://dmw2151-habana-model-outputs`
#
python3 -m msls.run_dcgan \
    -c '{"name": "msls-gpu-dcgan-128-001", "root": "/efs/trained_model/", "log_frequency": 50, "save_frequency": 1}' \
    -t '{"nc": 3, "nz": 128, "ngf": 128, "ndf": 32, "lr": 0.0002, "beta1": 0.5, "beta2": 0.999, "batch_size": 256, "img_size": 128, "weight_decay": 0.05}'\
    --s_epoch 0 \
    --n_epoch 24 \
    --dataroot /data/imgs/train_val/ \
    --logging True \
    --profile True  \
    --s3_bucket 'dmw2151-habana-model-outputs'
```

## Citations

**[1]** *"Generative Adversarial Networks." Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. ArXiv 2014.*
