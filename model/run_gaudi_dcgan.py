# USE:

# Command line wrapper around gaudi_dcgan.py
# Sample Usage on Command Line - See Notes on Running on DL1
#
# python3 run_gaudi_dcgan.py --dataroot "/efs/samples/" --seed 215 --name msls_dl1_2022_01_24_001 --s_epoch 0 --n_epoch 16

# General Deps
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import argparse

# Torch Deps
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

# DCGAN
import gaudi_dcgan as dcgan


parser = argparse.ArgumentParser(description="Run MSLS DCGAN")

parser.add_argument("-s", "--seed", type=int, help="An integer to seed Pytorch")

parser.add_argument("-n", "--name", type=str, help="An integer to seed Pytorch")

parser.add_argument("-d", "--dataroot", type=str, help="Root folder of training data")

parser.add_argument("-se", "--s_epoch", type=int, help="Epoch to resume training from")

parser.add_argument("-p", "--profile", type=int, default=False, help="Use Torch profiler")

parser.add_argument(
    "-ne", "--n_epoch", type=int, help="Number of Epochs to train until"
)


if __name__ == "__main__":

    args = parser.parse_args()

    # Root directory for dataset
    DATAROOT = args.dataroot or "/efs/samples/"
    MODEL_SEED = args.seed or 215
    NUM_EPOCHS = args.n_epoch or 16
    START_EPOCH = args.s_epoch or 0
    NAME = args.name or "msls_dcgan_ml_dl_24xlarge_001"
    PROFILE = args.profile or False

    # Seed Model
    random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)

    # Init Model Config w. Default DCGAN Values; Disallowing any custom values here
    # because the original DCGAN is a bit unstable when outside of the 64x64
    # img world!
    model_cfg = dcgan.ModelCheckpointConfig(model_name=NAME)

    train_cfg = dcgan.TrainingConfig()

    # We can use an image folder dataset; depending on the size of the training directory this can take a
    # little to instantiate; about 5-8 min for 25GB (also depends on EFS burst)
    dataset = dset.ImageFolder(
        root=DATAROOT,
        transform=transforms.Compose(
            [
                transforms.RandomAffine(degrees=0, translate=(0.3, 0.0)),
                transforms.CenterCrop(
                    train_cfg.img_size * 4
                ),  # Use the Middle 256 x 256
                transforms.Resize(train_cfg.img_size),  # Downsize to 64 x 64
                transforms.ToTensor(),  # Cast from ND -> Tensor
                transforms.Normalize(  # Normalize Normalize with mean and standard deviation.
                    (
                        0.5,
                        0.5,
                        0.5,
                    ),
                    (
                        0.5,
                        0.5,
                        0.5,
                    ),
                ),
            ]
        ),
    )

    # Create the dataloader
    if dcgan.HABANA_ENABLED:

        # If using Habana -> Try to Use the Habana DataLoader w. the params given in
        # https://docs.habana.ai/en/v1.1.0/PyTorch_User_Guide/PyTorch_User_Guide.html#current-limitations
        import habana_dataloader

        dataloader = habana_dataloader.HabanaDataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            timeout=0,
            prefetch_factor=2,
            persistent_workers=False,
        )

    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count() // 2, 8),
        )

    # Create Location For Model Outputs
    if not os.path.exists(f"{model_cfg.model_dir}/{model_cfg.model_name}/events"):
        os.makedirs(f"{model_cfg.model_dir}/{model_cfg.model_name}/events")

    # Run Model
    result = dcgan.start_or_resume_training_run(
        dataloader,
        train_cfg,
        model_cfg,
        n_epochs=NUM_EPOCHS,
        st_epoch=START_EPOCH,
        profile_run=PROFILE,
    )
