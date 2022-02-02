# USE:

# Command line wrapper around gaudi_dcgan.py
# Sample Usage on Command Line - See Notes on Running on DL1
#
# python3 run_gaudi_dcgan.py --dataroot "/data/imgs/" --name msls_test_001 --s_epoch 0 --n_epoch 16

# General Deps
import random
import os
import sys

import numpy as np
import argparse

# Torch Deps
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

# DCGAN
import gaudi_dcgan as dcgan
import msls_dcgan_utils as dcgan_utils


parser = argparse.ArgumentParser(description="Run MSLS DCGAN")

parser.add_argument(
    "-n",
    "--name",
    type=str,
    help="Name to save model with; see /${model_dir}/${name} for model artifacts and traces",
)

parser.add_argument(
    "-d",
    "--dataroot",
    type=str,
    help="Root folder of training data; recursively selects all *.jpg, *.png, *.tiff, *.ppm files.",
    default="/data/imgs/test/",
)

parser.add_argument(
    "-se",
    "--s_epoch",
    type=int,
    help="Epoch to resume training from - requires a prior checkpoint",
    default=0,
)

parser.add_argument(
    "-p",
    "--profile",
    type=bool,
    help="Run the Torch profiler/save traces during training",
    default=False,
)

parser.add_argument(
    "-ne", "--n_epoch", type=int, help="Train model through N epochs", default=16
)

parser.add_argument(
    "-md",
    "--model_dir",
    type=str,
    help="Root folder to save model artifacts and traces",
    default="/efs/trained_model/",
)

parser.add_argument(
    "-pf",
    "--progress_freq",
    type=int,
    help="Save progress images every N batches",
    default=1000,
)

parser.add_argument(
    "-lf",
    "--logging_freq",
    type=int,
    help="Print loss metrics to STDOUT every N batches",
    default=50,
)

parser.add_argument(
    "-cf",
    "--checkpoint_freq",
    type=int,
    help="Save a model checkpoint to disk every N epochs",
    default=1,
)

parser.add_argument(
    "-b",
    "--batch",
    type=int,
    help="Batch size...",
    default=512,
)


if __name__ == "__main__":

    args = parser.parse_args()

    # Model Config Args
    MODEL_NAME = args.name
    MODEL_DIR = args.model_dir or "/efs/trained_model/"
    PROGRESS_FREQ = args.progress_freq
    LOG_FREQ = args.logging_freq
    CHECKPOINT_FREQ = args.checkpoint_freq

    # Training Args
    NUM_EPOCHS = args.n_epoch or 16
    START_EPOCH = args.s_epoch or 0
    DATAROOT = args.dataroot or "/data/imgs/train_val"
    PROFILE = args.profile or False

    IMG_SIZE = 64
    BATCH_SIZE = args.batch or 512

    DEVICE = "cuda" if (torch.cuda.is_available()) else "cpu"
    if dcgan.HABANA_ENABLED:
        DEVICE = "hpu"

    # Init Model Config w. Default DCGAN Values; Disallowing any custom values here
    # because the original DCGAN is a bit unstable when outside of the 64x64
    # img world!
    model_cfg = dcgan.ModelCheckpointConfig(
        model_name=MODEL_NAME,
        model_dir=MODEL_DIR,
        log_frequency=LOG_FREQ,
        save_frequency=CHECKPOINT_FREQ,
        gen_progress_frequency=PROGRESS_FREQ,
    )

    train_cfg = dcgan.TrainingConfig(
        batch_size=BATCH_SIZE,  # At Recommendation of Pytorch Profiler
        dev=torch.device(DEVICE),
    )

    # We can use an image folder dataset; depending on the size of the training directory this can take a
    # little to instantiate; about 3 min for 40GB (also depends on EBS...)
    if False:
        dataset = dset.ImageFolder(
            root=DATAROOT,
            transform=transforms.Compose(
                [
                    transforms.RandomAffine(degrees=0, translate=(0.3, 0.0)),
                    transforms.CenterCrop(IMG_SIZE * 4),
                    transforms.Resize(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

    # NOTE: TODO: Get Better Performance on this dataset!!
    dataset = dcgan_utils.MSLSImageDataset(
        root=DATAROOT,
        recursive=True,
        transforms=transforms.Compose(
            [
                transforms.RandomAffine(degrees=0, translate=(0.3, 0.0)),
                transforms.CenterCrop(IMG_SIZE * 4),
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
            pin_memory=True,
            shuffle=False,
            prefetch_factor=2,
            persistent_workers=True,
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
