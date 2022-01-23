# General Deps
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from IPython.display import HTML

# Torch Deps
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# DCGAN
import gaudi_dcgan as dcgan

# Root directory for dataset
dataroot = "/efs/images/sample"
MODEL_SEED = 215

if __name__ == "__main__":

    # Seed Model
    random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)

    # Init Model Config w. Default DCGAN Values; Disallowing any custom values here
    # because the original DCGAN is a bit unstable when outside of the 64x64 img world!
    model_cfg = dcgan.ModelCheckpointConfig()
    train_cfg = dcgan.TrainingConfig()

    # We can use an image folder dataset the way we have it setup.
    dataset = dset.ImageFolder(
        root=dataroot,
        transform=transforms.Compose(
            [
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
            batch_sampler=None,
            num_workers=8,
            collate_fn=None,
            pin_memory=True,
            timeout=0,
            worker_init_fn=None,
            multiprocessing_context=None,
            generator=None,
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

    # Run Model
    result = dcgan.start_or_resume_training_run(
        dataloader, train_cfg, model_cfg, num_epochs=16, start_epoch=0
    )
