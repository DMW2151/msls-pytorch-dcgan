# General
import datetime
import os

# DCGAN
from gan import Discriminator, Generator
import dcgan_utils as utils
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from habana_frameworks.torch.utils.library_loader import load_habana_module

load_habana_module()
from habana_dataloader import HabanaDataLoader
from habana_frameworks.torch.core import mark_step
from habana_frameworks.torch.hpex.optimizers import FusedAdamW
from torch.utils.tensorboard import SummaryWriter

# Import Torch Habana && Init Values
HABANA_ENABLED = 1
HABANA_LAZY = 1
HPU_WORLD_SIZE = 8

os.environ["PT_HPU_LAZY_MODE"] = "1"
os.environ["GRAPH_VISUALIZATION"] = "True"


def get_msls_dataloader(rank, train_cfg, params=utils.DEFAULT_LOADER_PARAMS):

    # We can use an image folder dataset; depending on the size of the training
    # directory this can take a little to instantiate; about 3 min for 40GB
    #
    # Unclear why these specific parameters are needed for acceleration,
    # unclear if the `transforms.RandomAffine` ruins it.
    # See: https://docs.habana.ai/en/v1.1.0/PyTorch_User_Guide/PyTorch_User_Guide.html

    # transforms.RandomAffine(degrees=0, translate=(0.3, 0.0)), ## NOT SUPPORTED w, AEON
    # GaussianNoise(0.0, 0.05),

    dataset = torchvision.datasets.ImageFolder(
        root=train_cfg.data_root,
        transform=transforms.Compose(
            [
                transforms.CenterCrop(train_cfg.img_size * 4),
                transforms.Resize(train_cfg.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )

    msls_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=HPU_WORLD_SIZE, rank=rank, shuffle=False
    )

    params["dataset"] = dataset
    params["sampler"] = msls_sampler

    return HabanaDataLoader(**params)
