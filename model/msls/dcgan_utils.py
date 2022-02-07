"""
Common utilities to allow for training on both HPU and GPU instances
"""

from dataclasses import dataclass
import collections

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union
from torch.utils.tensorboard import SummaryWriter
from gan import (
    Discriminator128,
    Generator128,
    Discriminator64,
    Generator64,
)

DEFAULT_LOADER_PARAMS = {
    "shuffle": False,
    "num_workers": min(8, os.cpu_count() // 2),
    "pin_memory": True,
    "timeout": 0,
    "prefetch_factor": 2,
    "persistent_workers": False,
    "batch_size": 256,
}

# NOTE: in `start_or_resume_training_run` we're assuming there are at least
# as many batches as (wait + (warmup + active)) * repeat; else may fail to
# profile.
DEFAULT_TORCH_PROFILER_SCHEDULE = {
    "wait": 2,
    "warmup": 2,
    "active": 2,
    "repeat": 1,
}


@dataclass
class TrainingConfig:
    """
    TrainingConfig holds the training parameters for both the generator
    and discriminator networks. This is configurable during training, the
    original values from the DCGAN paper are listed below in parens.

    Args:
    -------
        dev: torch.device: The device to initialize the model on (N/A)

        batch_size: int: Batch size during training (128)

        img_size: int: Width and Height of training images (64)

        nc: int: Number of channels in the training images; e.g. use
            3 for colored images, 1 for grayscale (3)

        nz: int: The size (1, ${NZ}) of the latent noize vector the model
            accepts as input (100)

        ngf: int: Size of feature maps in Generator (64)

        ndf: int: Size of feature maps in Discriminator (64)

        lr: float: Learning rate for both Generator and Discriminator
            optimizers (0.0002)

        beta1: float: Beta1 for Adam Optimizer, common to both the Generator
            and Discriminator network (0.5)

        beta2: float: Beta2 for Adam Optimizer, common to both the Generator
            and Discriminator network (0.999)

        weight_decay: float: Weight decay on Adam Optimizer, common to both the
            Generator and Discriminator networks. (0.0)

        ngpu: int: Number of GPUs available (N/A)

        data_root: str: Root of the training images, torch.dataloader created
            from this config will recursively crawl this root for images (N/A)
    """

    dev: torch.device
    batch_size: int = 256
    img_size: int = 128
    nc: int = 3
    nz: int = 256
    ngf: int = 256
    ndf: int = 64
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.05
    ngpu: int = int(torch.cuda.device_count())
    data_root: str = "/data/images/train_val"

    def _announce(self) -> None:
        """Display PyTorch, HPU, and CUDA attributes before Training"""

        print("====================")
        print(self.dev.__repr__())
        print(f"Pytorch Version: {torch.__version__}")

        if torch.cuda.device_count():
            print(f"Running with {torch.cuda.device_count()} GPUs Available.")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            for i in range(torch.cuda.device_count()):
                print("device %s:" % i, torch.cuda.get_device_properties(i))

        if True:
            print("Habana Statistics ...")

        print("====================")

    def get_network(
        self,
        network: Union[Discriminator64, Discriminator128, Generator64, Generator128],
        world_size: int = 1,
        device_rank: int = 0,
    ) -> (
        Union[Discriminator64, Discriminator128, Generator64, Generator128],
        optim.AdamW,
    ):
        """
        Instantiate an instance of a DCGAN network with it's associated optimizer.

        Args:
        --------
            network: nn.module: Specifically, one of the four network classes defined
                in this module

            world_size: int: We use DDP by default, even on single nodes, specifies a
                training world size

            device_rank: int: See note above, int to uniquely identify this GPU/HPU given
                a world size of `world_size`

        NOTE:/TODO:
        --------
            - Return type hints aren't technically correct: — A check for the
            union type, `Union[optim.AdamW, FusedAdamW]` would fail on instances that do not have the
            correct Habana drivers, leave it out...
        """

        # Put model on device(s)
        N = network(self).to(self.dev)

        # If we have multiple devices - Enable DDP; Assumes the world_size is known
        # if we're training on HPU, else use cuda.device_count() and treat it as a
        # multi-gpu,
        DEVICE_COUNT = max(torch.cuda.device_count(), world_size) > 0

        if (torch.cuda.is_available()) and (DEVICE_COUNT):
            N = nn.parallel.DistributedDataParallel(N, device_ids=[device_rank])

        # Patch in the hpex.optimizer; FusedAdamW allows for better kernel
        # launching on HPU units vs. regular Torch AdamW, SGD, etc...
        if self.dev.type == "hpu":
            from habana_frameworks.torch.hpex.optimizers import FusedAdamW

            optimizer = FusedAdamW(
                N.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                weight_decay=self.weight_decay,
            )
            return N, optimizer

        # If on GPU, fallback to AdamW
        optimizer = optim.AdamW(
            N.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )

        return N, optimizer


@dataclass
class ModelCheckpointConfig:
    """
    ModelCheckpointConfig is a dataclass which holds the model parameters for writing
    the model artifact + checkpoint data to disk

    Args:
    --------
        - name: str: Name which uniquely identifies a model-run, can be re-used, but
            will overwrite existing data

        - root: str: Directory to save model artifacts, training results, images,
            videos, etc.

        - save_frequency: int: During training, save the model w. progress estimations
            every `save_frequency` epochs

        - log_frequency: int: During training, log the results to STDOUT every
            `log_frequency` batches
    """

    name: str = "msls-dcgan-128"
    root: str = "/efs/trained_model"
    save_frequency: int = 1
    log_frequency: int = 50

    def get_msls_profiler(
        self, schedule=DEFAULT_TORCH_PROFILER_SCHEDULE
    ) -> torch.profiler.profile:
        """Returns a standard PyTorch.profiler"""

        return torch.profiler.profile(
            schedule=torch.profiler.schedule(**schedule),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"{self.root}/{self.name}/events"
            ),
            record_shapes=True,
            with_stack=False,
            profile_memory=True,
        )

    def get_msls_writer(self):
        """Returns a PyTorch SummaryWriter -> Write Traces to TensorBoard"""
        return SummaryWriter(f"{self.root}/{self.name}/events")

    def make_all_paths(self) -> None:
        """Create all subdirectories for model artifacts + Results"""

        paths = [
            f"{self.root}/{self.name}/events",
            f"{self.root}/{self.name}/figures",
            f"{self.root}/{self.name}/videos",
        ]

        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def checkpoint_path(self, checkpoint: int) -> str:
        expected_path = f"{self.root}/{self.name}/checkpoint_{checkpoint}.pt"
        self.make_all_paths()
        return expected_path

    def create_slim_checkpoint(self, checkpoint: int):
        """
        Create a `slim` checkpoint. The regular checkpoints contain information about
        training progress, both networks, and their optimizers.

        A slim checkpoint ONLY is concerned with the status of the **Generator** s.t.
        it can be saved/opened/transferred more quickly while still retaining it's use

        Args:
        --------
            - checkpoint: int: n-th checkpoint from a given self.model_name to load
                and save as slim...
        """

        # Load the full checkpoint
        checkpoint = torch.load(
            self.checkpoint_path(checkpoint), map_location=torch.device("cpu")
        )

        # Remove all keys except the most critical...
        for key, _ in checkpoint.items():
            if key not in ("epoch", "G_state_dict"):
                del checkpoint[key]

        # Save the model back to disk, expect a size reduction of ~90%
        torch.save(
            checkpoint,
            f"{self.root}/{self.name}/slim_checkpoint_{checkpoint}.pt",
        )

    def slim_checkpoint_to_cloud_storage(self, bucket: str, checkpoint: int):
        """Uploads a slim checkpoint to S3"""

        # Open S3 connedtion
        s3_client = boto3.client("s3")

        # Saves over the slim checkpoint if exists, else creates...
        self.create_slim_checkpoint(checkpoint)

        # Send the slim checkpoint up to S3!
        response = s3_client.upload_file(
            f"{self.root}/{self.name}/checkpoint_{checkpoint}.pt",
            bucket,
            f"{self.name}/slim_checkpoint_{checkpoint}.pt",
        )


class LimitDataset(torch.utils.data.Dataset):
    """
    Wrapper around torch.utils.data.Dataset to limit the of data-points passed to a
    DataLoader; not particularly efficient, but does it's job...
    """

    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        """Clobber the old Length"""
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]


def weights_init(m: nn.Module):
    """
    Custom weights initialization function called on Generator and Discriminator
    networks, uses the hardcoded values from the DCGAN paper (mean, std) => (0, 0.02)

    Args:
    --------
        m: nn.Module: a layer of a network...
    """

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_checkpoint(path: str, cpu: bool) -> dict:
    dev = "cuda" if (torch.cuda.is_available()) else "cpu"
    return torch.load(path, map_location=torch.device(dev))


def restore_G_for_inference(checkpoint: dict, G: nn.Module):
    gsd = checkpoint["G_state_dict"]
    mutated_gsd = collections.OrderedDict()

    for k, v in gsd.items():
        if "module." in k:
            inference_k = k.replace("module.", "")
            mutated_gsd[inference_k] = v
        else:
            mutated_gsd[k] = v

    G.load_state_dict(mutated_gsd)


def restore_model(
    checkpoint: dict,
    G: nn.Module,
    D: nn.Module,
    opt_G: torch.optim.AdamW,
    opt_D: torch.optim.AdamW,
):
    """
    Utility function to restart training from a given checkpoint file
    --------
    Args:
        G, D: nn.Module: The Generator and Discriminator networks to restore,
            (normally) without weight initialized...

        opt_D, opt_G - Union(torch.optim, torch.hpex.optimizers.FusedAdamW)
        Optimizer function for Discriminator and Generator Nets
    """

    # Seed Discriminator, Generator, and Optimizers...
    D.load_state_dict(checkpoint["D_state_dict"])
    G.load_state_dict(checkpoint["G_state_dict"])
    opt_D.load_state_dict(checkpoint["D_optim"])
    opt_G.load_state_dict(checkpoint["G_optim"])
