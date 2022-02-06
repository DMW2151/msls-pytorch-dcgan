from dataclasses import dataclass
from typing import Tuple, Dict

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from gan import Generator, Discriminator

DEFAULT_LOADER_PARAMS = {
    "shuffle": False,
    "num_workers": min(8, os.cpu_count() // 2),
    "pin_memory": True,
    "timeout": 0,
    "prefetch_factor": 2,
    "persistent_workers": False,
    "batch_size": 256
}

DEFAULT_TORCH_PROFILER_SCHEDULE = {
    "wait": 2,
    "warmup": 2,
    "active": 1,
    "repeat": 1,
}


@dataclass
class TrainingConfig:
    """
    TrainingConfig holds the training parameters for both the generator
    and discriminator networks.

    args:
        dev: torch.device
        batch_size: int = 256  # Batch size during training -> DCGAN: 128
        img_size: int = 64  # Spatial size of training images -> DCGAN: 64
        nc: int = 3  # Number of channels in the training image -> DCGAN: 3
        nz: int = 256  # NOTE: TODO: LARGER LATENT INPUT SPACE....
        ngf: int = 64  # Size of feature maps in generator -> DCGAN: 64
        ndf: int = 64  # Size of feature maps in discriminator -> DCGAN: 64
        lr: float = 0.0002  # Learning rate for optimizers
        beta1: float = 0.5  # Beta1 hyperparam for Adam optimizers
        beta2: float = 0.999  # Beta2 hyperparam for Adam optimizers
        ngpu: int = int(torch.cuda.device_count())  # No Support for Multi GPU!!
        data_root: str = "/data/images/train_val"
    """

    dev: torch.device
    batch_size: int = 256
    img_size: int = 64
    nc: int = 3
    nz: int = 256
    ngf: int = 64
    ndf: int = 64
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.05
    ngpu: int = int(torch.cuda.device_count())
    data_root: str = "/data/images/train_val"

    def _announce(self):
        """Show PyTorch, HPU, and CUDA attributes before Training"""

        print("====================")
        print(self.dev.__repr__())
        print(f"Pytorch Version: {torch.__version__}")

        if torch.cuda.device_count():
            print(f"Running with {torch.cuda.device_count()} GPUs Available.")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            for i in range(torch.cuda.device_count()):
                print("device %s:" % i, torch.cuda.get_device_properties(i))
        print("====================")

    def get_network(
        self, network: nn.Module, world_size: int = 1, device_rank: int = 0
    ):
        """Instantiate a Disctiminator Network:"""

        # Put model on device(s)
        N = network(self).to(self.dev)

        # If we have multiple devices - Enable DDP
        # TODO: Would be nice to check Num. HPUs here....
        DEVICE_COUNT = max(torch.cuda.device_count(), world_size) > 0

        if (torch.cuda.is_available()) and (DEVICE_COUNT):
            N = nn.parallel.DistributedDataParallel(N, device_ids=[device_rank])

        if self.dev == torch.device("hpu"):
            # Patch in the hpex.optimizer; FusedAdamW allows for better kernel
            # vs. regular AdamW, SGD, etc...
            from habana_frameworks.torch.hpex.optimizers import FusedAdamW

            optimizer = FusedAdamW(
                N.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                weight_decay=self.weight_decay,
            )
            return N, optimizer

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
    ModelCheckpointConfig holds the model parameters for writing the model
    artifact + checkpoint data to disk...

    # Save a model checkpoint every N epochs
    # Print logs to STDOUT every N batches
    # Save progress images every N batches
    """

    model_name: str = "msls_dcgan_001"
    model_dir: str = "/efs/trained_model"
    save_frequency: int = 1
    log_frequency: int = 50
    gen_progress_frequency: int = 1000

    def get_msls_profiler(self, schedule=DEFAULT_TORCH_PROFILER_SCHEDULE):
        """Returns a standard PyTorch.profiler"""

        return torch.profiler.profile(
            schedule=torch.profiler.schedule(**schedule),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"{self.model_dir}/{self.model_name}/events"
            ),
            record_shapes=True,
            with_stack=False,
            profile_memory=True,
        )

    def get_msls_writer(self):
        return SummaryWriter(f"{self.model_dir}/{self.model_name}/events")

    def make_all_paths(self):
        trace_path = f"{self.model_dir}/{self.model_name}/events"
        if not os.path.exists(trace_path):
            os.makedirs(trace_path)

    def checkpoint_path(self, checkpoint: int) -> str:
        expected_path = (
            f"{self.model_dir}/{self.model_name}/checkpoint_{checkpoint}.pt"
        )
        self.make_all_paths()
        return expected_path


def weights_init(m: nn.Module):
    """
    Custom weights initialization called on net_G and net_D, uses the
    hardcoded values from the DCGAN paper (mean, std) => (0, 0.02)
    """

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def generate_fake_samples(
    n_samples: int,
    train_cfg: TrainingConfig,
    model_cfg: ModelCheckpointConfig,
    epoch: int = 16,
    cpu: bool = True,
) -> torch.Tensor:
    """
    Generates samples from a model checkpoint saved to disk, writes a few
    sample grids to disk and also returns last to the user

    Use the Generator to create "believable" fake images - You can call a
    plotting function on this output to visualize the images vs real ones

    Ideally a Generator Net can use a CPU to (slowly) generate samples,
    this confirms it, we can run the net through via CPU for "inference"
    --------
    Args:
        - n_samples - int - Number of samples to generate
        - train_cfg - TrainingConfig - Used to initialize the Generator model
        - model_cfg - ModelCheckPointConfig - Defines how to fetch the model
            checkpoint from disk
        - as_of_epoch - int - Epoch to generate samples as of - will fail
            if no model checkpoint is available
    """

    # Generate Noise - Latent Vector for the Model...
    Z = torch.randn(n_samples, train_cfg.nz, 1, 1, device=train_cfg.dev)

    # Initialize empty models && initialize from `as_of_epoch`
    # Device rank constant because we do inference from a single machine...
    D, opt_D = train_cfg.get_network(Discriminator, device_rank=0)
    G, opt_G = train_cfg.get_network(Generator, device_rank=0)

    _, _, _, _ = restore_model(
        G,
        D,
        opt_G,
        opt_D,
        cpu,
        path=f"{model_cfg.model_dir}/{model_cfg.model_name}/checkpoint_{epoch}.pt",
    )

    generated_imgs = G(Z).detach().cpu()
    return generated_imgs


def restore_model(
    G: nn.Module,
    D: nn.Module,
    opt_G: torch.optim.AdamW,
    opt_D: torch.optim.AdamW,
    cpu: bool,
    path: str,
) -> Tuple[int, Dict[str, float], torch.Tensor, torch.Tensor]:
    """
    Utility function to restart training from a given checkpoint file
    --------
    Args:
        net_D, net_G - nn.Module - The Generator and Discriminator networks

        optim_D, optim_G - Union(torch.optim, torch.hpex.optimizers.FusedAdamW)
        Optimizer function for Discriminator and Generator Nets

        path - str Path to file to open...
    --------
    Example:
    cur_epoch, losses, fixed_noise, img_list = restore_model(
            net_D, net_G, optim_D, optim_G, path
    )

    TODO: Probably not the most efficient use of memory here, could so
    something clever w. (de)serialization, but IMO, this is OK for now...
    """
    dev = "cuda" if (torch.cuda.is_available()) else "cpu"
    checkpoint = torch.load(path, map_location=torch.device(dev))

    # Seed Discriminator, Generator, and Optimizers...
    D.load_state_dict(checkpoint["D_state_dict"])
    G.load_state_dict(checkpoint["G_state_dict"])
    opt_D.load_state_dict(checkpoint["D_optim"])
    opt_G.load_state_dict(checkpoint["G_optim"])

    return (
        checkpoint["epoch"],
        checkpoint["losses"],
        checkpoint["noise"],
        checkpoint["img_list"],
    )
