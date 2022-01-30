# Core Script for running DCGAN -> Implements the GAN architecture from the Original
# DCGAN Paper in PyTorch with slight adjustments to the optimizer.
#
# See: https://arxiv.org/abs/1406.2661

import datetime
import os
import random
import re
from dataclasses import dataclass

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

if (torch.__version__ > "1.8"):
    import torch.profiler
    
# Habana Imports - will fail if not on a Habana DL AMI instance
# Set Habana configuration or otherwise disable Habana...
try:
    from habana_frameworks.torch.utils.library_loader import load_habana_module
    from habana_frameworks.torch.hpex.optimizers import FusedAdamW
    import habana_frameworks.torch.core as htcore
    load_habana_module()

    HABANA_ENABLED = 1

    # Use lazy mode if Habana is available; allows SynapseAI graph compiler to
    # optimize the device execution for these ops Set internal `HABANA_LAZY`
    # and environment var `PT_HPU_LAZY_MODE`

    HABANA_LAZY = 1
    os.environ["PT_HPU_LAZY_MODE"] = "1"

    # Explicit Habana logging parameters, see: ~/.habana_logs/. and ./.graph_dumps
    # for SynapseAI
    os.environ["GRAPH_VISUALIZATION"] = True

except ImportError:
    # Failed Habana Import -> HABANA_ENABLED == 0
    HABANA_ENABLED = 0

USE_AMP = True

@dataclass
class TrainingConfig:
    """
    TrainingConfig holds the training parameters for both the generator
    and discriminator networks.

    --------
    Example - Increase learning rate from default (0.0002) -> (0.0004) and
    batch size from default (128) -> (512).

    train_cfg = dcgan.TrainingConfig(
        lr=0.0004, batch_size=512
    )
    """

    batch_size: int = 128  # Batch size during training -> DCGAN: 128
    img_size: int = 64  # Spatial size of training images -> DCGAN: 64
    nc: int = 3  # Number of channels in the training image -> DCGAN: 3
    # Size of Z vector (i.e. size of generator input) -> DCGAN: 100
    nz: int = 100
    ngf: int = 64  # Size of feature maps in generator -> DCGAN: 64
    ndf: int = 64  # Size of feature maps in discriminator -> DCGAN: 64
    lr: float = 0.0002  # Learning rate for optimizers
    beta1: float = 0.5  # Beta1 hyperparam for Adam optimizers
    beta2: float = 0.999  # Beta2 hyperparam for Adam optimizers

    dev: torch.device = torch.device(
        "cuda:0" if (torch.cuda.is_available()) else "hpu")
    ngpu: int = int(torch.cuda.is_available())  # No Support for Multi GPU!!

    def _announce(self):
        """Show Pytorch and CUDA attributes before Training"""

        print("====================")
        print(self.dev.__repr__())
        print(f"Pytorch Version: {torch.__version__}")
    
        if torch.cuda.device_count():
            print(f"Running with {torch.cuda.device_count()} GPUs Available.")
            print(f"CUDA Available: {torch.cuda.is_available()}")

            try:
                print(torch._C._cuda_getDriverVersion(), "cuda driver")
                print(
                    torch._C._cuda_getCompiledVersion(),
                    "cuda compiled version")
                print(torch._C._nccl_version(), "nccl")
                for i in range(torch.cuda.device_count()):
                    print(
                        "device %s:" %
                        i, torch.cuda.get_device_properties(i))
            except AttributeError:
                pass

        print("====================")

    def get_net_D(self):
        """
        Instantiate a Disctiminator Network:

        -   Note on Adam vs AdamW: Uses AdamW over Adam. In general, DCGAN and Adam are both
            susceptible to over-fitting on early samples. AdamW adds a weight decay parameter
            (default=0.01) on the optimizer for each model step

        -   Note on FusedAdamW vs AdamW: AdamW loops over parameters and launches kernels for
            each parameter when running the optimizer. This is CPU bound and can be a bottleneck
            on performance. FusedAdamW can batch the elementwise updates applied to all the
            model’s parameters into one or a few kernel launches.

        See: Fixing Weight Decay Regularization in Adam - https://arxiv.org/abs/1711.05101

        The Habana FusedAdamW optimizer uses a custom Habana implementation of `apex.optimizers.FusedAdam`,
        on Habana machines, enable this, otherwise use regular `AdamW`.
        """

        # Instantiate Discriminator Net
        netD = Discriminator(self)

        # Enable Data Parallelism across all available GPUs...
        # BUG: TODO: This disables the ability to run in Single GPU - In
        # practice, this is not ideal!
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                netD = nn.DataParallel(netD)

        # Put model on device(s)
        netD.to(self.dev)

        # Will fail if not on a Habana DL AMI Instance; See Note on FusedAdamW
        if HABANA_ENABLED:
            optimD = FusedAdamW(
                netD.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                weight_decay=0.01,
            )
            return netD, optimD

        # If not on Habana, then use Adam optimizer...
        optimD = optim.AdamW(
            netD.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=0.01,
        )
        return netD, optimD

    def get_net_G(self):
        """
        Instantiate a Generator Network - See notes on `get_net_D` re specific
        optimizer choices.
        """

        # Instantiate Generator Net
        netG = Generator(self)

        # Enable Data Parallelism across all available GPUs...
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                netG = nn.DataParallel(netG)

        # Put model on device(s)
        netG.to(self.dev)

        # Will fail if not on a Habana DL AMI Instance; See Note on FusedAdamW
        if HABANA_ENABLED:
            optimG = FusedAdamW(
                netG.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=self.lr,
                betas=(self.beta1, self.beta2)
            )

            return netG, optimG

        # If not on Habana, then use Adam optimizer...
        optimG = optim.AdamW(
            netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=0.01
        )

        return netG, optimG


@dataclass
class ModelCheckpointConfig:
    """
    ModelCheckpointConfig holds the model save parameters for both the generator
    and discriminator networks.

    --------
    Example: Rename the model and decrease save frequency from every epoch (1) to
    every fourth epoch (4)

    model_cfg = dcgan.ModelCheckpointConfig(
        model_name=msls_dcgan_habana_001,
        save_frequency=4
    )
    """

    model_name: str = "msls_dcgan_001"  # Name of the Model
    # Directory to save the model checkpoints to...
    model_dir: str = "/efs/trained_model" # Requires `/efs/trained_model` has permissions s.t. ec2-user/ubuntu can write.
    save_frequency: int = 1  # Save a model checkpoint every N epochs
    log_frequency: int = 50  # Print logs to STDOUT every N batches
    gen_progress_frequency: int = 250  # Save progress images every N batches


def weights_init(m):
    """
    Custom weights initialization called on netG and netD, uses the hardcoded values
    from the DCGAN paper (mean, std) => (0, 0.02)
    """

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    """
    Generator Net. The generator is designed to map the latent space vector (z) to believable data.
    Since our data are images, this means transforming (by default) a [1 x 100] latent vector to a
    3 x 64 x 64 RGB image.

    Applies 4 x (Strided 2DConv, BatchNorm, ReLu) layers, and then a TanH layer to transform the
    output data to (-1, 1) for each channel (color)...
    """

    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.ngpu = cfg.ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(cfg.nz, cfg.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(cfg.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(cfg.ngf * 8, cfg.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(cfg.ngf * 4, cfg.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(cfg.ngf * 2, cfg.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(cfg.ngf, cfg.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """
    Discriminator Net. Discriminator is a classifier network that (by default) takes a
    3 x 64 x 64 image as input and outputs a probability that the image is from the set
    of real images.

    Applies 1 x (Strided 2DConv, ReLu) + 3 x (Strided 2DConv, BatchNorm, ReLu) layers,
    and then a Sigmoid layer to transform the output data to (0, 1). No different from
    Logit ;)
    """

    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.ngpu = cfg.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(cfg.nc, cfg.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(cfg.ndf, cfg.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(cfg.ndf * 2, cfg.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(cfg.ndf * 4, cfg.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(cfg.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


def instantiate_from_checkpoint(netD, netG, optimD, optimG, path):
    """
    Utility function to restart training from a given checkpoint file
    --------
    Args:
        netD, netG - nn.Module - The Generator and Discriminator networks

        optimD, optimG - Union(torch.optim, torch.hpex.optimizers.FusedAdamW) - Optimizer
        function for Discriminator and Generator Nets

        path - str Path to file to open...
    --------
    Example:
    cur_epoch, losses, fixed_noise, img_list = instantiate_from_checkpoint(
            netD, netG, optimD, optimG, path
    )

    TODO: Probably not the most efficient use of memory here, could so
    something clever w. (de)serialization, but IMO, this is OK for now...
    """

    checkpoint = torch.load(path)

    # Seed Discriminator
    netD.load_state_dict(checkpoint["D_state_dict"])
    optimD.load_state_dict(checkpoint["D_optim"])

    # Seed Generator
    netG.load_state_dict(checkpoint["G_state_dict"])
    optimG.load_state_dict(checkpoint["G_optim"])

    return (
        checkpoint["epoch"],
        checkpoint["losses"],
        checkpoint["noise"],
        checkpoint["img_list"],
    )


def generate_fake_samples(n_samples, train_cfg, model_cfg, as_of_epoch=16):
    """
    Generates samples from a model checkpoint saved to disk, writes a few sample grids to disk
    and also returns last to the user
    --------
    Args:
        - n_samples - int - Number of samples to generate
        - train_cfg - TrainingConfig - Used to initialize the Generator model
        - model_cfg - ModelCheckPointConfig - Defines how to fetch the model checkpoint from disk
        - as_of_epoch - int - Epoch to generate samples as of - will fail if no model
            checkpoint is available
    --------
    Example: Plot 16 sample images from the 16th epoch of training

    plt.figure(figsize=(15, 15))

    imgs = dcgan.generate_fake_samples(
        n_samples=16,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        as_of_epoch=16
    )

    plt.imshow(
        np.transpose(
            vutils.make_grid(imgs.to(train_cfg.dev), padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
    """

    # Generate Noise - Latent Vector for the Model...
    rd_noise = torch.randn(
        n_samples, train_cfg.nz, 1, 1,
        device=train_cfg.dev
    )

    # Initialize empty models && initialize from `as_of_epoch`
    netD, optimD = train_cfg.get_net_D()
    netG, optimG = train_cfg.get_net_G()

    path = f"{model_cfg.model_dir}/{model_cfg.model_name}/checkpoint_{as_of_epoch}.pt"

    _, _, _, _ = instantiate_from_checkpoint(netD, netG, optimD, optimG, path)

    # Use the Generator to create "believable" fake images - You can call a plotting function
    # on this output to visualize the images vs real ones

    # Ideally a Generator Net can use a CPU to (slowly) generate samples, this confirms it, 
    # we can run the net through via CPU for "inference"
    generated_imgs = netG(rd_noise).detach().cpu()
    return generated_imgs


def start_or_resume_training_run(
        dl, train_cfg, model_cfg, n_epochs=256, st_epoch=0, profile_run=False):
    """
    Begin Training Model. That's It.
    --------
    Args:
        - dl - pytorch.dataloader
        - train_cfg - TrainingConfig - Used to initialize the Generator model
        - model_cfg - ModelCheckPointConfig - Defines how to fetch the model checkpoint from disk
        - n_epochs - intt - Number of Epochs to train through...
        - as_of_epoch - int - Epoch to generate samples as of - will fail if no model
        checkpoint is available

    --------
    Example: Start a training run from `START_EPOCH` and go until `NUM_EPOCHS` using the
    parameters given in `train_cfg` and `model_cfg`

    result = dcgan.start_or_resume_training_run(
        dataloader, train_cfg, model_cfg, num_epochs=NUM_EPOCHS, start_epoch=START_EPOCH
    )
    """
    # Announce
    train_cfg._announce()

    # Initialize Net and Optimizers
    netD, optimD = train_cfg.get_net_D()
    netG, optimG = train_cfg.get_net_G()

    # Check the save-path for a model with this name && Load Params
    if st_epoch:
        path = f"{model_cfg.model_dir}/{model_cfg.model_name}/checkpoint_{st_epoch}.pt"

        cur_epoch, losses, fixed_noise, img_list = instantiate_from_checkpoint(
            netD, netG, optimD, optimG, path
        )

    # If no start epoch specified; then apply weights from DCGAN paper and proceed
    # w. model training...
    else:
        netG.apply(weights_init)
        netD.apply(weights_init)
        cur_epoch = 0
        img_list = []
        losses = {"_G": [], "_D": [], "D_x": [], "D_G_z1": [], "D_G_z2": []}
        fixed_noise = torch.randn(64, train_cfg.nz, 1, 1, device=train_cfg.dev)

    # Initialize PyTorch Writer
    writer = SummaryWriter(
        f"{model_cfg.model_dir}/{model_cfg.model_name}/events"
    )

    # Initialize Stateless BCELoss Function
    criterion = nn.BCELoss()

    # Init Profiler
    if (profile_run) and (torch.__version__ > "1.8"):
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{model_cfg.model_dir}/{model_cfg.model_name}/events"),
            record_shapes=True,
            with_stack=True
        )

        prof.start()

    # Start new training epochs...
    for epoch in range(cur_epoch, n_epochs + 1):

        # Set Epoch Logging Iteration to 0 - For Plotting!
        log_i = 0

        for epoch_step, dbatch in enumerate(dl, 0):
            
            # Generate batch of latent vectors
            Z = torch.randn(b_size, train_cfg.nz, 1,
                                1, device=train_cfg.dev)

            # (1.1) Update D network: All-real batch; log(D(x)) + log(1 - D(G(z)))
            # Discriminator loss calculated as the sum of losses for the all
            # real and all fake batches
            netD.zero_grad()

            real_cpu = dbatch[0].to(train_cfg.dev)
            b_size = real_cpu.size(0)
            label = torch.full(
                (b_size,), 1.0, dtype=torch.float, device=train_cfg.dev)

            # Forward pass real batch && Calculate D_loss
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # (1.2) Update D Network; Train with All-fake batch
            fake = netG(Z)
            label.fill_(0.0)

            # Classify all fake batch with D && Calculate D_loss
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)

            # Calculate the gradients for this batch, accumulated with previous gradients &&\
            # Compute error of D as sum over the fake and the real batches
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            # NOTE: This assumes we're using a custom Habana optimizer, in which case we need
            # to call `htcore.mark_step()` twice per Net per training step: See
            # comments above!

            # Mark Habana Steps => Discriminator Optim;
            if HABANA_ENABLED and HABANA_LAZY:
                htcore.mark_step()

            optimD.step()

            if HABANA_ENABLED and HABANA_LAZY:
                htcore.mark_step()

            # (2) Update Net_G: maximize log(D(G(z)))

            netG.zero_grad()
            label.fill_(1.0)  # fake labels are real for generator cost

            # Forward pass fake batch through Net_D; Calculate G_loss
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                output = netD(fake).view(-1)
                errG = criterion(output, label)

            # Calculate gradients for Net_G
            errG.backward()
            D_G_z2 = output.mean().item()

            # Mark Habana Steps => Generator Optim
            if HABANA_ENABLED and HABANA_LAZY:
                htcore.mark_step()

            # Update G
            optimG.step()

            if (profile_run) and (torch.__version__ > "1.8"):
                prof.step()

            if HABANA_ENABLED and HABANA_LAZY:
                htcore.mark_step()

            # With default params - this shows loss every 6,400 images (50 training steps * 128/step)
            if (epoch_step % model_cfg.log_frequency) == 0:
                print(
                    f" [{datetime.datetime.utcnow().__str__()}] [{epoch}/{n_epochs}][{epoch_step}/{len(dl)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                )

                # Write Metrics to TensorBoard...
                for metric, val in zip(
                    ['G_loss', 'D_loss', 'D_x', 'D_G_z1', 'D_G_z2'],
                    [errG.item(), errD.item(), D_x, D_G_z1, D_G_z2]
                ):
                    writer.add_scalar(metric, val, (epoch * len(dl.dataset)) + (log_i * model_cfg.log_frequency))

                log_i += 1
                writer.flush()

            # Save Sample Imgs Every N Epochs...
            if (epoch_step % model_cfg.gen_progress_frequency) == 0:
                # And also save the progress on the fixed latent input
                # vector...
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    img_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True)
                    )

            # Save Losses (and a few other function values) for plotting later
            losses["_G"].append(errG.item())
            losses["_D"].append(errD.item())
            losses["D_x"].append(D_x)
            losses["D_G_z1"].append(D_G_z1)
            losses["D_G_z2"].append(D_G_z2)

        # Save Model && Progress Images Every N Epochs
        if (epoch % model_cfg.save_frequency == 0) | (epoch == n_epochs):

            # Ensure the Save Directory Exists
            if not os.path.exists(
                    f"{model_cfg.model_dir}/{model_cfg.model_name}"):
                os.makedirs(f"{model_cfg.model_dir}/{model_cfg.model_name}")

            # Save Checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "D_state_dict": netD.state_dict(),
                    "G_state_dict": netG.state_dict(),
                    "D_optim": optimD.state_dict(),
                    "G_optim": optimG.state_dict(),
                    "losses": losses,
                    "img_list": img_list,
                    "noise": fixed_noise,
                },
                f"{model_cfg.model_dir}/{model_cfg.model_name}/checkpoint_{epoch}.pt",
            )

    writer.close()

    if (profile_run) and (torch.__version__ > "1.8"):
        prof.stop()

    return {"losses": losses, "img_list": img_list}
