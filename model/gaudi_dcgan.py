# Core Script for running DCGAN -> Implements the GAN architecture from the Original
# DCGAN Paper in Pytorch with slight adjustments to the optimizer.
#
# https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
# https://arxiv.org/abs/1406.2661

# A Very Quick Intro To GAN + Notation used...
# Let:
#   - X be an vector (image) with dims C x W x H
#   - Z be a latent vector sampled from a standard normal distribution
#   - D(X) be a discriminator network which outputs the probability that X came from training data
#   - G(Z) be a generator network which maps the latent vector, Z, to data-space.
#   - D(G(z)) is the probability that the output of the generator G is a real image.
#
#  All that's going on is that D tries to maximize the probability it correctly classifies
#  reals and fakes (from G)


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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Habana Imports - Will Fail if not on a Habana DL AMI Instance
try:
    # Set Habana Configuration - Lazy vs Eager Mode - In Lazy mode, execution is triggered
    # wherever data is read back to the host from the Habana device. The placement of
    # mark_step() is required at the following points in a training script:
    #
    #   - Right after optimizer.step() to cleanly demarcate training iterations,
    #   - Between loss.backward and optimizer.step() if the optimizer being used is a Habana custom optimizer.
    from habana_frameworks.torch.utils.library_loader import load_habana_module

    load_habana_module()
    from habana_frameworks.torch.hpex.optimizers import FusedAdamW
    import habana_frameworks.torch.core as htcore

    HABANA_ENABLED = 1
    HABANA_LAZY = 1
    os.environ["PT_HPU_LAZY_MODE"] = "1"

except ImportError:
    # Failed Habana Import -> HABANA_ENABLED == 0
    HABANA_ENABLED = 0


@dataclass
class TrainingConfig:

    batch_size: int = 128  # Batch size during training -> DCGAN: 128
    img_size: int = 64  # Spatial size of training images -> DCGAN: 64
    nc: int = 3  #  Number of channels in the training image -> DCGAN: 3
    nz: int = 100  # Size of Z vector (i.e. size of generator input) -> DCGAN: 100
    ngf: int = 64  # Size of feature maps in generator -> DCGAN: 64
    ndf: int = 64  # Size of feature maps in discriminator -> DCGAN: 64

    lr: float = 0.0002  # Learning rate for optimizers
    beta1: float = 0.5  # Beta1 hyperparam for Adam optimizers
    beta2: float = 0.999  # Beta2 hyperparam for Adam optimizers

    dev: torch.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "hpu")
    ngpu: int = int(torch.cuda.is_available())  # No Support for Multi GPU!!

    def _announce(self):
        print("====================")
        print(self.dev.__repr__())
        print(f"Pytorch Version: {torch.__version__}")
        print("====================")

    def get_net_D(self):
        """Instantiate an uninitialized Disctiminator Network"""

        netD = Discriminator(self).to(self.dev)

        # Will Fail if not on a Habana DL AMI Instance; Use FusedAdamW over the Original
        # Adam Optimizer.
        if HABANA_ENABLED:
            # NOTE: Use AdamW optimizer instead of vanilla Adam; also allows use of the AdamW
            # Habana Optimizer on DL1 instances:
            # See: Fixing Weight Decay Regularization in Adam - https://arxiv.org/abs/1711.05101
            #
            # AFAIK: Adam loops over parameters and launches kernels for each parameter when
            # running the optimizer. This is CPU bound and can be a bottleneck on performance...

            optimD = FusedAdamW(
                netD.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=self.lr,
                betas=(self.beta1, self.beta2),
            )

            return netD, optimD

        # Not on Habana...Use Regular Adam Optimizer...
        optimD = optim.AdamW(
            netD.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )

        return netD, optimD

    def get_net_G(self):
        """Instantiate an uninitialized Generator Network"""

        netG = Generator(self).to(self.dev)

        # Will Fail if not on a Habana DL AMI Instance; Use FusedAdamW over the Original
        # Adam Optimizer.
        if HABANA_ENABLED:
            optimG = FusedAdamW(
                netG.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=self.lr,
                betas=(self.beta1, self.beta2),
            )

            return netG, optimG

        # Not on Habana...Use Regular Adam Optimizer...
        optimG = optim.AdamW(
            netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )

        return netG, optimG


@dataclass
class ModelCheckpointConfig:
    model_name: str = "msls_dcgan_001"
    model_dir: str = "/efs/trained_model"
    save_frequency: int = 1


def weights_init(m):
    """Custom weights initialization called on netG and netD"""

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    """
    See: https://github.com/pytorch/examples/issues/70
    See: https://github.com/soumith/dcgan.torch/issues/2#issuecomment-164862299
    See: https://github.com/openai/improved-gan/blob/master/imagenet/discriminator.py
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
    Sample Usage
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

    fixed_noise = torch.randn(n_samples, train_cfg.nz, 1, 1, device=train_cfg.dev)

    netD, optimD = train_cfg.get_net_D()
    netG, optimG = train_cfg.get_net_G()

    path = f"{model_cfg.model_dir}/{model_cfg.model_name}/checkpoint_{as_of_epoch}.pt"

    _, _, _, _ = instantiate_from_checkpoint(netD, netG, optimD, optimG, path)

    # And then call: vutils.make_grid(fake, padding=2, normalize=True))
    generated_imgs = netG(fixed_noise).detach().cpu()
    return generated_imgs


def start_or_resume_training_run(dl, train_cfg, model_cfg, n_epochs=256, st_epoch=0):
    """Train Model """
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
        losses = {"_G": [], "_D": []}
        fixed_noise = torch.randn(64, train_cfg.nz, 1, 1, device=train_cfg.dev)

    # Initialize Stateless BCELoss Function
    criterion = nn.BCELoss()

    # Start new training epochs...
    for epoch in range(cur_epoch, n_epochs + 1):
        for epoch_step, dbatch in enumerate(dl, 0):

            # (1) All-real batch; Update D network: log(D(x)) + log(1 - D(G(z)))
            # Discriminator loss calculated as the sum of losses for the all real and all fake batches 
            netD.zero_grad()

            real_cpu = dbatch[0].to(train_cfg.dev)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1.0, dtype=torch.float, device=train_cfg.dev)

            # Forward pass real batch && Calculate D_loss
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with All-fake batch

            # Generate batch of latent vectors
            noise = torch.randn(b_size, train_cfg.nz, 1, 1, device=train_cfg.dev)

            fake = netG(noise)
            label.fill_(0.0)

            # Classify all fake batch with D && Calculate D_loss
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)

            # Calculate the gradients for this batch, accumulated with previous gradients &&\
            # Compute error of D as sum over the fake and the real batches
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            # NOTE: This assumes we're using a custom Habana optimizer, in which case we need
            # to call `htcore.mark_step()` twice per Net per training step: See comments above!

            # Mark Habana Steps => Discriminator Optim;
            if HABANA_ENABLED and HABANA_LAZY:
                htcore.mark_step()

            optimD.step()

            if HABANA_ENABLED and HABANA_LAZY:
                htcore.mark_step()

            ## (2) Update Net_G: maximize log(D(G(z)))

            netG.zero_grad()
            label.fill_(1.0)  # fake labels are real for generator cost

            # Forward pass fake batch through Net_D; Calculate G_loss

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

            if HABANA_ENABLED and HABANA_LAZY:
                htcore.mark_step()

            # Show Loss Every 32,000 images (250 training steps * 128/step)
            if epoch_step % 250 == 0:
                print(
                    f" [{datetime.datetime.utcnow().__str__()}] [{epoch}/{n_epochs}][{epoch_step}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                )

                # And also save the progress on the fixed latent input vector...
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            # Save Losses (and a few other function values) for plotting later
            losses["_G"].append(errG.item())
            losses["_D"].append(errD.item())
            losses["D_x"].append(D_x)s
            losses["D_G_z1"].append(D_G_z1)
            losses["D_G_z2"].append(D_G_z2)

        # Save Model && Progress Images Every N Epochs
        if (epoch % model_cfg.save_frequency == 0) | (epoch == n_epochs):

            # Ensure the Save Directory Exists
            if not os.path.exists(f"{model_cfg.model_dir}/{model_cfg.model_name}"):
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

    return {"losses": losses, "img_list": img_list}
