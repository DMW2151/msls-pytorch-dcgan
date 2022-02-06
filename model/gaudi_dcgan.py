# Core script for running DCGAN -> Implements the DCGAN architecture from
# the original GAN/DCGAN papers in PyTorch with slight adjustments to the
# optimizer. Other changes discussed in post
#
# See: https://arxiv.org/abs/1406.2661

# General
import datetime
import os

# DCGAN
from gan import Discriminator, Generator
import dcgan.utils as utils
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

# Immport Torch Habana && Init Values
HABANA_ENABLED = 1
HABANA_LAZY = 1
WORLD_SIZE = 8


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
    # GaussianNoise(0.0, 0.1),

    dataset = torchvision.datasets.ImageFolder(
        root=train_cfg.data_root,
        transform=transforms.Compose(
            [
                transforms.CenterCrop(train_cfg.img_size * 4),
                transforms.Resize(train_cfg.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        ),
    )

    msls_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=8, rank=rank, shuffle=False
    )

    params["dataset"] = dataset
    params["sampler"] = msls_sampler

    return HabanaDataLoader(**params)


def start_or_resume_training_run(
    rank,
    train_cfg,
    model_cfg,
    n_epochs,
    st_epoch,
    enable_prof=True,
    enable_logging=True,
):
    """
    Begin Training Model. That's It.
    --------
    Args:
        - train_cfg - TrainingConfig - Used to initialize the Generator model
        - model_cfg - ModelCheckPointConfig - Defines how to fetch the model
                    checkpoint from disk
        - n_epochs - int - Number of Epochs to train through...
        - as_of_epoch - int - Epoch to generate samples as of - will fail if
            no model checkpoint is available

    --------
    Example: Start a training run from `START_EPOCH` and go until `NUM_EPOCHS`
    using the parameters given in `train_cfg` and `model_cfg`

    result = dcgan.start_or_resume_training_run(
        train_cfg, model_cfg, num_epochs=NUM_EPOCHS, start_epoch=START_EPOCH
    )
    """

    train_cfg.dev = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=torch.cuda.device_count(),
        rank=rank,
    )

    # Initialize Net and Optimizers
    net_D, optim_D = train_cfg.get_network(Discriminator, device_rank=rank)
    net_G, optim_G = train_cfg.get_network(Generator, device_rank=rank)

    # Check the save-path for a model with this name && Load Params
    if st_epoch:
        cur_epoch, losses, fixed_noise, img_list = utils.restore_model(
            net_G,
            net_D,
            optim_G,
            optim_D,
            f"{model_cfg.model_dir}/{model_cfg.model_name}/checkpoint_{st_epoch}.pt",
        )

    # If no start epoch specified; then start from 0 with the weights specified in the
    # DCGAN paper
    else:
        net_G.apply(utils.weights_init)
        net_D.apply(utils.weights_init)
        cur_epoch = 0
        img_list = []
        losses = {"_G": [], "_D": []}
        fixed_noise = torch.randn(64, train_cfg.nz, 1, 1, device=train_cfg.dev)

    # Initialize Stateless BCELoss Function
    torch.manual_seed(0)
    criterion = nn.BCEWithLogitsLoss().to(train_cfg.dev)
    scaler_D = torch.cuda.amp.GradScaler()
    scaler_G = torch.cuda.amp.GradScaler()

    dl = get_msls_dataloader(rank, train_cfg)

    if enable_prof:
        prof = utils.get_msls_profiler(model_cfg)

    if enable_logging:
        writer = utils.get_msls_writer(model_cfg)

    # Start new training epochs...
    for epoch in range(cur_epoch, n_epochs):

        if enable_prof:
            prof.start()

        if type(dl.sampler) == (
            torch.utils.data.distributed.DistributedSampler
        ):
            dl.sampler.set_epoch(epoch)

        for epoch_step, dbatch in enumerate(dl, 0):

            # (1.1) Update D network: All-real batch;
            # log(D(x)) + log(1 - D(G(z)))
            ###################################################################
            net_D.zero_grad()

            real_imgs = dbatch[0].to(train_cfg.dev)
            b_size = real_imgs.size(0)

            # Generate batch of latent vectors
            Z = torch.randn(b_size, train_cfg.nz, 1, 1, device=train_cfg.dev)
            label = torch.full((b_size,), 1.0, device=train_cfg.dev)

            # Forward pass real batch && Calculate D_loss
            with torch.cuda.amp.autocast():
                output = net_D(real_imgs.detach()).view(-1)
                err_D_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            scaler_D.scale(err_D_real).backward()
            D_X = torch.sigmoid(output).mean().item()

            # (1.2) Update D Network; Train with All-fake batch
            fake = net_G(Z)
            label.fill_(0.0)

            # Classify all fake batch with D && Calculate D_loss
            # Calculate the gradients for this batch, accumulated with
            # previous gradients &&\ Compute error of D as sum over the
            # fake and the real batches
            with torch.cuda.amp.autocast():
                output = net_D(fake.detach()).view(-1)
                err_D_fake = criterion(output, label)

            D_G_z1 = torch.sigmoid(output).mean().item()
            scaler_D.scale(err_D_fake).backward()
            err_D = err_D_real + err_D_fake

            # NOTE: This assumes we're using a custom Habana optimizer, in
            #  which case we need to call `htcore.mark_step()` twice per
            # Net per training step: See comments above! Mark Habana Steps
            # => Discriminator Optim...
            #
            # Update D - optim_D.step() is called in Scalar_D.step()
            # if no Inf...
            with MarkHTStep(HABANA_ENABLED and HABANA_LAZY):
                scaler_D.step(optim_D)  # Calls: optim_D.step() internally
            scaler_D.update()

            # (2) Update Net_G: maximize log(D(G(z)))
            ###################################################################
            net_G.zero_grad()
            label.fill_(1.0)  # `fake` labels are real for generator cost

            # Forward pass fake batch through Net_D; Calculate G_loss &&
            # Calculate gradients for Net_G
            with torch.cuda.amp.autocast():
                output = net_D(fake).view(-1)
                err_G = criterion(output, label)

            D_G_z2 = torch.sigmoid(output).mean().item()
            scaler_G.scale(err_G).backward()

            # Mark Habana Steps => Generator Optim;
            with MarkHTStep(HABANA_ENABLED and HABANA_LAZY):
                scaler_G.step(optim_G)  # Calls: optim_G.step()
            scaler_G.update()

            # If profiling enabled; then mark step...
            if enable_prof:
                prof.step()

            # Log Metrics to STDOUT or SAVE TO DISK
            ###################################################################
            if (epoch_step % model_cfg.log_frequency) == 0:
                print(
                    f" [{datetime.datetime.utcnow().__str__()}] [{epoch}/{n_epochs}][{epoch_step}/{len(dl)}] Loss_D: {err_D.item():.4f} Loss_G: {err_G.item():.4f} D(x): {D_X:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                )

                if enable_logging:
                    # Write Metrics to TensorBoard...
                    for metric, val in zip(
                        ["G_loss", "D_loss", "D_X", "D_G_z1", "D_G_z2"],
                        [err_G.item(), err_D.item(), D_X, D_G_z1, D_G_z2],
                    ):
                        writer.add_scalar(
                            metric,
                            val,
                            (epoch * len(dl.dataset))
                            + (epoch_step * train_cfg.batch_size),
                        )
                        writer.flush()

                    # Save Losses (and a few other function values) for plotting...
                    losses["_G"].append(err_G.item())
                    losses["_D"].append(err_D.item())

        # Save Model && Progress Images Every N Epochs
        if (epoch % model_cfg.save_frequency == 0) | (epoch == n_epochs - 1):

            # Add Epoch End Imgs...
            with torch.no_grad():
                fake = net_G(fixed_noise).detach().cpu()
                img_list.append(
                    vutils.make_grid(fake, padding=2, normalize=True)
                )

            # Save Checkpoint
            if train_cfg.dev == torch.device(f"cuda:{rank}"):
                dist.barrier()

            if hasattr(net_D, "module"):
                state_D = net_D.module.state_dict()
                state_G = net_G.module.state_dict()

            else:
                state_D = net_D.state_dict()
                state_G = net_G.state_dict()

            torch.save(
                {
                    "epoch": epoch,
                    "D_state_dict": state_D,
                    "G_state_dict": state_G,
                    "D_optim": optim_D.state_dict(),
                    "G_optim": optim_G.state_dict(),
                    "losses": losses,
                    "img_list": img_list,
                    "noise": fixed_noise,
                },
                f"{model_cfg.model_dir}/{model_cfg.model_name}/checkpoint_{epoch}.pt",
            )

            # Exit Writer and Profiler

        if (enable_prof) and ("1.10.0" in torch.__version__):
            prof.stop()

        if enable_logging:
            writer.close()

    return {"losses": losses, "img_list": img_list}
