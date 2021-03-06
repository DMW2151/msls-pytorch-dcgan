"""
Functions for running a PyTorch implementation of DCGAN on a GPU instance

sudo python3 -m msls.run_dcgan     \
    -c '{"name": "phx-msls-gpu-dcgan-64-001", "root": "/efs/trained_model/", "log_frequency": 250, "save_frequency": 4}'\
    -t '{"nc": 3, "nz": 128, "ngf": 128, "ndf": 32, "lr": 0.0002, "beta1": 0.5, "beta2": 0.999, "batch_size": 128, "img_size": 64, "weight_decay": 0.05}'\
    --s_epoch 0\
    --n_epoch 128\
    --dataroot /data/imgs/train_val/phoenix\
    --logging True \
    --profile True \
    --s3_bucket 'dmw2151-habana-model-outputs'

"""

import datetime

from packaging import version

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.profiler
import torch.cuda.amp as amp

from msls.dcgan_utils import (
    TrainingConfig,
    ModelCheckpointConfig,
    DEFAULT_LOADER_PARAMS,
    get_checkpoint,
    restore_model,
    weights_init,
)

from msls.gan import (
    Discriminator64,
    Generator64,
    Discriminator128,
    Generator128,
)

WORLD_SIZE = torch.cuda.device_count()


class GaussianNoise(object):
    """Add Noise to a tensor; reduce tendency for model collapse"""

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.1,
    ):
        self.std = std
        self.mean = mean

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t + torch.randn(t.size()) * self.std + self.mean


def get_msls_dataloader(
    rank: int,
    train_cfg: TrainingConfig,
    params: dict = DEFAULT_LOADER_PARAMS,
    use_ddp: bool = False,
) -> torch.utils.data.DataLoader:
    """Returns a PyTorch DataLoader w. special handling for MSLS dataset"""

    # Apply transformations tailored for MSLS
    dataset = torchvision.datasets.ImageFolder(
        root=train_cfg.data_root,
        transform=transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.0, 0.0),
                ),
                # transforms.Resize(train_cfg.img_size),
                transforms.CenterCrop(train_cfg.img_size),
                transforms.ToTensor(),
                # GaussianNoise(0.0, 0.01),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ]
        ),
    )

    # Replace default args w. training config where overlap
    for field in train_cfg.__dataclass_fields__:
        if field in params.keys():
            params[field] = train_cfg.__getattribute__(field)

        # Do not move `prefetch_factor` or `persistent_workers` when
        # trying to plot a chart on Sagemaker (or any other old torch)
        if version.parse(torch.__version__).release < (1, 8, 0):
            params.pop("persistent_workers", None)
            params.pop("prefetch_factor", None)

    # Create a torch.DDPSampler for DDP Loading...
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.cuda.device_count(),
            rank=rank,
            shuffle=False,
        )
        params["sampler"] = sampler

    # Add dataset and sampler to DEFAULT_LOADER_PARAMS...
    params["dataset"] = dataset

    return torch.utils.data.DataLoader(**params)


def start_or_resume_training_run(
    rank: int,
    train_cfg: TrainingConfig,
    model_cfg: ModelCheckpointConfig,
    n_epochs: int,
    st_epoch: int,
    enable_prof: bool = True,
    enable_logging: bool = True,
) -> dict:
    """
    Start a training run from `START_EPOCH` and go until `NUM_EPOCHS`
    using the parameters given in `train_cfg` and `model_cfg`.
    """

    torch.manual_seed(0)

    train_cfg.dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(int(rank))

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=torch.cuda.device_count(),
        rank=rank,
    )

    # Initialize Both Networks and Optimizers @ either very-small (64^2) or
    # small (128^2) size...
    if train_cfg.img_size == 64:
        D, opt_D = train_cfg.get_network(Discriminator64, device_rank=rank)
        G, opt_G = train_cfg.get_network(Generator64, device_rank=rank)
    elif train_cfg.img_size == 128:
        D, opt_D = train_cfg.get_network(
            Discriminator128,
            device_rank=rank,
        )
        G, opt_G = train_cfg.get_network(Generator128, device_rank=rank)
    else:
        raise NotImplementedError

    # Check the save-path for a model with this name && Load Params
    if st_epoch:
        checkpt = get_checkpoint(
            path=model_cfg.checkpoint_path(st_epoch),
            cpu=True,
        )

        restore_model(checkpt, G, D, opt_G, opt_D)

        cur_epoch = checkpt["epoch"]
        losses = checkpt["losses"]
        Z_fixed = checkpt["noise"]
        img_list = checkpt["img_list"]

    # If no start epoch specified; then apply weights from DCGAN paper, init
    # latent vector, training params dict, etc. && proceed w. model training...
    else:
        G.apply(weights_init)
        D.apply(weights_init)
        cur_epoch = 1
        img_list = []
        losses = {"_G": [], "_D": []}
        Z_fixed = torch.randn(
            64,
            train_cfg.nz,
            1,
            1,
            device=train_cfg.dev,
        )

    # Initialize Stateless BCELoss Function
    criterion = nn.BCEWithLogitsLoss().to(train_cfg.dev)
    scaler_D = torch.cuda.amp.GradScaler()
    scaler_G = torch.cuda.amp.GradScaler()

    # BUG/NOTE: [RESOLVED]: Profiling results can get distorted if
    # the number of batches per epoch is too small
    if enable_prof:
        prof = model_cfg.get_msls_profiler()
        prof.start()

    if enable_logging:
        writer = model_cfg.get_msls_writer()

    dl = get_msls_dataloader(rank, train_cfg, use_ddp=True)

    # Begin the Training Cycle...
    for epoch in range(cur_epoch, n_epochs + 1):

        # If running with DDP; set the epoch to prevent deterministic order
        if type(dl.sampler) == (
            torch.utils.data.distributed.DistributedSampler
        ):
            dl.sampler.set_epoch(epoch)

        # For Each Batch...
        for epoch_step, batch in enumerate(dl, start=0):

            ###################################################################
            # (0.1) Generate labels + input noise for this batch (GPU)
            real_imgs = batch[0].to(train_cfg.dev)  # Img Batch (CPU) -> (GPU)
            b_size = real_imgs.size(0)

            # TODO: Try Soft Noise on Labels...
            Z = torch.randn(
                b_size,
                train_cfg.nz,
                1,
                1,
                device=train_cfg.dev,
            )

            label = torch.full(
                (b_size,),
                1.0,
                device=train_cfg.dev,
            )

            ###################################################################
            # (1.1) Update D: all real batch
            D.zero_grad()

            # Forward pass && Calculate D_loss
            with amp.autocast():
                output = D(real_imgs.detach()).view(-1)
                err_D_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            scaler_D.scale(err_D_real).backward()
            D_X = torch.sigmoid(output).mean().item()

            # (1.2) Update D: all fake batch
            fake = G(Z)
            label.fill_(0.0)

            # Classify fake w. D && Calculate D_loss && get Loss of D as sum
            # over the fake and the real batches
            with amp.autocast():
                output = D(fake.detach()).view(-1)
                err_D_fake = criterion(output, label)

            D_G_z1 = torch.sigmoid(output).mean().item()
            scaler_D.scale(err_D_fake).backward()

            err_D = err_D_real + err_D_fake

            # Update D - Scalar_D.step() calls optim_D.step() internally
            scaler_D.step(opt_D)
            scaler_D.update()

            ###################################################################
            # (2) Update G: maximize log(D(G(z)))
            G.zero_grad()
            label.fill_(1.0)  # Labels swap for the generared batch pass

            # Forward pass fake batch through D; Calculate G_loss &&
            # Calculate gradients for G
            with amp.autocast():
                output = D(fake).view(-1)
                err_G = criterion(output, label)

            D_G_z2 = torch.sigmoid(output).mean().item()
            scaler_G.scale(err_G).backward()

            # Update G - Scalar_G.step() calls opt_G.step() internally
            scaler_G.step(opt_G)
            scaler_G.update()

            # If profiling enabled; then mark step...
            prof.step() if enable_prof else None

            ###################################################################
            # (3) Post Batch Metrics Collection
            if (epoch_step % model_cfg.log_frequency) == 0:
                print(
                    f" [{datetime.datetime.utcnow().__str__()}] [{epoch}/{n_epochs}][{epoch_step}/{len(dl)}] Loss_D: {err_D.item():.4f} Loss_G: {err_G.item():.4f} D(x): {D_X:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                )

                # Write Metrics to TensorBoard: (GPU) -> (CPU)
                if enable_logging:
                    imgs_processed_ct = (epoch * len(dl.dataset)) + (
                        epoch_step * train_cfg.batch_size
                    )

                    for metric, val in zip(
                        [
                            "G_loss",
                            "D_loss",
                            "D_X",
                            "D_G_z1",
                            "D_G_z2",
                        ],
                        [
                            err_G.item(),
                            err_D.item(),
                            D_X,
                            D_G_z1,
                            D_G_z2,
                        ],
                    ):
                        writer.add_scalar(
                            metric,
                            val,
                            imgs_processed_ct,
                        )
                    writer.flush()

                    # Save losses for metric plots...
                    losses["_G"].append(err_G.item())
                    losses["_D"].append(err_D.item())

        # Save model && progress images every N epochs
        if (epoch % model_cfg.save_frequency == 0) | (epoch == n_epochs):

            with torch.no_grad():
                generated_images = G(Z_fixed).detach().cpu()
                img_list.append(generated_images)

            # Save Checkpoint - block if running DDP
            if train_cfg.dev == torch.device(f"cuda:{rank}"):
                dist.barrier()

            # NOTE: in DDP vs Single GPU processing; `state_dict()` will have
            # different keys (really namespaces), leave that to reader...
            torch.save(
                {
                    "epoch": epoch,
                    "D_state_dict": D.state_dict(),
                    "G_state_dict": G.state_dict(),
                    "D_optim": opt_D.state_dict(),
                    "G_optim": opt_G.state_dict(),
                    "losses": losses,
                    "img_list": img_list,
                    "noise": Z_fixed,
                },
                model_cfg.checkpoint_path(epoch),
            )

    # Epoch's over; Stop Profiling!; This explicitly closes the profiler; will fail
    # if wait, warmup, active, and repeat batches can't fit on the first epoch
    prof.stop() if enable_logging else None

    return {
        "losses": losses,
        "img_list": img_list,
    }
