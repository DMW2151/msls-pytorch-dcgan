# Core script for running DCGAN -> Implements the DCGAN architecture from
# the original GAN/DCGAN papers in PyTorch with slight adjustments to the
# optimizer. Other changes discussed in post
#
# See: https://arxiv.org/abs/1406.2661

# General
import datetime
import os
from dataclasses import dataclass

# Torch
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


# DCGAN
from msls_dcgan_utils import MarkHTStep

if torch.__version__ == "1.10.0":
    import torch.profiler
    from torch.profiler import ProfilerActivity
    import torch.cuda.amp as amp

# Init Habana Values
HABANA_ENABLED = 0
HABANA_LAZY = 0

# Habana Imports - will fail if not on a Habana DL AMI instance
try:
    from habana_frameworks.torch.utils.library_loader import load_habana_module
    from habana_frameworks.torch.hpex.optimizers import FusedAdamW
    import habana_frameworks.torch.core as htcore
    import habana_dataloader

    load_habana_module()

    HABANA_ENABLED = 1
    HABANA_LAZY = 1
    os.environ["PT_HPU_LAZY_MODE"] = "1"
    os.environ["GRAPH_VISUALIZATION"] = True

except ImportError:
    # Failed imports, will not use HPU drivers/shared libs
    pass


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

    dev: torch.device
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
    ngpu: int = int(torch.cuda.device_count())  # No Support for Multi GPU!!
    data_root: str = "/data/images/train_val"

    def _announce(self):
        """Show Pytorch and CUDA attributes before Training"""

        print("====================")
        print(self.dev.__repr__())
        print(f"Pytorch Version: {torch.__version__}")

        if torch.cuda.device_count():
            print(f"Running with {torch.cuda.device_count()} GPUs Available.")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            for i in range(torch.cuda.device_count()):
                print("device %s:" % i, torch.cuda.get_device_properties(i))

            try:
                print(torch._C._cuda_getCompiledVersion(), "cuda compiled version")
                print(torch._C._nccl_version(), "nccl")
            except AttributeError:
                pass

        print("====================")

    def get_net_D(self, gpu_id=None):
        """
        Instantiate a Disctiminator Network:

        -   Note on Adam vs AdamW: Uses AdamW over Adam. In general, DCGAN and
            Adam are both
            susceptible to over-fitting on early samples. AdamW adds a
            weight decay parameter
            (default=0.01) on the optimizer for each model step

        -   Note on FusedAdamW vs AdamW: AdamW loops over parameters and
            launches kernels for each parameter when running the optimizer.
            This
            is CPU bound and can be a bottleneck
            on performance. FusedAdamW can batch the elementwise updates
            applied to all the
            model’s parameters into one or a few kernel launches.

        See: Fixing Weight Decay Regularization in Adam
        # https://arxiv.org/abs/1711.05101

        The Habana FusedAdamW optimizer uses a custom Habana implementation of
        `apex.optimizers.FusedAdam`,
        on Habana machines, enable this, otherwise use regular `AdamW`.
        """

        # Instantiate Discriminator Net, # Put model on device(s),
        # Enable Data Parallelism across all available GPUs
        net_D = Discriminator(self)

        if (torch.cuda.is_available()) and (torch.cuda.device_count() > 1):
            net_D = nn.parallel.DistributedDataParallel(net_D, device_ids=[gpu_id])

        else:
            net_D.to(self.dev)

        if HABANA_ENABLED:
            # Will fail if not on a Habana DL AMI Instance; See Note
            # on FusedAdamW
            optim_D = FusedAdamW(
                net_D.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                weight_decay=0.01,
            )
            return net_D, optim_D

        # If not on Habana, then use Adam optimizer...
        optim_D = optim.AdamW(
            net_D.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=0.01,
        )
        return net_D, optim_D

    def get_net_G(self, gpu_id=None):
        """
        Instantiate a Generator Network - See notes on `get_net_D` re specific
        optimizer choices.
        """

        # Enable Data Parallelism across all available GPUs && Put model on
        # device(s)
        net_G = Generator(self)

        if (torch.cuda.is_available()) and (torch.cuda.device_count() > 1):
            net_G = nn.parallel.DistributedDataParallel(net_G, device_ids=[gpu_id])

        else:
            net_G.to(self.dev)

        if HABANA_ENABLED:
            # Will fail if not on a Habana DL AMI Instance; See Note on
            # FusedAdamW
            optim_G = FusedAdamW(
                net_G.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=self.lr,
                betas=(self.beta1, self.beta2),
            )

            return net_G, optim_G

        # If not on Habana, then use Adam optimizer...
        optim_G = optim.AdamW(
            net_G.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=0.01,
        )

        return net_G, optim_G


@dataclass
class ModelCheckpointConfig:
    """
    ModelCheckpointConfig holds the model save parameters for both the
    generator and discriminator networks.s
    --------
    Example: Rename the model and decrease save frequency from every epoch (1)
    to every fourth epoch (4)

    model_cfg = dcgan.ModelCheckpointConfig(
        model_name=msls_dcgan_habana_001,
        save_frequency=4
    )
    """

    model_name: str = "msls_dcgan_001"  # Name of the Model
    # Directory to save the model checkpoints to; Requires `/efs/trained_model`
    # has permissions s.t. ${USER} can write.
    model_dir: str = "/efs/trained_model"
    save_frequency: int = 1  # Save a model checkpoint every N epochs
    log_frequency: int = 50  # Print logs to STDOUT every N batches
    gen_progress_frequency: int = 1000  # Save progress images every N batches


def weights_init(m):
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


# Generator Code
class Generator(nn.Module):
    """
    Generator Net. The generator is designed to map the latent space vector (z)
    to believable data. Since our data are images, this means transforming (by
    default) a [1 x 100] latent vector to a 3 x 64 x 64 RGB image.

    Applies 4 x (Strided 2DConv, BatchNorm, ReLu) layers, and then a TanH
    layer to transform the output data to (-1, 1) for each channel (color)...
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
    Discriminator Net. Discriminator is a classifier network that (by default)
    takes a 3 x 64 x 64 image as input and outputs a probability that the image
    is from the set of real images

    Applies 1 x (Strided 2DConv, ReLu) + 3 x (Strided 2DConv, BatchNorm, ReLu)
    layers, and then a sigmoid layer to transform the output data to (0, 1).
    No different from Logit ;)
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
            # nn.sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


def instantiate_from_checkpoint(net_D, net_G, optim_D, optim_G, path):
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
    cur_epoch, losses, fixed_noise, img_list = instantiate_from_checkpoint(
            net_D, net_G, optim_D, optim_G, path
    )

    TODO: Probably not the most efficient use of memory here, could so
    something clever w. (de)serialization, but IMO, this is OK for now...
    """

    checkpoint = torch.load(path)

    # Seed Discriminator
    net_D.load_state_dict(checkpoint["D_state_dict"])
    optim_D.load_state_dict(checkpoint["D_optim"])

    # Seed Generator
    net_G.load_state_dict(checkpoint["G_state_dict"])
    optim_G.load_state_dict(checkpoint["G_optim"])

    return (
        checkpoint["epoch"],
        checkpoint["losses"],
        checkpoint["noise"],
        checkpoint["img_list"],
    )


def generate_fake_samples(n_samples, train_cfg, model_cfg, as_of_epoch=16):
    """
    Generates samples from a model checkpoint saved to disk, writes a few
    sample grids to disk and also returns last to the user
    --------
    Args:
        - n_samples - int - Number of samples to generate
        - train_cfg - TrainingConfig - Used to initialize the Generator model
        - model_cfg - ModelCheckPointConfig - Defines how to fetch the model
            checkpoint from disk
        - as_of_epoch - int - Epoch to generate samples as of - will fail
            if no model checkpoint is available
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
    Z = torch.randn(n_samples, train_cfg.nz, 1, 1, device=train_cfg.dev)

    # Initialize empty models && initialize from `as_of_epoch`
    net_D, optim_D = train_cfg.get_net_D()
    net_G, optim_G = train_cfg.get_net_G()

    path = f"{model_cfg.model_dir}/{model_cfg.model_name}/checkpoint_{as_of_epoch}.pt"

    _, _, _, _ = instantiate_from_checkpoint(net_D, net_G, optim_D, optim_G, path)

    # Use the Generator to create "believable" fake images - You can call a
    # plotting function on this output to visualize the images vs real ones

    # Ideally a Generator Net can use a CPU to (slowly) generate samples,
    # this confirms it, we can run the net through via CPU for "inference"
    generated_imgs = net_G(Z).detach().cpu()
    return generated_imgs


def get_msls_dataloader(rank, train_cfg):
    """ """

    default_loader_params = {
        "batch_size": train_cfg.batch_size,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": True,
        "timeout": 0,
        "prefetch_factor": 4,
        "persistent_workers": False,
    }

    # We can use an image folder dataset; depending on the size of the training
    # directory this can take a little to instantiate; about 3 min for 40GB
    #
    # Unclear why these specific parameters are needed for acceleration,
    # unclear if the `transforms.RandomAffine` ruins it.
    # See: https://docs.habana.ai/en/v1.1.0/PyTorch_User_Guide/PyTorch_User_Guide.html
    dataset = dset.ImageFolder(
        root=train_cfg.data_root,
        transform=transforms.Compose(
            [
                transforms.RandomAffine(degrees=0, translate=(0.3, 0.0)),
                transforms.CenterCrop(train_cfg.img_size * 4),
                transforms.Resize(train_cfg.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 406), (0.229, 0.224, 0.225)),
            ]
        ),
    )

    default_loader_params["dataset"] = dataset

    # Get Sampler for "Distributed" Training
    if train_cfg.dev == torch.device("cuda"):
        msls_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=torch.cuda.device_count(), rank=rank
        )
        default_loader_params["sampler"] = msls_sampler

    # If using Habana -> Try to Use the Habana DataLoader w. the params
    if HABANA_ENABLED:
        dataloader = habana_dataloader.HabanaDataLoader(**default_loader_params)

    else:
        dataloader = torch.utils.data.DataLoader(**default_loader_params)

    return dataloader


def get_msls_profiler(
    model_cfg, schedule={"wait": 2, "warmup": 2, "active": 6, "repeat": 4}
):

    prof = torch.profiler.profile(
        activities=[
            ProfilerActivity.CPU, 
            ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(**schedule),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"{model_cfg.model_dir}/{model_cfg.model_name}/events"
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )

    return prof


def get_msls_writer(model_cfg):
    return SummaryWriter(f"{model_cfg.model_dir}/{model_cfg.model_name}/events")


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
    if train_cfg.dev == torch.device("cuda"):
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=torch.cuda.device_count(),
            rank=rank,
        )

    # Initialize Net and Optimizers
    net_D, optim_D = train_cfg.get_net_D()
    net_G, optim_G = train_cfg.get_net_G()

    # Check the save-path for a model with this name && Load Params
    if st_epoch:
        cur_epoch, losses, fixed_noise, img_list = instantiate_from_checkpoint(
            net_D,
            net_G,
            optim_D,
            optim_G,
            f"{model_cfg.model_dir}/{model_cfg.model_name}/checkpoint_{st_epoch}.pt",
        )

    # If no start epoch specified; then apply weights from DCGAN paper and
    # proceed w. model training...
    else:
        net_G.apply(weights_init)
        net_D.apply(weights_init)
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
        prof = get_msls_profiler(model_cfg)

    if enable_logging:
        writer = get_msls_writer(model_cfg)

    # Start new training epochs...
    for epoch in range(cur_epoch, n_epochs):

        # Set Epoch Logging Iteration to 0 - For Plotting!
        log_i = 0

        if enable_prof:
            prof.start()

        if type(dl.sampler) == (torch.utils.data.distributed.DistributedSampler):
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
            D_X = 1  # torch.sigmoid(output).mean().item()

            # (1.2) Update D Network; Train with All-fake batch
            fake = net_G(Z)
            label.fill_(0.0)

            # Classify all fake batch with D && Calculate D_loss
            # Calculate the gradients for this batch, accumulated with
            # previous gradients &&\ Compute error of D as sum over the
            # fake and the real batches
            with amp.autocast():
                output = net_D(fake.detach()).view(-1)
                err_D_fake = criterion(output, label)

            scaler_D.scale(err_D_fake).backward()
            D_G_z1 = 1  # torch.sigmoid(output).mean().item()
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
            with amp.autocast():
                output = net_D(fake).view(-1)
                err_G = criterion(output, label)

            scaler_G.scale(err_G).backward()
            D_G_z2 = 1  # torch.sigmoid(output).mean().item()

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
                            + (log_i * model_cfg.log_frequency),
                        )

                    # Save Losses (and a few other function values) for plotting...
                    losses["_G"].append(err_G.item())
                    losses["_D"].append(err_D.item())

                    log_i += 1
                    writer.flush()

            # Save Sample Imgs Every N Epochs && save the progress on the
            # fixed latent input vector for plotting
            if (epoch_step % model_cfg.gen_progress_frequency) == 0:
                with torch.no_grad():
                    fake = net_G(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # Save Model && Progress Images Every N Epochs
        if (epoch % model_cfg.save_frequency == 0) | (epoch == n_epochs - 1):

            # Ensure the Save Directory Exists
            if not os.path.exists(f"{model_cfg.model_dir}/{model_cfg.model_name}"):
                os.makedirs(f"{model_cfg.model_dir}/{model_cfg.model_name}")

            # Save Checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "D_state_dict": net_D.state_dict(),
                    "G_state_dict": net_G.state_dict(),
                    "D_optim": optim_D.state_dict(),
                    "G_optim": optim_G.state_dict(),
                    "losses": losses,
                    "img_list": img_list,
                    "noise": fixed_noise,
                },
                f"{model_cfg.model_dir}/{model_cfg.model_name}/checkpoint_{epoch}.pt",
            )

            # Exit Writer and Profiler

        if (enable_prof) and (torch.__version__ == "1.10.0"):
            prof.stop()

        if enable_logging:
            writer.close()

    return {"losses": losses, "img_list": img_list}
