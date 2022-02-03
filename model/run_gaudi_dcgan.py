# USE:

# Command line wrapper around gaudi_dcgan.py
# Sample Usage on Command Line - See Notes on Running on DL1
#
# python3 run_gaudi_dcgan.py \
#   --dataroot "/data/imgs/"\
#    --name msls_test_001 \
#   --s_epoch 0 \
#   --n_epoch 16

# General Deps
import os

import argparse

# Torch Deps
import torch
import socket
import torch.multiprocessing as mp

# DCGAN
import gaudi_dcgan as dcgan

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
    "-l",
    "--logging",
    type=bool,
    help="Enable logging to tensorboard",
    default=True,
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
    default=50,
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
    default=1024,
)

# Set Device to CPU, GPU or **HPU**
DEVICE = "cuda" if (torch.cuda.is_available()) else "cpu"
if dcgan.HABANA_ENABLED:
    DEVICE = "hpu"

os.environ[
    "MASTER_ADDR"
] = "localhost"  ## OR: socket.gethostbyname(socket.gethostname())
os.environ["MASTER_PORT"] = "8888"

if __name__ == "__main__":

    args = parser.parse_args()

    # Init Model Config && Create directory for this run...
    model_cfg = dcgan.ModelCheckpointConfig(
        model_name=args.name,
        model_dir=args.model_dir or "/efs/trained_model/",
        log_frequency=args.logging_freq,
        save_frequency=args.checkpoint_freq,
        gen_progress_frequency=args.progress_freq,
    )

    # Create Training Config && Announce Model Training Situation...
    train_cfg = dcgan.TrainingConfig(
        batch_size=args.batch,
        dev=torch.device(DEVICE),
        data_root=args.dataroot,
    )

    # Create Location For Model Outputs
    if not os.path.exists(f"{model_cfg.model_dir}/{model_cfg.model_name}/events"):
        os.makedirs(f"{model_cfg.model_dir}/{model_cfg.model_name}/events")

    train_cfg._announce()

    # ================================================================
    # Run in distributed mode;l but on a single node...
    mp.spawn(
        dcgan.start_or_resume_training_run,
        nprocs=torch.cuda.device_count(),
        args=(
            train_cfg,
            model_cfg,
            args.n_epoch,
            args.s_epoch,
            args.profile,
            args.logging,
        ),
    )
