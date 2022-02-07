# Command line wrapper around gaudi_dcgan.py

import argparse
import os
import socket

import torch
import torch.multiprocessing as mp


from dcgan_utils import ModelCheckpointConfig, TrainingConfig

if torch.cuda.is_available():
    import gpu_dcgan as dcgan

    DEVICE = "cuda"
else:
    import gaudi_dcgan as dcgan

    DEVICE = "hpu"


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
    "-ne",
    "--n_epoch",
    type=int,
    help="Train model through N epochs",
    default=16,
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

# Assumes Single Node w. DDP...
os.environ["MASTER_ADDR"] = socket.gethostbyname(socket.gethostname())
os.environ["MASTER_PORT"] = "8888"

if __name__ == "__main__":

    args = parser.parse_args()

    # Init Model Config && Create directory for this run...
    model_cfg = ModelCheckpointConfig(
        name=args.name,
        root=args.model_dir,
        log_frequency=args.logging_freq,
        save_frequency=args.checkpoint_freq,
        gen_progress_frequency=args.progress_freq,
    )

    # Create Location For Model Outputs
    model_cfg.make_all_paths()

    # Create Training Config && Announce Model Training Situation...
    train_cfg = TrainingConfig(
        batch_size=args.batch,
        dev=torch.device(DEVICE),
        data_root=args.dataroot,
    )

    train_cfg._announce()

    # ================================================================
    # Run in distributed mode;l but on a single node...
    mp.spawn(
        dcgan.start_or_resume_training_run,
        nprocs=dcgan.WORLD_SIZE,
        args=(
            train_cfg,
            model_cfg,
            args.n_epoch,
            args.s_epoch,
            args.profile,
            args.logging,
        ),
        join=True,
    )
