# Command line wrapper around gaudi_dcgan.py

import argparse
import os
import socket
import json

# Very Sensitive to Order of Imports...
try:
    import gaudi_dcgan as dcgan
    from dcgan_utils import ModelCheckpointConfig, TrainingConfig
    DEVICE = "hpu"

except ImportError as e:
    print(f"Failed HPU Import -> Using GPU: {e}")
    import msls.gpu_dcgan as dcgan
    from msls.dcgan_utils import ModelCheckpointConfig, TrainingConfig
    DEVICE = "cuda"

# Very Sensitive to Order of Imports...
import torch
import torch.multiprocessing as mp

if DEVICE == "hpu":
    print("Using HPU")
    torch.multiprocessing.set_start_method('spawn')

parser = argparse.ArgumentParser(description="Run MSLS DCGAN")

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
    "-ne",
    "--n_epoch",
    type=int,
    help="Train model through N epochs",
    default=16,
)

parser.add_argument(
    "-s3",
    "--s3_bucket",
    type=str,
    help="Bucket...",
    default="dmw2151-habana-model-outputs",
)

parser.add_argument(
    "-c",
    "--checkpoint_params",
    type=str,
    help="checkpoint_params",
    default=json.dumps(
        {
            "name": "msls-dcgan-128",
            "root": "/efs/trained_model/",
            "log_frequency": 50,
            "save_frequency": 1,
        }
    ),
)

parser.add_argument(
    "-t",
    "--train_params",
    type=str,
    help="training_params",
    default=json.dumps(
        {
            "nc": 3,
            "nz": 128,
            "ngf": 128,
            "ndf": 32,
            "lr": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999,
            "batch_size": 128,
            "img_size": 128,
            "weight_decay": 0.05,
        }
    ),
)

parser.add_argument(
    "-prof",
    "--profile",
    type=bool,
    help="Run the Torch profiler/save traces during training",
    default=False,
)

parser.add_argument(
    "-log",
    "--logging",
    type=bool,
    help="Enable logging to tensorboard",
    default=True,
)

# Assumes Single Node w. DDP...
os.environ["MASTER_ADDR"] = socket.gethostbyname(socket.gethostname())
os.environ["MASTER_PORT"] = "8888"

if __name__ == "__main__":

    args = parser.parse_args()

    # Init Model Config && Create directory for this run...
    model_cfg = ModelCheckpointConfig(**json.loads(args.checkpoint_params))

    # Create Location For Model Outputs
    model_cfg.make_all_paths()

    # Create Training Config && Announce Model Training Situation...
    train_cfg = TrainingConfig(
        data_root=args.dataroot,
        dev=torch.device(DEVICE),
        **json.loads(args.train_params)
    )

    train_cfg._announce()

    # ================================================================
    # Run in distributed mode;l but on a single node...
    if DEVICE == "hpu":
        dcgan.init_habana_params()
        dcgan.start_or_resume_training_run(
            1,
            train_cfg,
            model_cfg,
            args.n_epoch,
            args.s_epoch,
            args.profile,
            args.logging,
        )

    else:
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

    # On finish training -> send to s3
    if args.s3_bucket:
        model_cfg.slim_checkpoint_to_cloud_storage(
            args.s3_bucket, args.n_epoch
        )
