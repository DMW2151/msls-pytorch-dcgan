"""
silly-little-game that implements an API that generates images
and serves them, that's all...

Run the API container with...


sudo docker build . \
    -t dmw2151/deep-dash-api-flask

sudo docker run \
    -e TRAIN_CFG__IMG_SIZE=64 \
    -e MODEL_CFG__NAME=ottawa-msls-gpu-dcgan-64-001\
    -e ML_CONFIG_CHECKPOINT_NUM=16\
    --net=host \
    -v /efs/trained_model/:/efs/trained_model \
    dmw2151/deep-dash-api-flask python3 ./api/img_svc.py

"""

import uuid
import os
from typing import Union

import boto3

from flask import Flask, send_file

import torch
import torchvision.utils as vutils
from flask_cors import CORS
from msls.dcgan_utils import (
    ModelCheckpointConfig,
    TrainingConfig,
    get_checkpoint,
    restore_G_for_inference,
    gen_img_sequence_array,
)
from msls.gan import Generator64, Generator128

app = Flask(__name__)
CORS(app)


# Allow the training config to be modified at runtime w. environment vars
# really, this should just specify 64px or 128px and Gray or Color...
#
# Even for HPU models, I think it's safe to assume we'll serve the model from
# either a spot GPU instance, CPU instance, or a CPU instance with GPU
# acceleration...
DEVICE = "cuda" if (torch.cuda.is_available()) else "cpu"

TRAIN_CFG = TrainingConfig(
    dev=torch.device(DEVICE),
    **{
        "nc": os.environ.get("TRAIN_CFG__N_CHANNELS") or 3,
        "img_size": os.environ.get("TRAIN_CFG__IMG_SIZE") or 64,
    },
)

# Allow the model config to be modified at runtime w. environment vars
MODEL_CFG = ModelCheckpointConfig(
    **{
        "name": os.environ.get("MODEL_CFG__NAME") or "msls-gaudi-dcgan-128-001",
        "root": os.environ.get("MODEL_CFG__ROOT") or "/efs/trained_model",
        "s3_bucket": os.environ.get("MODEL_CFG__BUCKET")
        or "dmw2151-habana-model-outputs",
    }
)


def get_generator(
    train_cfg: TrainingConfig,
    model_cfg: ModelCheckpointConfig,
    epoch: int = 8,
) -> Union[Generator128, Generator64]:
    """Get a model from storage to use as a generator..."""

    # Check if the desired checkpoint is local, or if we need to download
    # from cloud storage, silly implementation of `touch`...
    slim_checkpoint_path = (
        f"{model_cfg.root}/{model_cfg.name}/slim_checkpoint_{epoch}.pt"
    )
    try:
        with open(slim_checkpoint_path, "rb") as fi:
            fi.close()

    # If only on the remote, then try to get it from S3, save local, and
    # *then* open...
    except FileNotFoundError:
        s3 = boto3.client("s3")

        with open(slim_checkpoint_path, "wb", os.O_CREAT) as fi:

            s3.download_fileobj(
                model_cfg.s3_bucket,
                f"{model_cfg.name}/slim_checkpoint_{epoch}.pt",
                fi,
            )

    # Get from local storage, kinda inefficient, but oh well...
    checkpoint = get_checkpoint(path=slim_checkpoint_path, cpu=True)

    # Create an un-initialized model to load the weights from our
    # pre-trained model..
    G, _ = train_cfg.get_network(Generator64, device_rank=0)

    restore_G_for_inference(checkpoint, G)
    return G


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers["Cache-Control"] = "public, max-age=0"
    return r


@app.route("/health")
def health_check():
    return {"Health": "OK"}


@app.route("/")
def root_check():
    return {"Health": "OK"}


@app.route("/gan_imgs")
def generate_static_img():
    """Generates static images from the Generator model..."""

    Z = torch.randn(
        16,
        TRAIN_CFG.nz,
        1,
        1,
        device=TRAIN_CFG.dev,
    )

    g = vutils.make_grid(
        G(Z).detach().to(DEVICE), padding=4, normalize=True, nrow=4
    ).cpu()

    # TODO: Use tmpfile instead of this mess...
    tmp_img_hash = uuid.uuid4().__str__()
    vutils.save_image(g, f"/tmp/{tmp_img_hash}.png")

    return send_file(f"/tmp/{tmp_img_hash}.png", mimetype="image/png")


@app.route("/gan_gifs")
def generate_gif():
    """Serves a Gif - Series of predictions w. SLERP"""

    # TODO: Use tmpfile instead of this mess...
    tmp_gif_location = gen_img_sequence_array(
        MODEL_CFG, G, n_frames=16, Z_size=TRAIN_CFG.nz
    )

    return send_file(tmp_gif_location, mimetype="image/gif")


if __name__ == "__main__":

    G = get_generator(
        TRAIN_CFG,
        MODEL_CFG,
        epoch=int(os.environ.get("ML_CONFIG_CHECKPOINT_NUM", "8")),
    )

    app.run(debug=True, host="0.0.0.0")
