import msls.gaudi_dcgan as dcgan
import torch

from msls.dcgan_utils import (
    TrainingConfig,
    ModelCheckpointConfig,
)

from msls.gan import (
    Discriminator128,
    Generator128,
)


dcgan.init_habana_default_params()

T_cfg = TrainingConfig(
    **{
        "nc": 3,
        "nz": 128,
        "ngf": 128,
        "ndf": 32,
        "ngpu": 0,
        "batch_size": 128,
        "data_root": "/data/imgs/test",
    },
    dev=torch.device("hpu")
)

M_cfg = ModelCheckpointConfig(
    **{
        "name": "gaudi-global-001",
        "root": "/efs/trained_model/",
        "log_frequency": 50,
        "save_frequency": 1,
    }
)

dcgan.start_or_resume_training_run(
    "1",
    T_cfg,
    M_cfg,
    16,
    0,
    False,
    False,
)
