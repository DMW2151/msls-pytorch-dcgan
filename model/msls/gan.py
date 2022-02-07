import torch.nn as nn


class Generator(nn.Module):
    """
    Generator Network — The generator maps the latent space vector (z)
    to believable images.

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
    Discriminator — Discriminator is a classifier network that takes an image
    as input and produces the probability that the image is from the set of
    real images

    Applies:
        - 1 x (Strided 2DConv, ReLu)
        - 3 x (Strided 2DConv, BatchNorm, ReLu)

    And then (normally) a sigmoid layer to transform the output data to (0, 1).
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
