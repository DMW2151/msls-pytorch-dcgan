# Run on Deep Learning AMI Habana PyTorch 1.10.0 SynapseAI 1.2.0 (Ubuntu 18.04) 
# /usr/bin/python3.7 run_example.py
import random
import torch
import torch.nn as nn

from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()
import habana_frameworks.torch.core as htcore

# Run Vars
IMAGE_SIZE = 64
LAZY = True
NC = 3
NDF = 64
device = torch.device("hpu")

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(in_channels=NC, out_channels=NDF, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, padding_mode='zeros', bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (NDF) x 32 x 32
			nn.Conv2d(in_channels=NDF, out_channels=NDF * 2, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, padding_mode='zeros', bias=False),
			nn.BatchNorm2d(NDF * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (NDF*2) x 16 x 16
			nn.Conv2d(in_channels=NDF * 2, out_channels=NDF * 4, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, padding_mode='zeros', bias=False),
			nn.BatchNorm2d(NDF * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (NDF*4) x 8 x 8
			nn.Conv2d(in_channels=NDF * 4, out_channels=NDF * 8, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, padding_mode='zeros', bias=False),
			nn.BatchNorm2d(NDF * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (NDF*8) x 4 x 4
			nn.Conv2d(in_channels=NDF * 8, out_channels=1, kernel_size=4, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', bias=False),
			nn.Sigmoid(),
		)

	def forward(self, input):
		return self.main(input)

# Set Seed
random.seed(2151)
torch.manual_seed(2151)

# Create the Network 
netD = Discriminator().to(device)
permute_params(netD, True, LAZY)

# Create batch of latent vectors to test D
noise_img = torch.randn(
    1, 3, 64, 64, device=device
)

output = netD(noise_img)

## ON CPU: tensor([[[[0.4511]]]], grad_fn=<SigmoidBackward0>)
## ON HPU: RuntimeError: Number of input channels doesn't match weight channels times groups weight_channel = 2input_channel = 364 3 4 4 1 64 64 3 groups = 1
print(output)
