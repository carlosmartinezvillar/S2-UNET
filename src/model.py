import torch
import torch.functional as F

class ResUnit(torch.nn.Module):
	def __init__(self,ch_in,ch_out):
		super(ResUnit,self).__init__()

		self.layers = torch.nn.Sequential(
			torch.nn.

			)

class DownBlock(torch.nn.Module):
	def __init__(self,ch_in,ch_out):
		super(DownBlock,self).__init__()

		self.layers = torch.nn.Sequential(
			torch.nn.Conv2d(ch_in,ch_out,3,1),
			torch.nn.BatchNorm2d(),
			torch.nn.ReLU()
			)

	def forward(self,x):
		return self.layers(x)


class UpBlock(torch.nn.Module):
	def __init__(self,ch_in,ch_out):
		super(UpBlock,self).__init__()

		self.layers = torch.nn.Sequential(
			torch.nn.Conv2d(ch_in,ch_out,3,1)
			torch.BatchNorm2d(),
			torch.nn.ReLU()
			)

		#todo--residuals herE? or...


	def forward(self,x):
		x = self.layers(x)
		return x


class UNet(torch.nn.Module):
	def __init__(self,P):
		super(UNet,self).__init__()
		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()

		#append downblocks
		#todo--residuals hier?

		#append upblocks
		#todo--residuals hier?


	def forward(self,x):
		for e in self.encoder:
			x = e(x)

		#todo--catch residuals

		for d in self.decoder:
			x = d(x)

		#todo--append skip connections