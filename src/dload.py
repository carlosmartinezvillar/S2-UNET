import os
import time
import sys
import numpy as np
import torch
# import torch.nn.functional as F #if pad collate is needed
# import multiprocessing
import torchvision as tv
from pillow import Image

import utils

################################################################################
# FILE PATHS
################################################################################
DATA_DIR = "../dat/"


################################################################################
# CLASSES
################################################################################
class SentinelDatasetTorchvision(torch.utils.data.Dataset):
	def __init__(self,image_list,get_func='PIL'):
		self.BATCH_SIZE = 16
		self.CHANNELS   = 3
		self.images = image_list[:,0]
		self.labels = image_list[:,1]

		assert get_func is not None, "Image opening function get_func is undefined."
		if get_func == 'PIL':
			self.get_func = self.pillow_open()
		if get_func == 'torchvision':
			self.get_func = self.torch_open()

	def __len___(self):
		return len(self.images)

	def __getitem__(self,idx):
		#do something important here...)
		x = torch.zeros((self.BATCH_SIZE,self.CHANNELS,512,512))
		t = torch.zeros((self.BATCH_SIZE,self.CHANNELS,512,512))
		x,t = get_func(idx)
		return (img,lbl)

	def flip(self):
		#flip/transform here?
		pass

	def torch_open(self,idx):
		img = tv.io.read_image(self.images[idx,path],mode=tv.io.ImageReadMode=GRAY)
		lbl = tv.io.read_image(self.labels[idx,path],mode=tv.io.ImageReadMode=GRAY)
		return (img, lbl)

	def pillow_open(self,idx):
		pass

################################################################################
# FUNCTIONS
################################################################################
def train_validation_split():
	'''
	returns training_set, validation_set
	'''
	pass

def train_test_split():
	# returns tr_set, va_set, test_set
	pass