import numpy as np
import torch
# import torch.nn.functional as F #if pad collate is needed
import os
import time
import sys
# import multiprocessing

import utils

################################################################################
# FILE PATHS
################################################################################
DATA_DIR = "../dat/"


################################################################################
# CLASSES
################################################################################
class SentinelDataset(torch.utils.data.Dataset):
	def __init__(self,image_list):
		self.images = image_list[:,0]
		self.labels = image_list[:,1]

	def __len___(self):
		return len(self.images)

	def __getitem__(self,idx):
		#do something important here...
		img = torchvision.io.read_image(self.images[idx],mode=torchvision.io.ImageReadMode.UNCHANGED)
		lbl = torchvision.io.read_image(self.labels[idx],mode=torchvision.io.ImageReadMode.UNCHANGED)
		return (img,lbl)

	def flip(self):
		#in case we would like to flip/transform the image on-the-fly rather than store it...
		pass


################################################################################
# FUNCTIONS
################################################################################
def train_test_split():
	# return tr_set, va_set, te_set
	pass