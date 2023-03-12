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
class SentinelDataset(torch.utils.data.Dataset):
	def __init__(self,image_list):
		self.images = image_list[:,0]
		self.labels = image_list[:,1]

	def __len___(self):
		return len(self.images)

	def __getitem__(self,idx):
		#do something important here...
		img = tv.io.read_image(self.images[idx],mode=tv.io.ImageReadMode.UNCHANGED)
		lbl = tv.io.read_image(self.labels[idx],mode=tv.io.ImageReadMode.UNCHANGED)
		return (img,lbl)

	def flip(self):
		#in case we would like to flip/transform the image on-the-fly rather than store it...
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