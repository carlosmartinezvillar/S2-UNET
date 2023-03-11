import os
import numpy as np
import torch
import random
import time
import sys

####################################################################################################
# FILE PATHS
####################################################################################################
LOG_DIR = "../log/"

####################################################################################################
# FUNCTIONS
####################################################################################################
# def train_mixed_precision(model,loss_fn,optimizer,dataloader,device=0):
# 	model.train()
# 	sum_loss = 0
# 	for batch_idx,(X,T) in enumerate(dataloader):
# 		X,T = X.cuda(device,non_blocking=True), T.cuda(device,non_blocking=True)
# 		optimizer.zero_grad()
# 		with torch.cuda.amp.autocast(enabled=True,dtype=toch.float16):
# 			Y = model(X)
# 			loss = loss_fn(Y,T)
# 		scaler.scale(loss).backward()
# 		scaler.unscale_(optimizer) #unscale for loss calc at fp32
# 		sum_loss += loss.item()
# 		scaler.step(optimizer)
# 		scaler.update()

def train(model,loss_fn,optimizer,dataloader,device=0):
	model.train()
	sum_loss   = 0

	for batch_id, (X,T) in enumerate(dataloader):
		#Batch to gpu
		X,T = X.to(device,non_blocking=True),T.to(device,non_blocking=True)

		#Fwd pass
		Y    = model(X)
		loss = loss_fn(Y,T)

		#Clear previous and compute new gradients
		optimizer.zero_grad()
		loss.backward()

		#Update weights
		optimizer.step()

		sum_loss += loss.item()	

		#Log training for debugging

	return sum_loss

def validate(model,loss_fn,optimizer,dataloader,device=0):
	model.eval()
	sum_loss = 0
	#TO DO


def train_and_validate(model, tr_dloader, va_dloader, optimizer, loss_fn):
	#TO DO


if __name__ == "__main__":
	pass