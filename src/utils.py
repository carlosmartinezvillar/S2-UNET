# import gdal
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import rasterio
# from PIL import Image, ImageDraw
import os
import json
import numpy as np
####################################################################################################
LOG_DIR = "../log/"
MOD_DIR = "../mod/"
CFG_DIR = "../cfg/"
####################################################################################################
# def vector_to_raster(raster, vector):
# 	img = Image.new('L', (raster.height,raster.width),0)

# 	polygons     = vector['geometry']

# 	top_left     = raster.transform*[0,0]
# 	bottom_right = raster.transform*[raster.height,raster.width]

# 	# first_polygon = polygons.iloc[0]                     #polygon datatype
# 	first_polygon = list(polygons.iloc[0].exterior.coords) #list of pairs

# 	indexed_polygon = []
# 	for x,y in first_polygon:
# 		i,j = raster.index(x,y)
# 		indexed_polygon.append((i,j))
# 	return indexed_polygon

# def read_raster_file(path):
# 	pass

# def read_vector_file(path):
# 	return gpd.read_file(path)

# def image_unit_norm(img):
# 	n = (img-img.min(axis=(1,2),keepdims=True))/(img.max(axis=(1,2),keepdims=True)-img.min(axis=(1,2),keepdims=True))
# 	return np.moveaxis(n,0,-1)

####################################################################################################
Class ConfusionMatrix():
	'''
	Y and T are taken to be two arrays, each of matching NxM dimensions, corresponding to the       
	predicted values and the true labels in a class. Values of 1 or true in Y are the outputs of the
	network (predicted values of a class), and values of 1 or true in T are pixels belonging to that
	belonging to that class. The confusion matrix is set as:

		      predicted
		        1   0
	         +----+----+
	      1  | TP | FN |
  	label    +----+----+
	      0  | FP | TN |
		     +----+----+

	For the case of 3 classes, the confusion matrix is set as

	             predicted
		        0    1    2
	         +----+----+----+
	      0  | TP | FN | FN |
  	         +----+----+----+
	label 1  | FP | TP | FN |
		     +----+----+----+
          2  | FP | FP | TP |
             +----+----+----+
	'''
	def __init__(self,n_classes=2,method='bool'):
		self.n_classes = n_classes
		self.M         = np.zeros((n_classes,n_classes))
		self.TP        = 0
		self.FP        = 0
		self.FN        = 0
		self.TN        = 0
		self.y_batches = None
		self.t_batches = None

	def update(Y,T):
		#for 2 classes, array of indices and mask are the same.
		if self.n_classes == 2:
			if method == 'bool':
				#using bool
				tpm   = (Y==1) & (T==1)
				fpm   = tpm ^ (Y==1)
				fnm   = tpm ^ (T==1)
				tnm   = ~(tpm | fpm | fnm)
				# tnm = ~((Y==1) | (T==1))
				# tnm = ~(Y==1) & ~(T==1)

			if method == 'int':
				#using ints
				tp = Y & T
				fp = Y - tp
				fn = T - tp
				tn = 1 - (Y | T)

			self.TP += tpm.sum()
			self.FP += fpm.sum()
			self.FN += fnm.sum()
			self.TN += tnm.sum()

		if self.n_classes == 3:
			#TODO
			pass

	def append(Y,T):
		if self.y_batches is None:
			self.y_batches = Y
			self.t_batches = T
		else:
			self.y_batches = np.stack((self.y_batches,Y),axis=0)
			self.t_batches = np.stack((self.t_batches,T),axis=0)

	def precision():
		return self.TP/(self.TP + self.FP)

	def recall():
		return self.TP/(self.TP + self.FN)

	def accuracy():
		return (self.TP+self.TN)/(self.TP+self.FN+self.FP+self.TN)

	def IoU(reverse=False):
		if reverse:
			return self.TN/(self.TN+self.FN+self.FP)
		return self.TP / (self.TP+self.FN+self.FP)

####################################################################################################
def IoU(Y,T,n_classes=2):
	'''
	2-class	
	'''
	if n_classes == 2:
		intersection = ((Y==1) & (T==1)).sum()
		union        = ((Y==1) | (T==1)).sum()
		return intersection/union
	if n_classes == 3:
		pass

def best_model_check():
	pass

def save_model(net,optimizer,tag):
	#path check
	j_id = os.getenv('JOB_ID')[-1]
	c_id = os.getenv('CONTAINER_ID')[-1]
	path = MOD_DIR + "%s_%s_%s.pt" % (j_id,c_id,tag)
	
	#save
	# torch.save(net.state_dict(),path)
	torch.save({
			'epoch': epoch,
			'model_state_dict': net.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,
		},path)

def load_model(net,optimizer,path):
	checkpoint = torch.load(path)
	net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss  = checkpoint['loss']
	return (epoch,loss)

def load_parameters(idx):
	with open(CFG_DIR + "parameters.json",'r') as fp:
		parameters = json.load(fp)
	return parameters

def save_parameters():
	pass

####################################################################################################
if __name__ == "__main__":

	#a small function check
	y0 = [
		[0,0,0,0,0],
		[0,0,1,1,1],
		[0,0,1,1,1],
		[0,0,0,0,0],
		[0,0,0,0,0]]
	y0 = np.array(y0)

	y1 = [
		[0,0,0,0,1],
		[0,0,0,1,0],
		[0,0,1,0,0],
		[0,0,0,0,0],
		[0,0,0,0,0]]
	y1 = np.array(y1)

	Y = np.stack((y0,y1),axis=0)

	t0 = [
		[0,0,0,0,0],
		[0,0,0,0,0],
		[0,1,1,0,0],
		[0,1,1,0,0],
		[0,0,0,0,0]]
	t0 = np.array(t0)

	t1 = [
		[0,1,1,1,1],
		[0,0,1,1,1],
		[0,0,0,1,1],
		[0,0,0,0,1],
		[0,0,0,0,0]]
	t1 = np.array(t1)

	T = np.stack((t0,t1),axis=0)
