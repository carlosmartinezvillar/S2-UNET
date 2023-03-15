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
class ConfusionMatrix():
	'''
	Y and T are taken to be two arrays, each of matching NxM dimensions, corresponding to the       
	predicted values and the true labels in a class. Values of 1 or true in Y are the outputs of the
	network (predicted class), and values of 1 or true in T are pixels belonging to that class. The 
	confusion matrix of the 2 class problem is set as:

	           predicted
	             1   0
	          +----+----+
	       1  | TP | FN |
	actual    +----+----+
	       0  | FP | TN |
	          +----+----+

	For the case of 3 classes, a single confusion matrix is analagously set so that it includes all 
	three classes in one matrix. The diagonal always corresponds to the true positives of a class      
	but FNs, FPs, and TNs vary. For example, for class 0 the resulting confusion matrix is 

	             predicted
	            0    1    2
	          +----+----+----+
	       0  | TP | FN | FN |
	          +----+----+----+
	actual 1  | FP | TN | TN |
	          +----+----+----+
	       2  | FP | TN | TN |
	          +----+----+----+
	'''

	def __init__(self, n_classes=2):
		self.n_classes = n_classes
		self.M         = np.zeros((n_classes,n_classes))
		self.TP        = 0
		self.FP        = 0
		self.FN        = 0
		self.TN        = 0
		self.y_batches = None
		self.t_batches = None

	def update(self,Y,T):
		#for 2 classes, array of indices and mask are the same.
		if self.n_classes == 2:
			#one way
			# tp_mask = (Y==1) & (T==1)
			# fp_mask = tp_mask ^ (Y==1)
			# fn_mask = tp_mask ^ (T==1)
			# tn_mask = ~(tp_mask | fp_mask | fn_mask)
			#another way...
			# tnm = ~((Y==1) | (T==1))
			# tn_mask = (Y==0) & (T==0)
			# tnm = ~(Y==1) & ~(T==1)
			# self.TP += tp_mask.sum()
			# self.FP += fp_mask.sum()
			# self.FN += fn_mask.sum()
			# self.TN += tn_mask.sum()
			#yet another way...
			# self.M[0,0] += ((T==0) & (Y==0)).sum()
			# self.M[0,1] += ((T==0) & (Y==1)).sum()
			# self.M[1,0] += ((T==1) & (Y==0)).sum()			
			# self.M[1,1] += ((T==1) & (Y==1)).sum()
			# self.TP = self.M[0,0]
			# self.FN = self.M[0,1]
			# self.FP = self.M[1,0]
			# self.TN = self.M[1,1]
			self.TP += ((T==1) & (Y==1)).sum()
			self.FN += ((T==1) & (Y==0)).sum()
			self.FP += ((T==0) & (Y==1)).sum()			
			self.TN += ((T==0) & (Y==0)).sum()

			print(self.TP,self.FN)
			print(self.FP,self.TN)

		if self.n_classes == 3:
			# intersections and such -- coulda been modulo loop
			self.M[0,0] += ((T==0) & (Y==0)).sum()
			self.M[0,1] += ((T==0) & (Y==1)).sum() 
			self.M[0,2] += ((T==0) & (Y==2)).sum() 
			self.M[1,0] += ((T==1) & (Y==0)).sum()
			self.M[1,1] += ((T==1) & (Y==1)).sum()
			self.M[1,2] += ((T==1) & (Y==2)).sum()
			self.M[2,0] += ((T==2) & (Y==0)).sum()
			self.M[2,0] += ((T==2) & (Y==1)).sum()
			self.M[2,2] += ((T==2) & (Y==2)).sum()
		
			# t = T.flatten()
			# y = Y.flatten()
			# for i,j in zip(t,y):
				# self.M[i,j] += 1

			# 1x3 arrays with counts for each class
			self.TP = self.M.diagonal()
			self.FP = self.M.sum(axis=0) - self.TP
			self.FN = self.M.sum(axis=1) - self.TP
			self.TN = self.M.sum() - self.TP - self.FP - self.FN


	def append(self,Y,T):
		if self.y_batches is None:
			self.y_batches = Y
			self.t_batches = T
		else:
			self.y_batches = np.stack((self.y_batches,Y),axis=0)
			self.t_batches = np.stack((self.t_batches,T),axis=0)

	def ppv(self):
		# Precision -- predictive positive rate
		return self.TP/(self.TP + self.FP)

	def tpr(self):
		# Recall, sensitivity -- true positive rate
		return self.TP/(self.TP + self.FN)

	def acc(self):
		# Accuracy -- hit+correct rejections
		return (self.TP+self.TN)/(self.TP+self.FN+self.FP+self.TN)

	def iou(self,reverse=False):
		# Intersection over union, jaccard index, critical success index, whatever...
		if self.n_classes==2 and reverse:
			# land iou--maybe useful for 2-class
			return self.TN/(self.TN+self.FN+self.FP)
		return self.TP / (self.TP+self.FN+self.FP)

	def get_metrics(self):
		return 

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
	y0 = np.array([
		[0,0,0,0,0],
		[0,0,1,1,1],
		[0,0,1,1,1],
		[0,0,0,0,0],
		[0,0,0,0,0]])

	t0 = np.array([
		[0,0,0,0,0],
		[0,0,0,0,0],
		[0,1,1,0,0],
		[0,1,1,0,0],
		[0,0,0,0,0]])
	cm2 = ConfusionMatrix(n_classes=2)

	# for i in range(1000):
	cm2.update(y0,t0)

	#another check for 3-way classification
	#from array [B,C,x,y] after argmax axis=1, [B,x,y]
	y0 = np.array([[
		[0,0,0,2,2],
		[0,1,2,2,2],
		[0,0,1,2,0],
		[0,0,0,1,0],
		[0,0,0,0,0]]])
		
	t0 = np.array([[
		[0,1,2,2,2],
		[0,1,2,2,2],
		[0,0,1,2,2],
		[0,0,0,1,1],
		[0,0,0,0,0]]])

	cm3 = ConfusionMatrix(n_classes=3)
	# for i in range(1000):
		# cm3.update(y0,t0)