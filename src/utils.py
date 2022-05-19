# import gdal
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from PIL import Image, ImageDraw

def vector_to_raster(raster, vector):
	img = Image.new('L', (raster.height,raster.width),0)

	polygons     = vector['geometry']

	top_left     = raster.transform*[0,0]
	bottom_right = raster.transform*[raster.height,raster.width]

	# first_polygon = polygons.iloc[0]                     #polygon datatype
	first_polygon = list(polygons.iloc[0].exterior.coords) #list of pairs

	indexed_polygon = []
	for x,y in first_polygon:
		i,j = raster.index(x,y)
		indexed_polygon.append((i,j))
	return indexed_polygon

def read_raster_file(path):
	pass

def read_vector_file(path):
	return gpd.read_file(path)

def image_unit_norm(img):
	n = (img-img.min(axis=(1,2),keepdims=True))/(img.max(axis=(1,2),keepdims=True)-img.min(axis=(1,2),keepdims=True))
	return np.moveaxis(n,0,-1)



if __name__ == "__main__":
	pass