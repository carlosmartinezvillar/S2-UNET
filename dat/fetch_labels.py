import ee #earth engine Python API
import rasterio as rio
from PIL import Image
import os
import pyproj

# ee.Authenticate()
ee.Initialize()

####################################################################################################
#FUNCTIONS
####################################################################################################
# PARAMETER SETTING
def download(ee_object,crs):
	if isinstance(ee_object, ee.Image):
		print('Downloading single image...')
		url = ee_object.getDownloadUrl({
				'scale':10,
				'crs': crs,
				'format':'ZIPPED_GEO_TIFF'
			})

	# if isinstance(ee_object, ee.ImageCollection.ImageCollection):
	# 	print('Downloading image collection...')
	# 	obj_copy = ee_object.mosaic()
	# 	url     = obj_copy.getDownloadUrl({
	# 			'scale':10,
	# 			'crs': 'EPSG:4326',
	# 			'region': region
	# 		})

def split_image():
	#open one image/band and get crs

	#calculate region of new image via raster indices
	pass

def build_gee_id(s2_image_id):

	#open metadata to retrieve datastrip id
	pass

####################################################################################################
# MAIN
####################################################################################################
# ITERATE AND CALL DL
if __name__ == '__main__':
	test_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")	
	# lbl = ee.Image('GOOGLE/DYNAMICWORLD/V1/20190101T182741_20190101T182744_T11SPA')
	test_lbl = ee.Image('GOOGLE/DYNAMICWORLD/V1/20190101T182741_20190101T182744_T11SQV').select('label')

	#TODO -- get properties first 

	gee_crs = test_lbl.

	crs = 'EPSG:32611'
	url = download(test_lbl)
	print("Got URL: ", url)

	#TODO

	#TODO -- 

	
# geometry = ee.Geometry.Rectangle([80.058, 26.347, 82.201, 28.447])
# region = geometry.toGeoJSONString()#region must in JSON for
# path = downloader(MyImage,region)#call function
