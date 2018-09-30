import numpy as np
import cv2
import math
from pyflann import *
import os

def genFragNdescr():
	directory = 'picturesRev'
	allDescriptors = []
	allFragments = []
	imageIndecies = []

	print(os.listdir(directory))

	for filename in os.listdir(directory):
		if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".PNG"): 
			#print(os.path.join(directory, filename))
			#im = cv2.imread('non_rigid_shape_A\\non_rigid_shape_A\\deer\\deer18.tif')
			im = cv2.imread(os.path.join(directory, filename))
			#cv2.imshow('?',im)
			height, width, channels = im.shape
			imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
			ret,thresh = cv2.threshold(imgray,127,255,0)
			height, width, channels = im.shape
			im3, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
			#blank_image = np.zeros((height,width,3), np.uint8)
			
			blank_image= np.full((height,width,3),255,np.uint8)

# >>> np.full((3, 5), 7, dtype=int)
# array([[7, 7, 7, 7, 7],
#        [7, 7, 7, 7, 7],
#        [7, 7, 7, 7, 7]])



			#cv2.drawContours(blank_image, contours, -1, (0,255,0), 3)

			#contours are are a list of contours, one contour is a list of points, eah point is an array 
			# (e.g. std::vector<std::vector<cv::Point> >).
			# list of contours, each contour is a lislt of points, a point is  - array whose first element is and array of 2(x,y)

			max = 0
			maxind = 0

			#this loop just makes sure we are looking at the longest contour if the algorithm finds multiple, as that likely our shape
			for x in range(len(contours)):
				contourLength = len(contours[x])
				if (contourLength>max):
					max = contourLength
					maxind = x
			#we store a proper list of points in listofpoints using this loop
			startlistOfPoints = []
			for ind, x in enumerate(contours[maxind]):
				xval = x[0][0]
				yval = x[0][1]
				startlistOfPoints.append([yval, xval])
			listOfPoints= np.array(startlistOfPoints)
			
			for x in listOfPoints:
				xval = x[1]
				yval = x[0]			
				blank_image[yval][xval] = [0,0,0]

			directory2 = directory+"\\new"
			cv2.imwrite( os.path.join(directory2, filename) , blank_image)


	# np.save("MPEG7dataset\\test\\extractions\\"+shape+"\\Descriptors", allDArray)
	# np.save("MPEG7dataset\\test\\extractions\\"+shape+"\\Fragments", allFArray)
	# np.save("MPEG7dataset\\test\\extractions\\"+shape+"\\metadata",imIndArray)


	# print("total Descriptors collected:", len(allDArray))
	# print("total Fragments collected:", len(allFArray))
	# a = np.load('MPEG7dataset\\test\\extractions\\'+shape+'\\Descriptors.npy')
	# b = np.load("MPEG7dataset\\test\\extractions\\"+shape+"\\Fragments.npy")
	# c = np.load("MPEG7dataset\\test\\extractions\\"+shape+"\\metadata.npy")

	# print("DescripptorArray deserialised size", len(a))
	# print("FragArray deserialized size", len(b))
	# print("meta: ",c)
genFragNdescr()