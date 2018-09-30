import numpy as np
import cv2
import math
from pyflann import *
import os

def genFragNdescr():
	shapes = ["apple","bell", "Bone","car","carriage","cellular_phone","children","chopper","face","flatfish","fountain","key","shoe","watch"]
	for shape in shapes:
		print("\n\n\n\n\n\n\n\n\n\n\n",shape)

		directory = 'MPEG7dataset\\test\\'+shape
		allDescriptors = []
		allFragments = []
		imageIndecies = []
		fragSizes = []
		print(os.listdir(directory))

		for filename in os.listdir(directory):
			if filename.endswith(".png"): 
				#print(os.path.join(directory, filename))
				#im = cv2.imread('non_rigid_shape_A\\non_rigid_shape_A\\deer\\deer18.tif')
				im = cv2.imread(os.path.join(directory, filename))
				#cv2.imshow('?',im)
				height, width, channels = im.shape
				imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
				ret,thresh = cv2.threshold(imgray,127,255,0)
				height, width, channels = im.shape
				im3, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
				blank_image = np.zeros((height,width,3), np.uint8)
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
				#fraglist will hold all the fragments in this image
				fragSizes.append(len(listOfPoints))
				continue
			else:
				continue
		 
		sizesArray = np.array(fragSizes)
		np.save("MPEG7dataset\\test\\extractions\\"+shape+"\\lengths",sizesArray)

genFragNdescr()