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

		print(os.listdir(directory))

		for filename in os.listdir(directory):
			if filename.endswith(".png"): 
				im = cv2.imread(os.path.join(directory, filename))
				height, width, channels = im.shape
				imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
				ret,thresh = cv2.threshold(imgray,127,255,0)
				height, width, channels = im.shape
				im3, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
				blank_image = np.zeros((height,width,3), np.uint8)

				#contours are are a list of contours, one contour is a list of points, eah point is an array 
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
				#fraglist will hold all the fragments in this image
				fragList = []
				for ind, point in enumerate(listOfPoints):
					#for only every eightieth point
					if(ind%30==0) :	
						for ind2, point2 in enumerate(listOfPoints):
							#to only every eightieth point
							if(ind2%55==0)or((ind2%5 ==0)and (ind2-ind<50)):
								#we don't want fragments too long or too short
								threshUpper = len(listOfPoints)/12
								threshlower = len(listOfPoints)/20
								#we musn't be starting and ending at the same point
								if((listOfPoints[ind][0]!=listOfPoints[ind2][0]) and (listOfPoints[ind][1]!=listOfPoints[ind2][1])):
									fragment = []
									index = ind+1
									#if the fragment is in the array normally, i.e. start...end we can go straight
									#otherwise we have to loop from the back to the start again
									#if this was not done we would have no fragments crossing the start point of the contour
									if(ind2>ind):
										while index<ind2:
											curpoint = listOfPoints[index]
											fragment.append(curpoint)
											index = index+1
									else:
										while index<len(listOfPoints):
											#if not we must first go to the end of the list and add the points
											curpoint = listOfPoints[index]
											fragment.append(curpoint)
											index = index+1
										indexnew = 0;
										while indexnew<ind:
											#then we start from the beginning till we reach our end
											curpoint = listOfPoints[indexnew]
											fragment.append(curpoint)					
											indexnew = indexnew+1
									if((len(fragment)<threshUpper) and (len(fragment)>threshlower)):
										fragList.append(fragment)
				
				#metadata for an image
				myfirst = len(allFragments)
				allFragments.extend(fragList)
				myLast = len(allFragments)-1
				imageIndecies.append([myfirst,myLast])



				descriptorList = []
				
				#local is the number of points to take a descriptor around
				local = 3
				#rows is the number of spherical sections
				rows = 8
				#columns is the number of distance bins
				columns = 4
				#max is the furthest any two points can be from one another
				max = math.sqrt(height**2 + width**2)
				#s is then the base of the logarithmic scale
				s = math.log10(max)/(columns+1)
				trk = -1
				for ind in fragList:
					#ind is a fragment
					trk = trk+1
					#finaldescriptor is the concatenation of the local descriptors, this is flattened after
					finaldescriptor = np.zeros((0,columns), int)
					for x in range(1,local + 1):
						# mp is the index of the local point
						mp = (len(ind)*x)//(local+1)
						midpoint = ind[mp]
						descriptor = np.zeros((rows,columns), int)
						origX = np.asscalar(midpoint[0])
						origY = np.asscalar(midpoint[1])
						for point in ind:				
							if point[0]==midpoint[0] and point[1]==midpoint[1]:
								#dont add the local centre to the descriptor
								pass
							else:
								#for the bins we need relative distance from the local centre, r
								xish = np.asscalar(point[0])-origX
								yish = np.asscalar(point[1])-origY
								xSqd = (xish)**2
								ySqd = (yish)**2
								r = math.sqrt(xSqd + ySqd)
								#atan gives us the angle -pi to pi and so the circle sector
								theta = math.atan2(xish,yish)
								#now we iterate through boxes to find the one with the correct highest and lowest r, and theta
								for box in range(1,rows+1):
									if((box/rows*math.pi*2)-math.pi)>=theta:
										for tray in range(1, columns+1):
											if (10**(s*tray))>r:
												descriptor[box-1,tray-1]= descriptor[box-1,tray-1]+1.0
												break
										break
						finaldescriptor= np.concatenate([finaldescriptor,descriptor])
					finalflat = finaldescriptor.flatten()
					descriptorList.append(finalflat)
					allDescriptors.append(finalflat)
				
				continue
			else:
				continue
		 
		allDArray = np.array(allDescriptors)
		allFArray = np.array(allFragments)
		imIndArray = np.array(imageIndecies)


		np.save("MPEG7dataset\\test\\extractions\\"+shape+"\\Descriptors", allDArray)
		np.save("MPEG7dataset\\test\\extractions\\"+shape+"\\Fragments", allFArray)
		np.save("MPEG7dataset\\test\\extractions\\"+shape+"\\metadata",imIndArray)


		print("total Descriptors collected:", len(allDArray))
		print("total Fragments collected:", len(allFArray))
		# a = np.load('MPEG7dataset\\test\\extractions\\'+shape+'\\Descriptors.npy')
		# b = np.load("MPEG7dataset\\test\\extractions\\"+shape+"\\Fragments.npy")
		# c = np.load("MPEG7dataset\\test\\extractions\\"+shape+"\\metadata.npy")

		# print("DescripptorArray deserialised size", len(a))
		# print("FragArray deserialized size", len(b))
		# print("meta: ",c)
