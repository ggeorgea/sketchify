import numpy as np
import cv2
import math
from pyflann import *
import os
import random
import functionMisc
import sys
from matplotlib import pyplot as plt

#TODO
#ADD A DAMPENING FACTOR SO THAT EACH LINE ONLY LOSES SO MUCH
#contours to canny edges!

functionMisc.fill(([0,0],[10,10]))
# sketch synthesis plan


# edge detector, fairly high standards

# break up contours that are too long

# contour by contour replace with a fairly similar contour from an alternative image w similar feature vector? 
# with % chance adjustable

# peturb - rotate, lengthen, translate, crook
# with % chance adjustable

# add random noise edges
# with % chance adjustable


# get perm edge library extracted from existing dataset 
# get ANN
# edge detect 

flann = FLANN()

#for getting stuff from round, could replace choice in createDescriptorSet also
#from 1 to 0
def getRandomSection(length, maxDist, minDist):
	result = []
	while True:
		result = [(random.random()+0.0)*length//1,(random.random()+0.0)*length//1]
		#print(result,length,maxDist,minDist)
		if (result[0]> result[1] and result[0]> result[1]+minDist and result[0]<result[1]+maxDist) or (result[0]<result[1] and (length - result[1] + result[0]) > minDist and (length - result[1] + result[0]) < maxDist  ):
			break
	return result


#THis takes a contour, randomly chooses a part of the contour, randomly chooses some fragments and the part with the fragment that most improves rel
#want to take a contour, randomly choose a part (or systematically replace all), and replace it with the most similar fragment from library



testImBig = np.load('descLib.npy')
testFrags = np.load('fragLib.npy')
flann.build_index(testImBig,algorithm="kmeans", branching=32, iterations=7, checks=16)


def adjustRoundList(Round:list, a, b):
	fullRound = Round
	#print(a,b, len(fullRound)//3)
	length = len(fullRound)
	#messed up if
	if a>b:
		#print("!") 
		fullround2 = []
		for be in range(a, len(fullRound)):
			fullround2.append(fullRound[be])
		for ae in range(0,a):
			fullround2.append(fullRound[ae])
		b = length - b + a
		a = 0 
		fullRound = fullround2
	fLength =  b-a
	if(a< length//3):
		#print("a")
		while (a< length//3):
			fullRound = np.concatenate(( [fullRound[len(fullRound)-1]] , fullRound[0: len(fullRound)-1 :1] ))
			a = a+1
	if(a>length//3):
		#print("b")
		while (a> length//3):
			fullRound = np.concatenate(( fullRound[1: len(fullRound) :1] , [fullRound[0]]  ))
			a= a-1

	return fullRound


def generateDescriptor(fragment1,height,width, local = 3):	
	#local is the number of points to take a descriptor around
	#local = 3
	#rows is the number of spherical sections
	rows = 8
	#columns is the number of distance bins
	columns = 4
	#max is the furthest any two points can be from one another
	max = math.sqrt(height**2 + width**2)
	#s is then the base of the logarithmic scale
	s = math.log10(max)/(columns+1)

	ind = fragment1
	#ind is a fragment
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
	return finalflat



img = cv2.imread('Asterix2.jpg')
height, width, channels = img.shape
#img = cv2.fastNlMeansDenoisingColored(img,None,20,20,7,21)
img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
edges = cv2.Canny(img,100,200)
kernel = np.ones((2,2),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 1)
im3, Ocontours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
c2 = []
forgotten = []
for cont in Ocontours:
  #print(len(cont))
  if len(cont)>10:
    c2.append(cont)
  elif len(cont)>4: 
  	forgotten.append(cont[:len(cont)//2])
contours = c2
#print(forgotten)
# top = [] 
# bot = []
# left = []
# right = []
# for x in range(0,width):
# 	top.append([x,0])
# 	bot.append([width-x,height])
# for y in range(0,height):
# 	right.append([width,y])
# 	left.append([0,height-y])
# outerContour = top + right + bot+ left
blank_image = np.zeros((height,width,3), np.uint8)


# im = cv2.imread("ex3.png")
# #im = cv2.imread("circle2.png")
# height, width, channels = im.shape
# imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
# height, width, channels = im.shape
# im3, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# blank_image = np.zeros((height,width,3), np.uint8)





# # startlistOfPoints = []
# # for ind, x in enumerate(contours[1]):
# # 	xval = x[0][0]
# # 	yval = x[0][1]
# # 	startlistOfPoints.append([yval, xval])
# # listOfPoints= np.array(startlistOfPoints)


# #blank_image = np.zeros(((height*4)//2,((width*4)//2),3), np.uint8)


outerContour = []
for ind, x in enumerate(contours[0]):
	xval = x[0][0]
	yval = x[0][1]
	outerContour.append([yval, xval])
# for x in outerContour:
# 	xval = x[1]
# 	yval = x[0]
# 	blank_image[yval][xval]=[0,255,255]


folderName = "flatfish"


frags = testFrags

for cont in range(1,len(contours)):
	print("onto ",cont," out of ",len(contours),len(contours[cont]))
	sys.stdout.flush()
	startlistOfPoints = []
	for ind, x in enumerate(contours[cont]):
		xval = x[0][0]
		yval = x[0][1]
		startlistOfPoints.append([yval, xval])
	listOfPoints= np.array(startlistOfPoints)


	steps = 10

	#maxAttempts = 1
	adjusted2 = listOfPoints
	fails = 0

	for x in range(0,steps):

		# selection = []
		# distances = []

		crossing = True
		while crossing:	
			if fails>0:
				fails = 0
				break
			#historic = adjusted2
			if(len(adjusted2)<10):
				break
			elif(len(adjusted2)<100):
				select = getRandomSection(len(adjusted2),len(adjusted2)//3, len(adjusted2)//6)
			else:
				select = getRandomSection(len(adjusted2),len(adjusted2)//10, len(adjusted2)//15)

			adjusted2 = adjustRoundList(adjusted2,math.floor(select[1]),math.floor(select[0]))

			sz  = 1
			if(select[0]> select[1]):
				sz = select[0]-select[1]
			else:
				sz = len(adjusted2) - select[1] + select[0]
			
			subfrag = adjusted2[math.floor((len(adjusted2)//3)):math.floor((len(adjusted2)//3))+math.floor(sz):1]


			subdesc = generateDescriptor(subfrag,height,width)

			myResult, myDistance = flann.nn_index(subdesc, 5, algorithm="kmeans", branching=32, iterations=7, checks=16)
			frg = frags[myResult[0][0]]

			relation = functionMisc.relatePoints( adjusted2[len(adjusted2)//3],frg[0])
			
			transdEdge =  functionMisc.rotateEdgeToEdge(adjusted2[len(adjusted2)//3],adjusted2[math.floor((len(adjusted2)//3))+math.floor(sz)], frg)

			transdEdge2 = functionMisc.ratioTransfer(adjusted2[math.floor((len(adjusted2)//3))+math.floor(sz)],transdEdge)
			if len(transdEdge2) == 0:
				fails = fails+1
				#print("lenfail length of frg ",len(frg), " sz ",sz)
				#break
			if fails<1:
			#crossing = functionMisc.ecross(outerContour,transdEdge2) or (functionMisc.ecross(adjusted2[0 : math.floor((len(adjusted2)//3))-1 : 1],transdEdge2) or functionMisc.ecross(adjusted2[math.floor((len(adjusted2)//3))+math.floor(sz)+1 : len(adjusted2) : 1], transdEdge2))
			#CAUSE WE NO LONGER CARE
				crossing = functionMisc.ecross(outerContour,transdEdge2)
				if not crossing:
					adjusted2 = np.concatenate((np.concatenate((adjusted2[0 : math.floor((len(adjusted2)//3)) : 1],transdEdge2)),adjusted2[math.floor((len(adjusted2)//3))+math.floor(sz) : len(adjusted2) : 1]))
	
	for x in adjusted2:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[255,255,255]

print("fails: ",fails)

for line in forgotten:
	for x in line:
		yval = x[0][1]
		xval = x[0][0]
		blank_image[yval][xval]=[255,255,255]




# cv2.imshow('Edges',blank_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.title('original Image'), plt.xticks([]), plt.yticks([])


blank_image2 = np.zeros((height,width,3), np.uint8)
cv2.drawContours(blank_image2, Ocontours, -1, (255,255,255), 1)
plt.subplot(132),plt.imshow(blank_image2,cmap = 'gray')
plt.title('contour Image'), plt.xticks([]), plt.yticks([])

plt.subplot(133),plt.imshow(blank_image,cmap = 'gray')
plt.title('peturbed Image'), plt.xticks([]), plt.yticks([])



plt.show()


def betterRotTest():
	rList = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]]
	rs = getRandomSection(len(rList),5,3)
	myRe = adjustRoundList(rList,math.floor(rs[1]),math.floor(rs[0]))
	#print(myRe,"   ..... ", myRe[len(myRe)//3])
#betterRotTest()

def testRotateAndSelect():
	rList = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	for x in range(1,10):
		rs = getRandomSection(len(rList),7,4)
		rLister = rList
		myRe = adjustRoundList(rLister,math.floor(rs[1]),math.floor(rs[0]))
		#print(myRe, myRe[len(myRe)//3])




###   CHECK THE RELEVANCE OF THE PREVIOUS FRAGMENT THAT YOURE REPLACING TO SEE IF WE CAN MAKE A requirement that it only improves