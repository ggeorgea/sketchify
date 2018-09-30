import numpy as np
import cv2
import math
from pyflann import *
import os
import random
import functionMisc
flann = FLANN()

#for getting stuff from round, could replace choice in createDescriptorSet also
#from 1 to 0
def getRandomSection(length, maxDist, minDist):
	result = []
	while True:
		result = [(random.random()+0.0)*length//1,(random.random()+0.0)*length//1]
		if (result[0]> result[1] and result[0]> result[1]+minDist and result[0]<result[1]+maxDist) or (result[0]<result[1] and (length - result[1] + result[0]) > minDist and (length - result[1] + result[0]) < maxDist  ):
			break
	return result







folderName = "flatfish"
c = np.load("MPEG7dataset\\test\\extractions\\"+folderName+"\\metadata.npy")

apples = np.load('MPEG7dataset\\test\\extractions\\apple\\Descriptors.npy')
bell= np.load('MPEG7dataset\\test\\extractions\\bell\\Descriptors.npy')
bone= np.load('MPEG7dataset\\test\\extractions\\Bone\\Descriptors.npy')
car= np.load('MPEG7dataset\\test\\extractions\\car\\Descriptors.npy')
carriage= np.load('MPEG7dataset\\test\\extractions\\carriage\\Descriptors.npy')
cellphone= np.load('MPEG7dataset\\test\\extractions\\cellular_phone\\Descriptors.npy')
child= np.load('MPEG7dataset\\test\\extractions\\children\\Descriptors.npy')
chopper= np.load('MPEG7dataset\\test\\extractions\\chopper\\Descriptors.npy')
face= np.load('MPEG7dataset\\test\\extractions\\face\\Descriptors.npy')
flatfish= np.load('MPEG7dataset\\test\\extractions\\flatfish\\Descriptors.npy')
fountain= np.load('MPEG7dataset\\test\\extractions\\fountain\\Descriptors.npy')
keys= np.load('MPEG7dataset\\test\\extractions\\key\\Descriptors.npy')
shoe= np.load('MPEG7dataset\\test\\extractions\\shoe\\Descriptors.npy')
watch= np.load('MPEG7dataset\\test\\extractions\\watch\\Descriptors.npy')
testImBig = np.concatenate([bone,car,child,carriage,cellphone,face,keys,bell,apples,shoe,fountain,cellphone,chopper,watch,flatfish], axis = 0)
chosen =flatfish
flann.build_index(testImBig,algorithm="kmeans", branching=32, iterations=7, checks=16)

#des is some descriptor
#DO NOT CALL
def fullRel(des):
	myResult, myDistance = flann.nn_index(des, 50, algorithm="kmeans", branching=32, iterations=7, checks=16)
	highestNonCow = len(testImBig)-len(chosen);
	inspectArray  =  []
	#print(myResult)
	#print(highestNonCow)
	for myR in range(len(myResult[0])):
		test = myResult[0][myR]- highestNonCow
		if test<0:
			inspectArray = np.split(myResult[0],[0,myR])[1]
			break
		else: pass
	numNearerfound = 0
	if len(inspectArray)>1:
		dict = {'init':'fake'}
		for cand in inspectArray:
			imgPlace = cand - (highestNonCow +1)
			imgID=0
			for imageBound in c:
				if (imageBound[1]>imgPlace) and (imageBound[0]<imgPlace):
					imgID = imageBound[0]
					break
			if imgID in dict :
				pass
			else:
				dict[imgID]=1
				numNearerfound = numNearerfound+1
	#NUM NEARER FOUND IS RELEVANCE METRIC
	#print(numNearerfound," ",len(inspectArray))
	#numNearerfound=len(inspectArray) 





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

im = cv2.imread("exampleImage.png")
#im = cv2.imread("circle2.png")
height, width, channels = im.shape
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
height, width, channels = im.shape
im3, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
blank_image = np.zeros((height,width,3), np.uint8)
startlistOfPoints = []
for ind, x in enumerate(contours[1]):
	xval = x[0][0]
	yval = x[0][1]
	startlistOfPoints.append([yval, xval])
listOfPoints= np.array(startlistOfPoints)
blank_image = np.zeros(((height*5)//2,((width*5)//2),3), np.uint8)


outerContour = []
for ind, x in enumerate(contours[0]):
	xval = x[0][0]
	yval = x[0][1]
	outerContour.append([yval, xval])
for x in outerContour:
	xval = x[1]
	yval = x[0]
	blank_image[yval][xval]=[0,255,255]


folderName = "flatfish"


#====replaced with full descriptor?
#NOT USED ANYWAYmetadata = np.load("MPEG7dataset\\test\\extractions\\"+folderName+"\\metadata.npy")
# im = cv2.imread('MPEG7dataset\\test\\flatfish\\flatfish-1.png')
# height, width, channels = im.shape
# blank_image = np.zeros(((height*5)//2,((width*5)//2),3), np.uint8)
#====
#descriptors is the thing replaced
#descrips = np.load('MPEG7dataset\\test\\extractions\\'+folderName+'\\Descriptors.npy')
#===
#keep frags cause frags
frags = np.load('MPEG7dataset\\test\\extractions\\'+folderName+'\\Fragments.npy')
#wrong index now
#flann.build_index(np.array(descrips),algorithm="kmeans", branching=32, iterations=7, checks=16)
#=====

#improvments = 0
#desiredImprovements = 5
steps = 20
#TODO ADD PRIOR ROTATION!!!
maxAttempts = 7
adjusted2 = listOfPoints
fails = 0

for x in range(0,steps):

	selection = []
	distances = []
	for x in range(0,maxAttempts):
		crossing = True
		while crossing:	
			if fails>8:
				fails = 0
				break
			historic = adjusted2
			select = getRandomSection(len(adjusted2),len(adjusted2)//15, len(adjusted2)//20)

			adjusted2 = adjustRoundList(adjusted2,math.floor(select[1]),math.floor(select[0]))
			#sz calculates from just select, could hapepn after the adjust round
			sz  = 1
			if(select[0]> select[1]):
				sz = select[0]-select[1]
			else:
				sz = len(adjusted2) - select[1] + select[0]
			#choose a random fragment
			index2Choose = math.floor((len(frags)*random.random())//1)
			#NOT USED ANYWAYdstr = descrips[index2Choose]
			frg = frags[index2Choose]
			relation = functionMisc.relatePoints( adjusted2[len(adjusted2)//3],frg[0])
			
			transdEdge =  functionMisc.rotateEdgeToEdge(adjusted2[len(adjusted2)//3],adjusted2[math.floor((len(adjusted2)//3))+math.floor(sz)], frg)

			transdEdge2 = functionMisc.ratioTransfer(adjusted2[math.floor((len(adjusted2)//3))+math.floor(sz)],transdEdge)
			if len(transdEdge2) == 0:
				fails = fails+1
				continue
			crossing = functionMisc.ecross(outerContour,transdEdge2) or (functionMisc.ecross(adjusted2[0 : math.floor((len(adjusted2)//3))-1 : 1],transdEdge2) or functionMisc.ecross(adjusted2[math.floor((len(adjusted2)//3))+math.floor(sz)+1 : len(adjusted2) : 1], transdEdge2))
			if not crossing:
				
				adjusted2 = np.concatenate((np.concatenate((adjusted2[0 : math.floor((len(adjusted2)//3)) : 1],transdEdge2)),adjusted2[math.floor((len(adjusted2)//3))+math.floor(sz) : len(adjusted2) : 1]))
				rtdd = functionMisc.createDescriptorSet(adjusted2,height,width)
				totalSum = 0
				for descr in rtdd:				
					#======= replaced with full descriptor
					# myResult, myDistance = flann.nn_index(descr, 5, algorithm="kmeans", branching=32, iterations=7, checks=16)
					# intersum = 0
					# for indist in myDistance[0]:
					# 	intersum = intersum + indist
					# totalSum = totalSum + intersum		
					#============
					#full relevance \/\/\/
					#=============

					myResult, myDistance = flann.nn_index(descr, 50, algorithm="kmeans", branching=32, iterations=7, checks=16)
					highestNonCow = len(testImBig)-len(chosen);
					inspectArray  =  []
					#print(myResult)
					#print(highestNonCow)
					for myR in range(len(myResult[0])):
						test = myResult[0][myR]- highestNonCow
						if test<0:
							inspectArray = np.split(myResult[0],[0,myR])[1]
							break
						else: pass
					numNearerfound = 0
					if len(inspectArray)>1:
						dict = {'init':'fake'}
						for cand in inspectArray:
							imgPlace = cand - (highestNonCow +1)
							imgID=0
							for imageBound in c:
								if (imageBound[1]>imgPlace) and (imageBound[0]<imgPlace):
									imgID = imageBound[0]
									break
							if imgID in dict :
								pass
							else:
								dict[imgID]=1
								numNearerfound = numNearerfound+1
					totalSum = totalSum-numNearerfound

					#==============

				selection.append(adjusted2)
				distances = distances + [totalSum]
				adjusted2 = historic

	indexI = 0
	#=======     for full descriptor i guess < just needs to become >   =======#
	best = distances[0]
	for distInd in range(0,len(distances)):
		if(distances[distInd]<best):
			indexI = distInd
			best = distances[distInd]
	adjusted2 = selection[indexI]

for x in adjusted2:
	xval = x[1]
	yval = x[0]
	blank_image[yval][xval]=[255,255,255]

print("fails: ",fails)
cv2.imshow('Edges',blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




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