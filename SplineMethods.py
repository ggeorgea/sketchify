import numpy as np
import cv2
import math
from pyflann import *
import os
import functionMisc
import random
import connectFrags
import numpy as np
import cv2
import math
from pyflann import *
import os
import functionMisc
import matplotlib.pyplot as plt
import scipy.interpolate as inter
import pylab as plt

def AllRelGet():

	numberOfPoints = 0
	accPoint0 = 0
	accPoint1 = 0

	folderName = "flatfish"

	c = np.load("MPEG7dataset\\test\\extractions\\"+folderName+"\\metadata.npy")

	# just for dimensions
	im = cv2.imread('MPEG7dataset\\test\\apple\\apple-1.png')
	height, width, channels = im.shape

	#finding most relevant in cow using cat as vs
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
	testImBig = np.concatenate([apples,child,cellphone,face,chopper,watch,bell,bone,car,cellphone,fountain,keys,shoe,flatfish], axis = 0)
	chosen =flatfish
	flann = FLANN()
	flann.build_index(testImBig,algorithm="kmeans", branching=32, iterations=7, checks=16)


	#relavantest stores the fragment indicies with the highest relevance, and their indicies
	relavantest = []
	#looker is the bound on the length of relevantest
	looker = 80

	ImgNo = 0

	for imager in c:
		a = np.split(chosen ,c[ImgNo])[1]
		#num is the index of the descriptor des relative to the lowest index of the image 
		num = 0;
		for des in a:
			myResult, myDistance = flann.nn_index(des, 50, algorithm="kmeans", branching=32, iterations=7, checks=16)
			highestNonCow = len(testImBig)-len(chosen);
			inspectArray  =  []
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
			myLower = c[ImgNo][0]
			#tic is our current index in our insertion search of relavantest
			tic = 0
			inserted = False
			#just an ordered assertion into relavantest
			for pre in relavantest:
				if tic>looker:
					#if weve gone over the length of the thing
					#just unsyrence fir the deletion part later
					break 
				if pre[0]<numNearerfound:
					#insertion at tic
					relavantest.insert(tic,([numNearerfound,num+myLower]))
					inserted = True
					if len(relavantest)>looker:
						#to ensure we dont go over our specified length, we delete the last element
						del relavantest[len(relavantest)-1]
					break
				tic=tic+1
			if inserted==False and len(relavantest)<looker:
				#initial filling of relecantest
				relavantest.append(([numNearerfound,num+myLower]))
			num = num+1
		ImgNo = ImgNo+1		


	diffrelMax =40
	diffrel = []

	diffdescriptors = []
	for desc in relavantest:
		diffdescriptors.append(chosen[desc[1]])
	flann.build_index(np.array(diffdescriptors),algorithm="kmeans", branching=32, iterations=7, checks=16)
	it = 0
	for someDesc in diffdescriptors:
		myResult, myDistance = flann.nn_index(someDesc, 2, algorithm="kmeans", branching=32, iterations=7, checks=16)
		nextNear = myDistance[0][1]
		tic2 = 0
		inserted = False
		#just an ordered assertion into diffrel
		for pre in diffrel:
			if tic2>diffrelMax:
				#if weve gone over the length of the thing
				#just unsyrence fir the deletion part later
				break 
			if pre[0]<nextNear:
				#insertion at tic
				diffrel.insert(tic2,([nextNear,relavantest[it][0],relavantest[it][1]]))
				inserted = True
				if len(diffrel)>diffrelMax:
					#to ensure we dont go over our specified length, we delete the last element
					del diffrel[len(diffrel)-1]
				break
			tic2=tic2+1
		if inserted==False and len(diffrel)<diffrelMax:
			#initial filling of relecantest
			diffrel.append(([nextNear,relavantest[it][0],relavantest[it][1]]))
		it = it+1

	lastAr = []
	for thing in diffrel:
		lastAr.append([thing[1],thing[2]])
	relavantest = lastAr

	pointsToPol = []

	cowFrags = np.load('MPEG7dataset\\test\\extractions\\'+folderName+'\\Fragments.npy')
	
	for f in cowFrags:
		for p in f:
			numberOfPoints = numberOfPoints+1
			accPoint1 = accPoint1 + p[1]
			accPoint0 = accPoint0 + p[0]

	for top in relavantest:
		number = top[1]
		for x in cowFrags[number]:
			xval = x[1]
			yval = x[0]
			pointsToPol.append(x)	

	#alternative regularisation

	#fish with fewer points
	#fact = 4.5

	#fish with iterations
	fact = 10
	
	#fish without iterations
	#fact = 12

	#applespline
	#fact = 7
	#tryWithIT TODO

	#bellfail fact = 4

	#bone
	#fact = 17.34

	#fact = 3.73
	#car

	#fact = 11
	#cellphone

	#fact = 3.5
	#fountain

	#fact = 15.75
	#key

	#shoe
	#fact = 19

	pointCenter = [accPoint0//numberOfPoints,accPoint1//numberOfPoints]
	pointRelToCenter = []
	acc = 0
	acc2 = 0
	for p in pointsToPol:
		acc = acc+1
		if acc%5!=0:
			continue
		acc2 = acc2+1
		relation = functionMisc.relatePoints(p,pointCenter)
		x = relation[1]
		y = relation[0]
		
		pointRelToCenter.append(np.array([np.sqrt(x**2 + y**2)/fact,np.arctan2(y, x)+math.pi*2]))
		pointRelToCenter.append(np.array([np.sqrt(x**2 + y**2)/fact,np.arctan2(y, x)-math.pi*2]))
		pointRelToCenter.append(np.array([np.sqrt(x**2 + y**2)/fact,np.arctan2(y, x)]))

	polarCollect = np.array(pointRelToCenter)
	sorted  = polarCollect[polarCollect[:,1].argsort()]

	stX = sorted[:,1]
	print(stX)
	stY = sorted[:,0]

	print(sorted)
	print(acc2)

	s2 = inter.pchip(stX,stY)

	xs = np.arange(-3.14158*2.5,3.14159*2.5,0.01)

	xx = np.arange(-3.14158*2.5,3.14159*2.5,0.01)

	yUnNew = s2(xs)
	xNext = []
	YNext = []
	for x in range(0,len(yUnNew)):
		if x%(len(yUnNew)//500) == 0:
			xNext.append(xs[x])
			YNext.append(yUnNew[x])
	spl3 = inter.UnivariateSpline(xNext,YNext,s=650)
	xx5 = np.arange(-3.14158*2.5,3.14159*2.5,0.01)
	yy5 = spl3(xx)

	plt.plot (xNext, YNext, 'r-', label='pchip fish')

	plt.plot(xx5,yy5,'b',label = 'evened')

	splineCart = []
	valsToUse = np.arange(-(math.pi),math.pi,0.001)
	rsToUse = spl3(valsToUse)
	for ind in range(len(valsToUse)):
		rho = rsToUse[ind]*fact
		phi = valsToUse[ind]
		x = rho * np.cos(phi)
		y = rho * np.sin(phi)
		splineCart.append([ int(y),int(x)])

	for x in splineCart:
	#outContour:
		if x[0]+pointCenter[0]>0 and x[1]+pointCenter[1]>0 and x[0]+pointCenter[0]<640 and x[1]+pointCenter[1]<640 :
			xval = x[1]
			yval = x[0]

	slipped = []
	for x in splineCart:
			slipped.append([x[0]+pointCenter[0], x[1]+pointCenter[1]])
	
	minW1 = min(np.array(slipped)[:,1])
	minW0 = min(np.array(slipped)[:,0])
	if(minW1<0):
		slipped2 = []
		for x in slipped:
			slipped2.append([x[0],x[1]+ abs(minW1)+5])
		slipped = slipped2
	if (minW0<0):
		slipped2 = []
		for x in slipped:
			slipped2.append([x[0]+ abs(minW0)+5,x[1]])
		slipped = slipped2
	
	final = []
	for x in functionMisc.fill(functionMisc.thin(slipped+[slipped[0]])):
		if x[0]>0 and x[1]>0 and x[0]<640 and x[1]<640 :
			final.append(x)

	return final

#for getting stuff from round, could replace choice in createDescriptorSet also
#from 1 to 0
def getRandomSection(length, maxDist, minDist):
	result = []
	while True:
		result = [(random.random()+0.0)*length//1,(random.random()+0.0)*length//1]
		if (result[0]> result[1] and result[0]> result[1]+minDist and result[0]<result[1]+maxDist) or (result[0]<result[1] and (length - result[1] + result[0]) > minDist and (length - result[1] + result[0]) < maxDist  ):
			break
	return result

#des is some descriptor
def fullRel(des):
	myResult, myDistance = flann.nn_index(des, 50, algorithm="kmeans", branching=32, iterations=7, checks=16)
	highestNonCow = len(testImBig)-len(chosen);
	inspectArray  =  []
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


def adjustRoundList(Round:list, a, b):
	fullRound = Round
	length = len(fullRound)
	if a>b:
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
		while (a< length//3):
			fullRound = np.concatenate(( [fullRound[len(fullRound)-1]] , fullRound[0: len(fullRound)-1 :1] ))
			a = a+1
	if(a>length//3):
		while (a> length//3):
			fullRound = np.concatenate(( fullRound[1: len(fullRound) :1] , [fullRound[0]]  ))
			a= a-1

	return fullRound



def complete():
	truths = 0
	folderName = "flatfish"


	c = np.load("MPEG7dataset\\test\\extractions\\"+folderName+"\\metadata.npy")

	# just for dimensions
	im = cv2.imread('MPEG7dataset\\test\\apple\\apple-1.png')
	height, width, channels = im.shape
	blank_image = np.zeros(((height*5)//2,((width*5)//2),3), np.uint8)

	#finding most relevant in cow using cat as vs
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
	testImBig = np.concatenate([apples,child,carriage,face,chopper,watch,bell,bone,car,cellphone,fountain,keys,shoe,flatfish], axis = 0)
	chosen =flatfish
	flann = FLANN()
	flann.build_index(testImBig,algorithm="kmeans", branching=32, iterations=7, checks=16)






	#relavantest stores the fragment indicies with the highest relevance, and their indicies
	relavantest = []
	#looker is the bound on the length of relevantest
	looker = 30

	ImgNo = 0

	for imager in c:
		a = np.split(chosen ,c[ImgNo])[1]
		#num is the index of the descriptor des relative to the lowest index of the image 
		num = 0;
		for des in a:
			#print(len(a))
			myResult, myDistance = flann.nn_index(des, 50, algorithm="kmeans", branching=32, iterations=7, checks=16)
			highestNonCow = len(testImBig)-len(chosen);
			inspectArray  =  []
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
			myLower = c[ImgNo][0]

			tic = 0
			inserted = False
			#just an ordered assertion into relavantest
			for pre in relavantest:
				if tic>looker:
					#if weve gone over the length of the thing
					#just unsyrence fir the deletion part later
					break 
				if pre[0]<numNearerfound:
					#insertion at tic
					relavantest.insert(tic,([numNearerfound,num+myLower]))
					inserted = True
					if len(relavantest)>looker:
						#to ensure we dont go over our specified length, we delete the last element
						del relavantest[len(relavantest)-1]
					break
				tic=tic+1
			if inserted==False and len(relavantest)<looker:
				#initial filling of relecantest
				relavantest.append(([numNearerfound,num+myLower]))
			num = num+1
		ImgNo = ImgNo+1		
	fragList=[]

	cowFrags = np.load('MPEG7dataset\\test\\extractions\\'+folderName+'\\Fragments.npy')

	for top in relavantest:
		number = top[1]
		fragList.append(cowFrags[number])

#phase 2===============================================================================================================================================================================
#=================================================================================================================
#=================================================================================================================
	
	fullShape = AllRelGet()


	im = cv2.imread("exampleImage.png")
	height, width, channels = im.shape
	flann.build_index(testImBig,algorithm="kmeans", branching=32, iterations=7, checks=16)
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,127,255,0)
	height, width, channels = im.shape
	im3, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	blank_image = np.zeros((height,width,3), np.uint8)
	c = np.load("MPEG7dataset\\test\\extractions\\"+folderName+"\\metadata.npy")


	outerContour = []
	for ind, x in enumerate(contours[0]):
		xval = x[0][0]
		yval = x[0][1]
		outerContour.append([yval, xval])
	for x in outerContour:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[0,255,255]

	frags = fragList
	listOfPoints = fullShape

	steps = 0
	maxAttempts = 5
	adjusted2 = listOfPoints
	fails = 0

	for x in range(0,steps):
		selection = []
		distances = []
		print("\n")
		for x in range(0,maxAttempts):
			crossing = True
			while crossing:	
				if fails>8:
					fails = 0
					break
				historic = adjusted2
				select = getRandomSection(len(adjusted2),len(adjusted2)//12, len(adjusted2)//20)

				adjusted2 = adjustRoundList(adjusted2,math.floor(select[1]),math.floor(select[0]))
				adjustedRot = adjusted2
				#sz calculates from just select, could hapepn after the adjust round
				sz  = 1
				if(select[0]> select[1]):
					sz = select[0]-select[1]
				else:
					sz = len(adjusted2) - select[1] + select[0]
				#choose a random fragment
				index2Choose = math.floor((len(frags)*random.random())//1)
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
					targetLength = len(frg)
					rtdd = functionMisc.createDescriptorSet(adjusted2,height,width,targetLength,2.2)
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
						#the place in our index of the last noncow, the remainder of the index is cows //^^^
						inspectArray  =  []
	
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
									print("!")
									numNearerfound = numNearerfound+1

						if numNearerfound == 0:
							for myR in range(len(myResult[0])):
								test = myResult[0][myR]- highestNonCow
								if test>0:
									inspectArray = np.split(myResult[0],[0,myR])[1]
									break
								else: pass
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
										#print("!")
										numNearerfound = numNearerfound-1

						totalSum = totalSum-numNearerfound

						#==============

					selection.append(adjusted2)


					rtdd = functionMisc.createDescriptorSet(adjustedRot,height,width,targetLength,2.2)
					totalSumOld = 0
					for descr in rtdd:				
						myResult, myDistance = flann.nn_index(descr, 50, algorithm="kmeans", branching=32, iterations=7, checks=16)
						highestNonCow = len(testImBig)-len(chosen);
						inspectArray  =  []

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
					

						if numNearerfound == 0:
							for myR in range(len(myResult[0])):
								test = myResult[0][myR]- highestNonCow
								if test>0:
									inspectArray = np.split(myResult[0],[0,myR])[1]
									break
								else: pass
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
										#print("!")
										numNearerfound = numNearerfound-1

						totalSumOld = totalSumOld-numNearerfound

					adjusted2 = historic
					print('NewNearer:',-totalSum," OldNearer:",-totalSumOld)
					distances = distances + [totalSum-totalSumOld]

		indexI = 0

		best = distances[0]
		for distInd in range(0,len(distances)):
			if(distances[distInd]<best):
				indexI = distInd
				best = distances[distInd]
		if(best<0):
			adjusted2 = selection[indexI]
			print("tic")



	print("fails: ",fails)

	#================================================================

	for x in fullShape:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[255,0,0]

	for x in adjusted2:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[255,255,255]
	cv2.imshow('Edges',blank_image)

	print("ok finish")
	print("truth",truths)
	plt.minorticks_on()
	plt.legend()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()




def completeSpline(folderName,testImBig,chosen):

	numberOfPoints = 0
	accPoint0 = 0
	accPoint1 = 0

	c = np.load("MPEG7dataset\\test\\extractions\\"+folderName+"\\metadata.npy")

	# just for dimensions
	im = cv2.imread('MPEG7dataset\\test\\apple\\apple-1.png')
	height, width, channels = im.shape

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
	flann = FLANN()
	flann.build_index(testImBig,algorithm="kmeans", branching=32, iterations=7, checks=16)


	#relavantest stores the fragment indicies with the highest relevance, and their indicies
	relavantest = []
	#looker is the bound on the length of relevantest
	looker = 80

	ImgNo = 0

	for imager in c:
		a = np.split(chosen ,c[ImgNo])[1]
		#num is the index of the descriptor des relative to the lowest index of the image 
		num = 0;
		for des in a:
			myResult, myDistance = flann.nn_index(des, 50, algorithm="kmeans", branching=32, iterations=7, checks=16)
			highestNonCow = len(testImBig)-len(chosen);
			inspectArray  =  []
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
			myLower = c[ImgNo][0]
			#tic is our current index in our insertion search of relavantest
			tic = 0
			inserted = False
			#just an ordered assertion into relavantest
			for pre in relavantest:
				if tic>looker:
					#if weve gone over the length of the thing
					#just unsyrence fir the deletion part later
					break 
				if pre[0]<numNearerfound:
					#insertion at tic
					relavantest.insert(tic,([numNearerfound,num+myLower]))
					inserted = True
					if len(relavantest)>looker:
						#to ensure we dont go over our specified length, we delete the last element
						del relavantest[len(relavantest)-1]
					break
				tic=tic+1
			if inserted==False and len(relavantest)<looker:
				#initial filling of relecantest
				relavantest.append(([numNearerfound,num+myLower]))
			num = num+1
		ImgNo = ImgNo+1		


	diffrelMax =40
	diffrel = []

	diffdescriptors = []
	for desc in relavantest:
		diffdescriptors.append(chosen[desc[1]])
	flann.build_index(np.array(diffdescriptors),algorithm="kmeans", branching=32, iterations=7, checks=16)
	it = 0
	for someDesc in diffdescriptors:
		myResult, myDistance = flann.nn_index(someDesc, 2, algorithm="kmeans", branching=32, iterations=7, checks=16)
		nextNear = myDistance[0][1]
		tic2 = 0
		inserted = False
		#just an ordered assertion into diffrel
		for pre in diffrel:
			if tic2>diffrelMax:
				#if weve gone over the length of the thing
				#just unsyrence fir the deletion part later
				break 
			if pre[0]<nextNear:
				#insertion at tic
				diffrel.insert(tic2,([nextNear,relavantest[it][0],relavantest[it][1]]))
				inserted = True
				if len(diffrel)>diffrelMax:
					#to ensure we dont go over our specified length, we delete the last element
					del diffrel[len(diffrel)-1]
				break
			tic2=tic2+1
		if inserted==False and len(diffrel)<diffrelMax:
			#initial filling of relecantest
			diffrel.append(([nextNear,relavantest[it][0],relavantest[it][1]]))
		it = it+1

	lastAr = []
	for thing in diffrel:
		lastAr.append([thing[1],thing[2]])
	relavantest = lastAr

	pointsToPol = []

	cowFrags = np.load('MPEG7dataset\\test\\extractions\\'+folderName+'\\Fragments.npy')


	for f in cowFrags:
		for p in f:
			numberOfPoints = numberOfPoints+1
			accPoint1 = accPoint1 + p[1]
			accPoint0 = accPoint0 + p[0]

	for top in relavantest:
		number = top[1]
		for x in cowFrags[number]:
			xval = x[1]
			yval = x[0]
			pointsToPol.append(x)	



	#fact = 25


	#fish with fewer points
	#fact = 4.5

	#fish with iterations
	#fact = 10
	
	#fish without iterations
	#fact = 12

	#applespline
	#fact = 7
	#tryWithIT TODO

	#bellfail fact = 4

	#bone
	#fact = 17.34

	#fact = 3.73
	#car

	#fact = 11
	#cellphone

	#fact = 3.5
	#fountain

	#fact = 15.75
	#key

	#shoe
	fact = 19

	pointCenter = [accPoint0//numberOfPoints,accPoint1//numberOfPoints]
	pointRelToCenter = []
	acc = 0
	acc2 = 0
	for p in pointsToPol:
		acc = acc+1
		if acc%5!=0:
			continue
		acc2 = acc2+1
		relation = functionMisc.relatePoints(p,pointCenter)
		x = relation[1]
		y = relation[0]
		
		pointRelToCenter.append(np.array([np.sqrt(x**2 + y**2)/fact,np.arctan2(y, x)+math.pi*2]))
		pointRelToCenter.append(np.array([np.sqrt(x**2 + y**2)/fact,np.arctan2(y, x)-math.pi*2]))
		pointRelToCenter.append(np.array([np.sqrt(x**2 + y**2)/fact,np.arctan2(y, x)]))

	polarCollect = np.array(pointRelToCenter)
	sorted  = polarCollect[polarCollect[:,1].argsort()]

	stX = sorted[:,1]
	stY = sorted[:,0]

	s2 = inter.pchip(stX,stY)

	xs = np.arange(-3.14158*2.5,3.14159*2.5,0.01)

	xx = np.arange(-3.14158*2.5,3.14159*2.5,0.01)


	yUnNew = s2(xs)
	xNext = []
	YNext = []
	for x in range(0,len(yUnNew)):
		if x%(len(yUnNew)//500) == 0:
			xNext.append(xs[x])
			YNext.append(yUnNew[x])
	spl3 = inter.UnivariateSpline(xNext,YNext,s=650)
	xx5 = np.arange(-3.14158*2.5,3.14159*2.5,0.01)
	yy5 = spl3(xx)


	plt.plot (xNext, YNext, 'r-', label='pchip fish')

	plt.plot(xx5,yy5,'b',label = 'evened')




	splineCart = []
	valsToUse = np.arange(-(math.pi),math.pi,0.001)
	rsToUse = spl3(valsToUse)
	for ind in range(len(valsToUse)):
		rho = rsToUse[ind]*fact
		phi = valsToUse[ind]
		x = rho * np.cos(phi)
		y = rho * np.sin(phi)
		splineCart.append([ int(y),int(x)])
	for x in splineCart:
	#outContour:
		if x[0]+pointCenter[0]>0 and x[1]+pointCenter[1]>0 and x[0]+pointCenter[0]<640 and x[1]+pointCenter[1]<640 :
			xval = x[1]
			yval = x[0]




	slipped = []
	for x in splineCart:
			slipped.append([x[0]+pointCenter[0], x[1]+pointCenter[1]])
	
	minW1 = min(np.array(slipped)[:,1])
	minW0 = min(np.array(slipped)[:,0])
	if(minW1<0):
		slipped2 = []
		for x in slipped:
			slipped2.append([x[0],x[1]+ abs(minW1)+5])
		slipped = slipped2
	if (minW0<0):
		slipped2 = []
		for x in slipped:
			slipped2.append([x[0]+ abs(minW0)+5,x[1]])
		slipped = slipped2
	
	final = []
	for x in functionMisc.fill(functionMisc.thin(slipped+[slipped[0]])):
		if x[0]>0 and x[1]>0 and x[0]<640 and x[1]<640 :
			final.append(x)
	return final

def getSpline(folderName,testImBig,chosen):
	return completeSpline(folderName,testImBig,chosen)