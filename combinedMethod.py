import numpy as np
import cv2
import math
from pyflann import *
import os
import functionMisc
import random
import connectFrags
import SplineMethods
import datetime
from scipy.spatial import distance

#this version has two-way comparison of location see=
#	return goCompareInd(myShape, splineShape, height, width, step) + goCompareInd(splineShape, myShape, height, width, step)
#this version assess length as a metric to prioritise choice see=
#				if (targetSize-newSize)**2 < (targetSize-currentsize)**2   or ( ((targetSize-newSize)**2)*0.9 < (targetSize-currentsize)**2  and     (targetLength - newLength)**2 < (targetLength - currentlength)**2  ):
#					bigger2= True
#adds an extra based on line length each time


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
	#NUM NEARER FOUND IS RELEVANCE METRIC


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

def goCompareInd(myShape, splineShape, height, width, step):
	totalDist = 0
	pointI  = 0
	while pointI<len(splineShape):
		point = splineShape[pointI]
		lowestDist = 1000000
		possI = 0
		while(possI<len(myShape)):
			possNN = myShape[possI]
			if functionMisc.distance([point, possNN]) < lowestDist:
				lowestDist = functionMisc.distance([point,possNN])
			possI = possI+step
		totalDist = totalDist+lowestDist
		pointI = pointI+step
	return totalDist

def goCompare2(myShape, splineShape, height, width, step):
	return goCompareInd(myShape, splineShape, height, width, step) + goCompareInd(splineShape, myShape, height, width, step)

def goCompare(shape1, shape2, height, width):
	array1 = np.array(shape1)
	array2 = np.array(shape2)
	#findplace1
	a1 = []
	for p in array1:
		a1.append([p[0],p[1]])
	array1 = np.array(a1)
	m1 = np.median(array1[:,0])

	foundindicies = 0
	foundval = 100000000
	ind = 0
	for x in array1:
		if x[0]-m1<1 and x[1] < foundval:
			foundindicies = ind
			foundval  = x[1]
		ind = ind +1
	sha1 = adjustRoundList(shape1, foundindicies, foundindicies + 5)
	#findplace2
	a2 = []
	for p in array2:
		a2.append([p[0],p[1]])
	array2 = np.array(a2)

	m2 = np.median(array2[:,0])
	foundindicies = 0
	foundval = 100000000
	ind = 0
	for x in array2:
		if x[0] - m2 < 1 and x[1] < foundval:
			foundindicies = ind
			foundval  = x[1]
		ind = ind +1
	sha2 = adjustRoundList(shape2, foundindicies, foundindicies + 5)

	thing1 = functionMisc.generateDescriptor(sha1,height,width, local = 12)
	thing2 = functionMisc.generateDescriptor(sha2,height,width, local = 12)
	totalDistance = distance.euclidean(thing1,thing2)
	return totalDistance



def addToQuest(objectFromList, meanSize,fragList,meanLength):
	soFar =  objectFromList[0][:]  
	shapeFragOrd = objectFromList[1][:]  
	linksOrd = objectFromList[2][:]  
	fullShape = objectFromList[3][:]  
	sizeScore = objectFromList[4]
	addedScore = objectFromList[5] 
	finished = objectFromList[6]
	frag2Ind = objectFromList[7]
	length1  = objectFromList[8]

	soFar2 =  objectFromList[0][:]  
	shapeFragOrd2 = objectFromList[1][:]  
	linksOrd2 = objectFromList[2][:]  
	fullShape2 = objectFromList[3][:]  
	sizeScore2 = objectFromList[4]
	addedScore2 = objectFromList[5] 
	finished2 = objectFromList[6]
	frag2Ind2 = objectFromList[7]
	length2 = objectFromList[8]

	nextFragind = frag2Ind+1
	while nextFragind<len(fragList):
		if nextFragind not in soFar:
			soFar.append(nextFragind)
			nextFrag = fragList[nextFragind]
			resulltsOfAdd = connectFrags.addFragment(shapeFragOrd,linksOrd,nextFrag)
			if resulltsOfAdd[0]:				
				currentsize = cv2.contourArea(np.array(fullShape))
				currentlength = len(np.array(fullShape))
				targetSize = meanSize
				targetLength =  meanLength
				newSize = cv2.contourArea(np.array(resulltsOfAdd[3]))
				newLength = len(np.array(np.array(resulltsOfAdd[3])))
				bigger2 = False
				if ( ((targetSize-newSize)**2)*0.9 < (targetSize-currentsize)**2  and     (targetLength - newLength)**2 < (targetLength - currentlength)**2  ):
					print("2")
					bigger2= True
				elif ( (targetSize-newSize)**2 < (targetSize-currentsize)**2  and     ((targetLength - newLength)**2)*0.9 < (targetLength - currentlength)**2  ):
					print("3")
					bigger2= True

				if bigger2 :
					shapeFragOrd = resulltsOfAdd[1]
					linksOrd = resulltsOfAdd[2]
					fullShape = resulltsOfAdd[3]
					addedScore = addedScore +1
					sizeScore = (targetSize-newSize)**2 
					nextFragind = nextFragind+1
					nextLeng = (targetLength - newLength)**2
					return [[soFar,shapeFragOrd,linksOrd,fullShape,sizeScore,addedScore,finished,nextFragind,nextLeng]        , [soFar,shapeFragOrd2,linksOrd2,fullShape2,sizeScore2,addedScore2,finished2,frag2Ind2,length2] ]
		nextFragind = nextFragind+1
	return [[soFar,shapeFragOrd2,linksOrd2,fullShape2,sizeScore2,addedScore2,True,frag2Ind2,length2]]


def AllRelGet():
	indexe = 0
	while indexe<10:
		indexe = indexe+1



		valueord = 0.3
		valuesiz = 40
		valuedist = 0.3
		maxInitFrontierSize = 100
		workingFrontierSize = 300

		heur1Max =  18
		heur2Max =12


		diffNumber = 30
		relvNumber = 70

		#maxAttempts
		maxAA = 17
		extras = 0

		truths = 0
		fullStage1Result = []
		folderName = "shoe"


		c = np.load("MPEG7dataset\\test\\extractions\\"+folderName+"\\metadata.npy")
		tgSizes = np.load("MPEG7dataset\\test\\extractions\\"+folderName+"\\sizes.npy")
		tgLengths = np.load("MPEG7dataset\\test\\extractions\\"+folderName+"\\lengths.npy")
		meanSize  = np.mean(tgSizes)
		meanLength  = np.mean(tgLengths)

		# just for dimensions
		im = cv2.imread('MPEG7dataset\\test\\apple\\apple-1.png')
		height, width, channels = im.shape
		blank_image = np.zeros(((height*5)//2,((width*5)//2),3), np.uint8)
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
		testImBig = np.concatenate([bone,car,child,carriage,face,bell,chopper,apples,fountain,keys,cellphone,watch,flatfish,shoe], axis = 0)
		chosen =shoe
		flann = FLANN()
		flann.build_index(testImBig,algorithm="kmeans", branching=32, iterations=7, checks=16)
		

		generalShape = SplineMethods.getSpline(folderName,testImBig,chosen)

		#relavantest stores the fragment indicies with the highest relevance, and their indicies
		relavantest = []
		#looker is the bound on the length of relevantest
		looker = relvNumber
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


		diffrelMax =  diffNumber
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

		print(relavantest)
		print("")
		print(diffrel)

		lastAr = []
		for thing in diffrel:
			lastAr.append([thing[1],thing[2]])
		relavantest = lastAr

		fragList=[]

		cowFrags = np.load('MPEG7dataset\\test\\extractions\\'+folderName+'\\Fragments.npy')
		print("\n\n")
		for top in relavantest:
			number = top[1]
			fragList.append(cowFrags[number])
			
		#DISTANCE MAXIMISING SORT ON RELEVANTEST=====================================
		origFragDist = []
		for fragD in fragList:
			tot = 0
			for fragTo in fragList:
				tot = tot + functionMisc.distance( [ fragD[len(fragD)//2],fragTo[len(fragTo)//2] ])
			origFragDist.append(tot)

		betterList = []
		fragDistInsList = []
		frInd = 0
		for fragIns in fragList:
			inserted = False
			insInd = 0
			for place in betterList:
				if origFragDist[frInd]>fragDistInsList[insInd]:
					betterList.insert(insInd,fragIns)
					fragDistInsList.insert(insInd,origFragDist[frInd])
					break
				insInd = insInd +1
			if not inserted:
				betterList.append(fragIns)
				fragDistInsList.append(origFragDist[frInd])
			frInd = frInd+1
		fragList = betterList



		###============================ code for putting stuff together, using things in relaventest
		frag1 = fragList[0]
		frag2 = fragList[1]
		frag1Ind = 0
		frag2Ind = 1
		fullShape = []
		adding = True

		objectLists = []
		while adding:

			startWork =  functionMisc.ecross(frag1,frag2)
			
			while startWork:
				print("X")
				frag2Ind = frag2Ind+1
				if(frag2Ind>=len(fragList)):
					frag1Ind = frag1Ind+1
					frag2Ind = frag1Ind+1
					if(frag2Ind>=len(fragList)):
						print("failup")
						return "failup"
						print("failup")
					frag1=fragList[frag1Ind]
					frag2=fragList[frag2Ind]
				startWork =  functionMisc.ecross(frag1,frag2)
			

			print(frag1Ind,frag2Ind,"startWork:",startWork)
			a=frag1[0]
			b=frag1[len(frag1)-1]
			c=frag2[0]
			d=frag2[len(frag2)-1]
			aToC = functionMisc.distance([a,c])
			aToD = functionMisc.distance([a,d])
			bToC = functionMisc.distance([b,c])
			bToD = functionMisc.distance([b,d])
			FirstFail = False
			SecondFail = False

			#forLater
			shapeFragOrd =[]
			linksOrd = []


			if True:
			#(aToC<bToC and aToC<aToD) or (bToD<bToC and bToD<aToD):
				#aToC and bToD)
				extra1 = functionMisc.fill([c,b])
				extra1 = extra1[1:len(extra1)-1:1]
				extra2 = functionMisc.fill([a,d])
				extra2 = extra2[1:len(extra2)-1:1]
				fullShape = frag1 + extra1[::-1] + frag2 + extra2[::-1]
				shapeFragOrd =[frag1,frag2]
				linksOrd = [extra1[::-1], extra2[::-1]]
				FirstFail =  not ( (not functionMisc.ecross(frag1,extra1))and (not functionMisc.ecross(frag1,extra2))and  (not functionMisc.ecross(frag2,extra1))and (not functionMisc.ecross(frag2,extra2)) )

			if FirstFail:
				#aToD and bTOC
				extra1 = functionMisc.fill([c,a])
				extra1 = extra1[1:len(extra1)-1:1]
				extra2 = functionMisc.fill([b,d])
				extra2 = extra2[1:len(extra2)-1:1]
				fullShape =  frag1 + extra2 + frag2[::-1] + extra1
				shapeFragOrd =[frag1,frag2[::-1]]
				linksOrd = [extra2, extra1]
				print("2222",functionMisc.ecross(extra1,extra2))
				print((functionMisc.ecross(frag1,extra1)), (functionMisc.ecross(frag1,extra2)), (functionMisc.ecross(frag2,extra1)), (functionMisc.ecross(frag2,extra2)) )
				SecondFail =  not ( (not functionMisc.ecross(frag1,extra1))and (not functionMisc.ecross(frag1,extra2))and  (not functionMisc.ecross(frag2,extra1))and (not functionMisc.ecross(frag2,extra2)) )
			print("firstfail,secondfail",FirstFail,SecondFail)
			if not SecondFail:
	#SUPPOSEDLY BETTER ALGO=============================================
				soFar = list(range(max(frag1Ind,frag2Ind)))
				sizeScore =  (meanSize  - cv2.contourArea(np.array(fullShape)))**2
				addedScore = 0
				newLength = len(np.array(fullShape))
				lengthScore = (meanLength - newLength)**2

				finished = False
				objectLists.append([soFar,shapeFragOrd,linksOrd,fullShape,sizeScore, addedScore, finished,frag2Ind,lengthScore])
				
			startWork = True

			if(len(objectLists)>maxInitFrontierSize):
				startWork = False
				adding = False

			while startWork:
				print("y")
				frag2Ind = frag2Ind+1
				if(frag2Ind>=len(fragList)):
					frag1Ind = frag1Ind+1
					frag2Ind = frag1Ind+1
					if(frag2Ind>=len(fragList)):
						adding = False
						break
					else:
						frag1=fragList[frag1Ind]
						frag2=fragList[frag2Ind]
						startWork =  functionMisc.ecross(frag1,frag2)
				else:
					startWork =  functionMisc.ecross(frag1,frag2)

		finalLength = len(objectLists)

		searching = True
		counter = 0
		while searching:
			counter = counter+1
			if len(objectLists)>workingFrontierSize:
				#delete worst size 
				worstind = 0
				worstScore = objectLists[0][4]
				cind = 1
				while cind < len(objectLists):
					if objectLists[cind][4]>worstScore:
						worstind = cind
						worstScore = objectLists[cind][4]
					cind = cind +1
				del objectLists[worstind]
				#delete worst size 
				worstind = 0
				worstScore = objectLists[0][4]
				cind = 1
				while cind < len(objectLists):
					if objectLists[cind][4]>worstScore:
						worstind = cind
						worstScore = objectLists[cind][4]
					cind = cind +1
				del objectLists[worstind]
				#and worst added
				worstind = 0
				worstScore = objectLists[0][5]
				cind = 1
				while cind < len(objectLists):
					if objectLists[cind][5]<worstScore:
						worstind = cind
						worstScore = objectLists[cind][5]
					cind = cind +1
				del objectLists[worstind]
				#extra doesnt matter cause thresholf, just need to limit the above
				# #delete worst size 
				worstind = 0
				worstScore = objectLists[0][8]
				cind = 1
				while cind < len(objectLists):
					if objectLists[cind][8]>worstScore:
						worstind = cind
						worstScore = objectLists[cind][8]
					cind = cind +1
				del objectLists[worstind]
				# #delete worst size 
				worstind = 0
				worstScore = objectLists[0][8]
				cind = 1
				while cind < len(objectLists):
					if objectLists[cind][8]>worstScore:
						worstind = cind
						worstScore = objectLists[cind][8]
					cind = cind +1
				del objectLists[worstind]
				# #delete worst size 
				worstind = 0
				worstScore = objectLists[0][8]
				cind = 1
				while cind < len(objectLists):
					if objectLists[cind][8]>worstScore:
						worstind = cind
						worstScore = objectLists[cind][8]
					cind = cind +1
				del objectLists[worstind]
			allFinished = False
			#add new best unfinished +1, keep old but with more tested
			besti = 0
			bests = 100000000000000000
			cind = 0

			while cind < len(objectLists):
				if objectLists[cind][4]<bests and objectLists[cind][6]==False:
					besti = cind
					bests = objectLists[cind][4]
				cind = cind +1
			if objectLists[besti][6] == True:
				allFinished= True
			else:
				resultion = addToQuest(objectLists[besti],meanSize,fragList,meanLength)
				if len(resultion) == 1:
					#check if its best or del
					betterExists = False
					for this in objectLists:
						if this[5]>resultion[0][5] or (this[5]==resultion[0][5] and this[4]<resultion[0][4]):
							better = True
					if betterExists:
						del(objectLists[besti])
					else:
						objectLists[besti]= resultion[0]
				else:
					objectLists[besti] = resultion[1]
					objectLists.append(resultion[0])

				besti = 0
				bests = -1
				cind = 0
				while cind < len(objectLists):
					if objectLists[cind][5]>bests and objectLists[cind][6]==False:
						besti = cind
						bests = objectLists[cind][5]
					cind = cind +1
				if objectLists[besti][6] == True:
					allFinished= True
				else:
					resultion = addToQuest(objectLists[besti],meanSize,fragList,meanLength)
					if len(resultion) == 1:
						#check if best or del 
						betterExists = False
						for this in objectLists:
							if this[5]>resultion[0][5] or (this[5]==resultion[0][5] and this[4]<resultion[0][4]):
								better = True
						if betterExists:
							del(objectLists[besti])
						else:
							objectLists[besti]= resultion[0]
					else:
						objectLists[besti] = resultion[1]
						objectLists.append(resultion[0])

					besti = 0
					bests = -1
					cind = 0
					while cind < len(objectLists):
						if objectLists[cind][8]>bests and objectLists[cind][6]==False:
							besti = cind
							bests = objectLists[cind][8]
						cind = cind +1
					if objectLists[besti][6] == True:
						allFinished= True
					else:
						resultion = addToQuest(objectLists[besti],meanSize,fragList,meanLength)
						if len(resultion) == 1:
							#check if best or del 
							betterExists = False
							for this in objectLists:
								if this[8]>resultion[0][8] or (this[8]==resultion[0][8] and this[4]<resultion[0][4]):
									better = True
							if betterExists:
								del(objectLists[besti])
							else:
								objectLists[besti]= resultion[0]
						else:
							objectLists[besti] = resultion[1]
							objectLists.append(resultion[0])

			if allFinished:

				listOfDistances = []
				for shp in objectLists:
					listOfDistances.append(goCompare2(shp[3], generalShape, height, width,12))
				biggest = np.max(np.array(listOfDistances))

				listOfDistances2 = []
				for shp in objectLists:
					listOfDistances2.append(goCompare2(shp[3], generalShape, height, width,12))
				biggest = np.max(np.array(listOfDistances2))

				heur1Max = len(objectLists)//10

				heur1Reduction = []
				it = 0
				for testing  in objectLists:
					#CALCULATE heurVal 
					heurVal = -math.sqrt(testing[4])
					tic2 = 0
					inserted = False
					#just an ordered assertion into diffrel
					for pre in heur1Reduction:
						if tic2>heur1Max:
							#if weve gone over the length of the thing
							#just unsyrence fir the deletion part later
							break 
						if pre[0]<heurVal:
							#insertion at tic
							heur1Reduction.insert(tic2,([heurVal,it]))
							inserted = True
							if len(heur1Reduction)>heur1Max:
								#to ensure we dont go over our specified length, we delete the last element
								del heur1Reduction[len(heur1Reduction)-1]
							break
						tic2=tic2+1
					if inserted==False and len(heur1Reduction)<heur1Max:
						#initial filling of relecantest
						heur1Reduction.append(([heurVal,it]))
					it = it+1

				print(heur1Reduction)


				h1Resuls = []
				for candidate in heur1Reduction:
					actual = objectLists[candidate[1]]
					h1Resuls.append(actual)

				heur2Max = len(h1Resuls)//2
				heur2Reduction = []
				it = 0
				for testing  in h1Resuls:
					#CALCULATE heurVal 
					heurVal =  testing[5]
					tic2 = 0
					inserted = False
					#just an ordered assertion into diffrel
					for pre in heur2Reduction:
						if tic2>heur2Max:
							#if weve gone over the length of the thing
							#just unsyrence fir the deletion part later
							break 
						if pre[0]<heurVal:
							#insertion at tic
							heur2Reduction.insert(tic2,([heurVal,heur1Reduction[it][1]]))
							inserted = True
							if len(heur2Reduction)>heur2Max:
								#to ensure we dont go over our specified length, we delete the last element
								del heur2Reduction[len(heur2Reduction)-1]
							break
						tic2=tic2+1
					if inserted==False and len(heur2Reduction)<heur2Max:
						#initial filling of relecantest
						heur2Reduction.append(([heurVal,heur1Reduction[it][1]]))
					it = it+1

				print(heur2Reduction)

				currentBestI = 0
				currentChosen = objectLists[0]
				currentBest = biggest 
				for candidate in heur2Reduction:
					print(listOfDistances2[candidate[1]])
					if listOfDistances2[candidate[1]]<currentBest:
						currentChosen = objectLists[candidate[1]]
						currentBestI = candidate[1]
						currentBest = listOfDistances2[candidate[1]]

				print("alt method result:",currentBestI," distances",currentBest,",added",currentChosen[5])

				searching = False

				besti = 0
				cind = 1
				while cind < len(objectLists):
					if besti == 0 or (objectLists[cind][4])<(objectLists[besti][4]):
						besti = cind
						bests = objectLists[cind][5]
					cind = cind +1
				#====================================================================================================================
				besti = currentBestI

				chosenOption = objectLists[besti]
				shapeFragOrd = chosenOption[1]
				linksOrd = chosenOption[2]
				fullShape = chosenOption[3]
				truths = chosenOption[5]
				confirmDone = True
				for what  in objectLists:
					if not what[6]:
						confirmDone = False

				print("length of objectLists was ", len(objectLists)," started at ",finalLength, "did thrus: ",counter," congirm done",confirmDone)

				print("biggest was",biggest, 'chosenOption was ',listOfDistances[besti])
				print("we go wrong: ", 1-(listOfDistances[besti]/biggest))

				fullStage1Result = chosenOption
		
		numToReplace = len(fullStage1Result[2])
		linkPoints = []
		for link in fullStage1Result[2]:
			linkPoints.append([link[0],link[len(link)-1]])

	#phase 2===============================================================================================================================================================================
	#=================================================================================================================

	#using slightly larger fragments
		apples = np.load('MPEG7dataset\\test\\extractions2\\apple\\Descriptors.npy')
		bell= np.load('MPEG7dataset\\test\\extractions2\\bell\\Descriptors.npy')
		bone= np.load('MPEG7dataset\\test\\extractions2\\Bone\\Descriptors.npy')
		car= np.load('MPEG7dataset\\test\\extractions2\\car\\Descriptors.npy')
		carriage= np.load('MPEG7dataset\\test\\extractions2\\carriage\\Descriptors.npy')
		cellphone= np.load('MPEG7dataset\\test\\extractions2\\cellular_phone\\Descriptors.npy')
		child= np.load('MPEG7dataset\\test\\extractions2\\children\\Descriptors.npy')
		chopper= np.load('MPEG7dataset\\test\\extractions2\\chopper\\Descriptors.npy')
		face= np.load('MPEG7dataset\\test\\extractions2\\face\\Descriptors.npy')
		flatfish= np.load('MPEG7dataset\\test\\extractions2\\flatfish\\Descriptors.npy')
		fountain= np.load('MPEG7dataset\\test\\extractions2\\fountain\\Descriptors.npy')
		keys= np.load('MPEG7dataset\\test\\extractions2\\key\\Descriptors.npy')
		shoe= np.load('MPEG7dataset\\test\\extractions2\\shoe\\Descriptors.npy')
		watch= np.load('MPEG7dataset\\test\\extractions2\\watch\\Descriptors.npy')
		testImBig = np.concatenate([bone,car,child,carriage,face,bell,chopper,apples,fountain,keys,cellphone,watch,flatfish,shoe], axis = 0)
		chosen =shoe
		flann = FLANN()
		flann.build_index(testImBig,algorithm="kmeans", branching=32, iterations=7, checks=16)

		im = cv2.imread("exampleImage.png")
		height, width, channels = im.shape
		flann.build_index(testImBig,algorithm="kmeans", branching=32, iterations=7, checks=16)
		imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(imgray,127,255,0)
		height, width, channels = im.shape
		im3, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		blank_image = np.zeros((height,width,3), np.uint8)
		c = np.load("MPEG7dataset\\test\\extractions2\\"+folderName+"\\metadata.npy")


		outerContour = []
		for ind, x in enumerate(contours[0]):
			xval = x[0][0]
			yval = x[0][1]
			outerContour.append([yval, xval])
		for x in outerContour:
			xval = x[1]
			yval = x[0]
			blank_image[yval][xval]=[0,255,255]

		frags = np.load('MPEG7dataset\\test\\extractions2\\'+folderName+'\\Fragments.npy')
		listOfPoints = fullShape

		steps = numToReplace + extras
		maxAttempts = maxAA
		adjusted2 = listOfPoints
		fails = 0

		num = 0
		while(num<steps):
			num = num+1
			selection = []
			distances = []

			replacingAlink = False

			for x in range(0,maxAttempts):
				crossing = True
				while crossing:	
					if fails>8:
						fails = 0
						break
					historic = adjusted2

					if numToReplace>0:
						replacingAlink = True
					select = []

					if replacingAlink:
						replacingIndex = len(fullStage1Result[2]) - numToReplace
						p1 = 0
						p2 = 1
						searchInd = 0
						for point in adjusted2:
							if linkPoints[replacingIndex][0][0] == point[0] and linkPoints[replacingIndex][0][1] == point[1]:
								p1 = searchInd
							if linkPoints[replacingIndex][1][0] == point[0] and linkPoints[replacingIndex][1][1] == point[1]:
								p2 = searchInd
							searchInd = searchInd + 1
						if(p2-p1)>(p1+len(adjusted2)-p2):
							select = [p1,p2]
						else:
							select = [p2,p1]

					else:
						#if we're going off of nothing
						select = getRandomSection(len(adjusted2),len(adjusted2)//12, len(adjusted2)//20)
					
					adjusted2 = adjustRoundList(adjusted2,math.floor(select[1]),math.floor(select[0]))
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
			#==============
			rtdd = functionMisc.createDescriptorSet(adjusted2,height,width)
			totalSum = 0
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
				totalSum = totalSum-numNearerfound


			sizeScores = []
			for select in selection:
				sizeScores.append((meanSize  - cv2.contourArea(np.array(select)))**2)

			splineScores = []
			for select in selection:
				splineScores.append(goCompare2(select, generalShape, height, width,12))

			#===========================================0000000000000000
			bestSplTracker = 1000000
			bestSizTracker = 1000000
			bestdistancesTracker = -10000

			currentSplineScore = goCompare2(adjusted2, generalShape, height, width,12)
			currentSizeScore = (meanSize  - cv2.contourArea(np.array(adjusted2)))**2

			best = distances[0]
			bestSpline = splineScores[0]
			bestSizeScore = sizeScores[0]
			bestInd = 0
			foundAtLeastOne = False
			foundAminImprovement = False
			for distInd in range(0,len(distances)):
				#seeing if anything improves
				if(splineScores[distInd]<bestSplTracker):
					bestSplTracker = splineScores[distInd]
				if(sizeScores[distInd]<bestSizTracker):
					bestSizTracker = sizeScores[distInd]
				if(distances[distInd]>bestdistancesTracker):
					bestdistancesTracker = distances[distInd]

				if(splineScores[distInd]<bestSplTracker and sizeScores[distInd]<bestSizTracker and distances[distInd]>bestdistancesTracker):
					indexI = distInd
					best = distances[distInd]
					bestInd = distInd
					bestSpline = splineScores[distInd]				
					bestSizeScore = sizeScores[distInd]
					break

				if( splineScores[distInd]<=bestSpline   and  sizeScores[distInd]<=bestSizeScore  ):
					indexI = distInd
					best = distances[distInd]
					bestInd = distInd
					bestSpline = splineScores[distInd]				
					bestSizeScore = sizeScores[distInd]

				if(splineScores[distInd]<currentSplineScore   and  sizeScores[distInd]<currentSizeScore and distances[distInd]<totalSum):
					foundAtLeastOne = True
				if(splineScores[distInd]<currentSplineScore   and  sizeScores[distInd]<currentSizeScore):
					foundAminImprovement = True
				
				

			if(best<totalSum or replacingAlink):
				adjusted2 = selection[indexI]
				numToReplace = numToReplace-1

				#doesnt do anything while purely replacing \/\/\/
				print("replaced")
			else:
				print("unsatisfied")
				if replacingAlink:
					num = num-1

			print("chosen",bestSpline,bestSizeScore,best," (",bestInd,")")
			print("foundAtLeastOne:", foundAtLeastOne)
			print("foundAminImprovement:", foundAminImprovement)
			print("bestswere:",bestSplTracker,bestSizTracker,bestdistancesTracker)
			print("we had",currentSplineScore,currentSizeScore,totalSum ,"\n") 

		print("fails: ",fails)

		#ctrlv================================================================
		#PIXELS ARE BGR

		for x in adjusted2:
			xval = x[1]
			yval = x[0]
			blank_image[yval][xval]=[255,255,255]

		# cv2.imshow('Edges',blank_image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		additional = datetime.datetime.now().strftime("((%Y!%m!%d)%H!%M)")
		print(additional)
		print(os.path.join('results', str(indexe)+additional+'.png'))
		cv2.imwrite( os.path.join('results', str(indexe)+additional+'.png') , blank_image)

		print("ok finish")
		print("truth",truths)
		print("cArea = ",cv2.contourArea(np.array(fullShape)))
		print("target = ",meanSize)



print(AllRelGet())





#ordered insertion sort on heurVal (code pattern)
			# heur1Reduction = []
			# it = 0
			# for testing  in objectLists:
			# 	#CALCULATE heurVal HBERE
			# 	tic2 = 0
			# 	inserted = False
			# 	#just an ordered assertion into diffrel
			# 	for pre in heur1Reduction:
			# 		if tic2>heur1Max:
			# 			#if weve gone over the length of the thing
			# 			#just unsyrence fir the deletion part later
			# 			break 
			# 		if pre[0]<heurVal:
			# 			#insertion at tic
			# 			heur1Reduction.insert(tic2,([heurVal,it]))
			# 			inserted = True
			# 			if len(heur1Reduction)>heur1Max:
			# 				#to ensure we dont go over our specified length, we delete the last element
			# 				del heur1Reduction[len(heur1Reduction)-1]
			# 			break
			# 		tic2=tic2+1
			# 	if inserted==False and len(heur1Reduction)<heur1Max:
			# 		#initial filling of relecantest
			# 		heur1Reduction.append(([heurVal,it]))
			# 	it = it+1
