import numpy as np
import cv2
import math
from pyflann import *
import os


def AllRelGet():

	folderName = "flatfish"

	c = np.load("MPEG7dataset\\test\\extractions\\"+folderName+"\\metadata.npy")

	# just for dimensions
	im = cv2.imread('MPEG7dataset\\test\\apple\\apple-1.png')
	height, width, channels = im.shape
	blank_image = np.zeros(((height*8)//2,((width*8)//2),3), np.uint8)

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
	flann = FLANN()
	flann.build_index(testImBig,algorithm="kmeans", branching=32, iterations=7, checks=16)


	#relavantest stores the fragment indicies with the highest relevance, and their indicies
	relavantest = []
	#looker is the bound on the length of relevantest
	looker = 70

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


	diffrelMax =30
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

	cowFrags = np.load('MPEG7dataset\\test\\extractions\\'+folderName+'\\Fragments.npy')
	print("\n\n")
	for top in relavantest:
		number = top[1]
		for x in cowFrags[number]:
			xval = x[1]
			yval = x[0]
			blank_image[yval][xval]=[255,255,255]


	cv2.imshow('Edges',blank_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
AllRelGet()