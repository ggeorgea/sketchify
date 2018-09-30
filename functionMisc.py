import numpy as np
import cv2
import math
from pyflann import *
import os
from scipy.spatial import distance

def  onSegment( p,  q,  r):
    if ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and  (q[1] <= max(p[1], r[1]) ) and (q[1] >= min(p[1], r[1]))):
       return True
    return False
def max(a,b):
	if a>b :
		return a
	return b
def min(a,b):
	if a>b :
		return b
	return a

def orient(a,b,c):
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if (val == 0):
    	 return 0 
    if(val>0):
    	return 1
    else:
    	return 2

newDesc = [1,2,3]
newFrag = [1,2,3]
neweEdge = [newDesc, newFrag]


def thin(fragment1:list):
	fragment2 = []
	fragment2.append(fragment1[0])
	for p in fragment1:
		if (fragment2[len(fragment2)-1][0]!= p[0]) or (fragment2[len(fragment2)-1][1]!= p[1]):
			fragment2.append(p)
	return fragment2


#this method fills a line between two points, so can create straight lines, or fill in gaps
def fill(fragment1:list):
	fragment2 = []
	fi = 0
	se = 1
	while se<len(fragment1):
		fragment2.append(fragment1[fi])
		x = fragment1[fi][1]
		y = fragment1[fi][0]
		i = fragment1[se][1]
		j = fragment1[se][0]
		if ((x==i) and (-2< y-j<2)) or ((y==j) and (-2< x-i<2)):
			pass
		elif x==i:
			op = [y,j]
			if(y>j):
				op = [j,y]
			for s in range(op[0],op[1]):
				fragment2.append(np.array((s,x)))
		elif y==j:
			op = [x,i]
			if(x>i):
				op = [i,x]
			for s in range(op[0],op[1]):
				fragment2.append(np.array((y,s)))
		else:
			s = (j-y)/(i-x)
			case = ""
			if   s>=1    and x<i:
				case = "a"
			elif s>=1    and i<x:
				case = "b"
			elif 1>s>0  and x<i:
				case = "c"
			elif 1>s>0  and i<x:
				case = "d"
			elif 0>s>=-1 and x<i:
				case = "e"
			elif 0>s>=-1 and i<x:
				case = "f"
			elif s<-1   and x<i:
				case = "g"
			elif s< -1  and i<x:
				case = "h"			
			currx = x
			curry = y
			while not (((currx ==i) and (curry == j) )) :
				#print(fragment2)
				po = 1
				if case == "f" or case == "g" or case == "h":
					po=-1
				m = 99999999*po
				if currx != x :
					m = (curry-y)/(currx -x)
				#print(case,m, currx,curry,x,y,s)
				
				if i==currx and curry==(j+1):
					fragment2.append(np.array((j,i))) 
				elif i==currx and curry==(j-1):
					fragment2.append(np.array((j,i)))
				elif i==(currx+1) and curry==j:
					fragment2.append(np.array((j,i))) 
				elif i==(currx-1) and curry==j:
					fragment2.append(np.array((j,i)))
				elif case == "a":
					if s>m:
						fragment2.append(np.array((curry+1,currx)))
					else:
						fragment2.append(np.array((curry+1,currx)))
						fragment2.append(np.array((curry+1,currx+1)))
				elif case == "b":
					if s>m:
						fragment2.append(np.array((curry-1,currx)))
					else:
						fragment2.append(np.array((curry-1,currx)))
						fragment2.append(np.array((curry-1,currx-1)))
				elif case == "c":
					if s>m:
						fragment2.append(np.array((curry,currx+1)))
						fragment2.append(np.array((curry+1,currx+1)))
					else:
						fragment2.append(np.array((curry,currx+1)))
				elif case == "d":
					if s>m:
						fragment2.append(np.array((curry,currx-1)))
						fragment2.append(np.array((curry-1,currx-1)))
					else:
						fragment2.append(np.array((curry,currx-1)))
				elif case == "e":
					if s>m:
						fragment2.append(np.array((curry,currx+1)))
					else:
						fragment2.append(np.array((curry,currx+1)))
						fragment2.append(np.array((curry-1,currx+1)))
				elif case == "f":
					if s>m:
						fragment2.append(np.array((curry,currx-1)))
					else:
						fragment2.append(np.array((curry,currx-1)))
						fragment2.append(np.array((curry+1,currx-1)))
				elif case == "g":
					if s>m:
						fragment2.append(np.array((curry-1,currx)))
						fragment2.append(np.array((curry-1,currx+1)))
					else:
						fragment2.append(np.array((curry-1,currx)))
				elif case == "h":
					if s>m:
						fragment2.append(np.array((curry+1,currx)))
						fragment2.append(np.array((curry+1,currx-1)))
					else:
						fragment2.append(np.array((curry+1,currx)))
				currx = fragment2[len(fragment2)-1][1]
				curry = fragment2[len(fragment2)-1][0]
		fi = fi+1
		se = se+1	
	fragment2.append(fragment1[len(fragment1)-1])
	return thin(fragment2)

#method to translate one fragment into another position defined by end points
def ratioTransfer(newPoint, fragment1:list):
	#CREATING GAPS
	fragment2 = []
	st = fragment1[0]
	o = fragment1[len(fragment1)-1]
	n = newPoint
	for p in fragment1:
		e0l0 = p[0] - st[0]
		e0l1 = o[0] - st[0]
		e0l2 = n[0] -  o[0]
		if e0l1!=0:
			l0n = ((e0l2+0.0)/(e0l1+0.0))*e0l0
			new0 = math.floor(p[0] + l0n  )
		else :
			return []
		e1l0 = p[1] - st[1]
		e1l1 = o[1] - st[1]
		e1l2 = n[1] -  o[1]
		if e1l1!=0:
			l1n = ((e1l2+0.0)/(e1l1+0.0))*e1l0
			new1 = math.floor(p[1] + l1n  )
		else:
			return []
		fragment2.append(np.array((new0, new1)))

	fragment2 = fill(fragment2)

	return fragment2


def translate(dirV, fragment1:list):
	fragment2 = []
	for p in fragment1:
		fragment2.append(  np.array([p[0]+dirV[0], p[1]+dirV[1]]))
	return fragment2

#defines a method to return the relation between two points in x,y
def relatePoints (p1, p2):
	return [p1[0]-p2[0], p1[1]-p2[1]]

def concat (fragment1:list, fragment2:list):
	newList = []
	newList.extend(fragment1)
	lastp = fragment1[len(fragment1)-1]
	firstp = fragment2[0]
	difference = relatePoints(firstp,lastp)
	i = len(newList)-1
	del newList[i]
	for p in fragment2:
		newList.append(   np.array([p[0]-difference[0], p[1]-difference[1]]))
	return newList

def distance(fragment1:list):
	return (( fragment1[0][0] - fragment1[len(fragment1)-1][0] )**2 +(fragment1[0][1] - fragment1[len(fragment1)-1][1])**2)**0.5


#a method to rotate an edge n degrees bout a point 
def rotatePoint (p1, rotatepoint, rads):
	p2=[0,0]	
	try:
		p2[0]= int((rotatepoint[0] + (math.cos(rads)*(p1[0]-rotatepoint[0])) - (math.sin(rads)*(p1[1]-rotatepoint[1])))//1)
		p2[1]= int((rotatepoint[1] + (math.sin(rads)*(p1[0]-rotatepoint[0])) + (math.cos(rads)*(p1[1]-rotatepoint[1])))//1)
	except ValueError:
		return p2
	return p2

def rotateEdge(edge1, roratepoint, rads):
	edge2 = []
	edge3 = []
	for point in edge1:
		edge2.append(rotatePoint(point,roratepoint,rads))
	for p2 in range(0,len(edge2)-2):
		if edge2[p2]!=edge2[p2+1]:
			edge3.append(edge2[p2])
	edge3.append(edge2[len(edge2)-1])
	return edge3

def ecross (edge1,edge2):
	for x in edge1:
		for y in edge2:
			if x[0]==y[0] and x[1] == y[1] :
				#print(x)
				return True

	return False 

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


def createDescriptorSet(fullEdge, height,width,targetLength = 0,sect = 3):
	DescrList = []
	end = len(fullEdge)
	tri = math.floor(end/sect)
	i = 0
	for x in range(0,tri):
		for y in range(0, tri ):
			i = i+1
			if i%((tri**2)//80) == 0:
				todesc = fullEdge[x : (end - y) : 1]
				DescrList.append(generateDescriptor(todesc,height,width))
	if targetLength!=0:
		startOfFrg = end//2 - targetLength//2
		endOffFrg = startOfFrg + targetLength
		DescrList.append(generateDescriptor(fullEdge[startOfFrg:endOffFrg:1],height,width))
	return DescrList

def rotateEdgeToEdge(st, ed, transdEdgeNew):
	transdEdgeNew = translate(relatePoints(st,transdEdgeNew[0]),transdEdgeNew)
	pstart = transdEdgeNew[0]
	pendTest = ed
	pendOld = transdEdgeNew[len(transdEdgeNew)-1]
	yyy1 = pendTest[0]-pstart[0]
	xxx1 = pendTest[1]-pstart[1]
	pi3 = -math.atan(yyy1/xxx1)
	yyy = pendOld[0]-pstart[0]
	xxx = pendOld[1]-pstart[1]
	pi2 = -math.atan(yyy/xxx)
	properEdge =  rotateEdge(transdEdgeNew, transdEdgeNew[0], pi3-pi2)
	related1= relatePoints(properEdge[len(properEdge)-1], properEdge[0])
	related2 = relatePoints(pendTest,properEdge[0])
	if (related1[0]*related2[0])<0 or (related1[1]*related2[1])<0:
		properEdge = rotateEdge(properEdge,properEdge[0],math.pi)
	return properEdge


def testing():
	im = cv2.imread('non_rigid_shape_A\\non_rigid_shape_A\\cow\\cow18.tif')
	height, width, channels = im.shape


	blank_image = np.zeros((height*2,width*2,3), np.uint8)
	folderName = "flatfish"

	c = np.load("MPEG7dataset\\test\\extractions\\"+folderName+"\\metadata.npy")

	# dimensions
	im = cv2.imread('MPEG7dataset\\test\\flatfish\\flatfish-1.png')
	height, width, channels = im.shape
	blank_image = np.zeros(((height*5)//2,((width*5)//2),3), np.uint8)

	apples = np.load('MPEG7dataset\\test\\extractions\\flatfish\\Descriptors.npy')
	cowFrags = np.load('MPEG7dataset\\test\\extractions\\'+folderName+'\\Fragments.npy')


	for x in cowFrags[2020]:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[0,255,255]

	for x in cowFrags[1399]:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[80,255,120]


	mp = [cowFrags[2020][len(cowFrags[2020])//2][0],cowFrags[2020][len(cowFrags[2020])//2][1]]

	newEdge = rotateEdge(cowFrags[2020],mp,0)
	for x in newEdge:
		xval = x[1]
		yval = x[0]
		#blank_image[yval][xval]=[255,0,120]
	newEdge = rotateEdge(cowFrags[2020],mp,0.5)
	for x in newEdge:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[255,0,120]
	newEdge = rotateEdge(cowFrags[2020],mp,3)
	for x in newEdge:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[255,0,120]
	newEdge = rotateEdge(cowFrags[2020],mp,1)
	for x in newEdge:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[255,0,120]

	cfRev = cowFrags[1399]
	cfRevA = cfRev[::-1]
	newEdge = concat(rotateEdge(cowFrags[2020],mp,2),cfRevA)
	for x in newEdge:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[0,0,255]

	transdEdge = translate([0,250],newEdge)
	for x in transdEdge:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[0,0,255]


	pot = (350,550)
	transdEdge2 = ratioTransfer(pot,transdEdge) 
	#print(transdEdge2)
	for x in transdEdge2:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[255,0,0]
	blank_image[pot]=[255,255,0]



	fullEdge = transdEdge2
	Dset = createDescriptorSet(fullEdge, height,width)

	finTestFrag =  rotateEdgeToEdge([600,700], [500,800], transdEdge)

	for x in finTestFrag:
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[0,255,0]

	for x in fill([[600,700],[500,800]]):
		xval = x[1]
		yval = x[0]
		blank_image[yval][xval]=[255,255,255]




	cv2.imshow('Edges',blank_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def testing2():
	fillable = [np.array((100,100)),np.array((92,112))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))
	fillable = [np.array((100,100)),np.array((91,105))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))
	fillable = [np.array((100,100)),np.array((88,98))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))
	fillable = [np.array((100,100)),np.array((91,97))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))
	fillable = [np.array((100,100)),np.array((92,91))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))

	fillable = [np.array((100,100)),np.array((97,111))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))
	fillable = [np.array((100,100)),np.array((96,92))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))

	fillable = [np.array((100,100)),np.array((100,116))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))
	fillable = [np.array((100,100)),np.array((101,93))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))

	fillable = [np.array((100,100)),np.array((105,110))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))
	fillable = [np.array((100,100)),np.array((105,90))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))

	fillable = [np.array((100,100)),np.array((110,110))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))
	fillable = [np.array((100,100)),np.array((110,105))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))
	fillable = [np.array((100,100)),np.array((110,100))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))
	fillable = [np.array((100,100)),np.array((110,95))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))
	fillable = [np.array((100,100)),np.array((110,90))]
	filled = fill(fillable)
	print("\nfinal:\n",filled)
	print("\nthin:\n",thin(filled))

#testing()
#testing2()