import numpy as np
import cv2
import math
from pyflann import *
import os
import random
import functionMisc
import sys

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
#testImBig = flatfish
#testImBig.tofile('descLib.npy', sep=',')
np.save('descLib.npy',testImBig)


apples = np.load('MPEG7dataset\\test\\extractions\\apple\\Fragments.npy')
bell= np.load('MPEG7dataset\\test\\extractions\\bell\\Fragments.npy')
bone= np.load('MPEG7dataset\\test\\extractions\\Bone\\Fragments.npy')
car= np.load('MPEG7dataset\\test\\extractions\\car\\Fragments.npy')
carriage= np.load('MPEG7dataset\\test\\extractions\\carriage\\Fragments.npy')
cellphone= np.load('MPEG7dataset\\test\\extractions\\cellular_phone\\Fragments.npy')
child= np.load('MPEG7dataset\\test\\extractions\\children\\Fragments.npy')
chopper= np.load('MPEG7dataset\\test\\extractions\\chopper\\Fragments.npy')
face= np.load('MPEG7dataset\\test\\extractions\\face\\Fragments.npy')
flatfish= np.load('MPEG7dataset\\test\\extractions\\flatfish\\Fragments.npy')
fountain= np.load('MPEG7dataset\\test\\extractions\\fountain\\Fragments.npy')
keys= np.load('MPEG7dataset\\test\\extractions\\key\\Fragments.npy')
shoe= np.load('MPEG7dataset\\test\\extractions\\shoe\\Fragments.npy')
watch= np.load('MPEG7dataset\\test\\extractions\\watch\\Fragments.npy')
testFrags = np.concatenate([bone,car,child,carriage,cellphone,face,keys,bell,apples,shoe,fountain,cellphone,chopper,watch,flatfish], axis = 0)
#testFrags.tofile('fragLib.npy', sep=',')
np.save('fragLib.npy',testFrags)
