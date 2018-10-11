import os
import cv2 
import sys
import sketchAlt
i = 0
j = len(os.listdir('./Multiracial'))
for filename in os.listdir('./Multiracial'):
	print("onto ",i," out of ",j)
	sys.stdout.flush()
	i = i+1

	if filename.endswith('.jpg'):
		

		loc = 'Multiracial/'+ filename
		image = cv2.imread(loc  ,1);

		newpath = "./faceIms2/"
		dst =(newpath+filename)
		if not os.path.exists(newpath):
		    os.makedirs(newpath)
		cv2.imwrite( dst, image );
		# # print(dst)
		# cv2.imshow('i',image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# break




		newpath2 = newpath+filename[0:-3]
		if not os.path.exists(newpath2):
		    os.makedirs(newpath2)

		for ind in range(0,10):    
			print('\t',ind)
			sys.stdout.flush()
			num = str(ind)


			dst2 =(newpath2+'/gen'+num +'.jpg')
			#print(dst2)
			newImage = sketchAlt.petsketch(image)
			cv2.imwrite( dst2, newImage );
		