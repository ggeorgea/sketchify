import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('bt.jpg')
# height, width, channels = img.shape
# img = cv2.fastNlMeansDenoisingColored(img,None,20,20,7,21)
# edges = cv2.Canny(img,100,200)
#kernel = np.ones((2,2),np.uint8)
# dilation = cv2.dilate(edges,kernel,iterations = 1)
# im3, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# c2 = []
# for cont in contours:
#   #print(len(cont))
#   if len(cont)>20:
#     c2.append(cont)







img = cv2.imread('bt.jpg')
height, width, channels = img.shape

img = cv2.fastNlMeansDenoisingColored(img,None,20,20,7,21)

edges = cv2.Canny(img,100,200)

kernel = np.ones((2,2),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 1)
im3, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
blank_image2 = np.zeros((height,width,3), np.uint8)
cv2.drawContours(blank_image2, contours, -1, (255,255,255), 1)

plt.subplot(141),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])


plt.subplot(143),plt.imshow(blank_image2,cmap = 'gray')
plt.title('contour Image'), plt.xticks([]), plt.yticks([])



c2 = []
for cont in contours:
  print(len(cont))
  if len(cont)>20:
    c2.append(cont)
print('#: ',len(contours),len(c2))
blank_image = np.zeros((height,width,3), np.uint8)
cv2.drawContours(blank_image, c2, -1, (255,255,255), 1)


plt.subplot(144),plt.imshow(blank_image,cmap = 'gray')
plt.title('contour Image'), plt.xticks([]), plt.yticks([])


plt.show()

# im = img

# img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
# height, width, channels = im.shape
# imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_TOZERO_INV)#0)
# #thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
# thresh = cv2.fastNlMeansDenoising(thresh,None,8,7,21)

# #cv2.imshow('t',thresh)
# #height, width, channels = im.shape
# im3, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


# blank_image = np.zeros((height,width,3), np.uint8)
# cv2.drawContours(blank_image, contours, -1, (255,255,255), 1)
# cv2.imshow('Edges',blank_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
