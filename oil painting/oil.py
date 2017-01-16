import cv
import cv2
import numpy as np

img = cv2.imread("girl.jpg")
dim=img.shape
one_band = np.zeros((dim[0],dim[1],1))#BGR
#task 1
for i in range(0,dim[0]):
	for j in range(0,dim[1]): 
		one_band[i][j][0]=0.299*img[i,j][2]+0.587*img[i,j][1]+0.114*img[i,j][0]

one_band_img=np.uint8(one_band)
cv2.imwrite( "task_1.jpg", one_band_img )

#task 2
r=2
img_1 = cv2.imread('task_1.jpg')

for i in range(0,dim[0]):
	for j in range(0,dim[1]):
		mask = np.zeros(img_1.shape[:2], np.uint8)
		for x in range(i-r,i+r+1):
			if (x<0) | (x>dim[0]-1):
				continue
			for y in range(j-r,j+r+1):
				if (y<0) | (y>dim[1]-1):
					continue
				mask[x][y]=255

		hist_mask=cv2.calcHist([img_1],[0],mask,[256],[0,256])
		min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(hist_mask)

		one_band[i][j][0]=max_loc[1]
one_band_img=np.uint8(one_band)
cv2.imwrite( "task_2.jpg", one_band_img)
#task 3
for i in range(0,dim[0]):
	for j in range(0,dim[1]):
		index_list=[]
		intensity_0=0
		intensity_1=0
		intensity_2=0
		for x in range(i-r,i+r+1):
			if (x<0) | (x>dim[0]-1):
				continue
			for y in range(j-r,j+r+1):
				if (y<0) | (y>dim[1]-1):
					continue
				if (one_band_img[i][j][0]==one_band_img[x][y][0]):
					index_list.append([x,y])
					#print "yes"
		for index in index_list:
			intensity_0+=img[index[0],index[1]][0]
			intensity_1+=img[index[0],index[1]][1]
			intensity_2+=img[index[0],index[1]][2]
		img[i][j][0]=intensity_0/len(index_list)
		img[i][j][1]=intensity_1/len(index_list)
		img[i][j][2]=intensity_2/len(index_list)

average_triple_band_img=np.uint8(img)
cv2.imwrite( "task_3.jpg", average_triple_band_img)

