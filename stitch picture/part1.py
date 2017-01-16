import part1_lib
import numpy as np
import cv2
import imutils
import sys
import os


dir_name = raw_input('Enter images directory: ')
content_list = []
images = []
for content in os.listdir(dir_name): # "." means current directory
    content_list.append(content)

print content_list

for content in content_list:
    if ".jpg" not in content:
        continue
    filePath = dir_name+"/"+content
    print "find one image: ", filePath
    image = cv2.imread(filePath)
    image = imutils.resize(image, width=400)
    images.append(image)

print ("scaned %d images" %len(images))
baseIdx = int(raw_input("Enter which image is the base image (0 - %d): " %(len(images) - 1)))


if baseIdx <= 0 or baseIdx >= len(images):
    print >> sys.stderr, ("Error base index")
    sys.exit(-1)

result = images[baseIdx]
del images[baseIdx]

while len(images) > 0:
    (newIdx, matches, H, status) = part1_lib.findBestMatches(result, images)
    print newIdx

    result = part1_lib.stitch(result, images[newIdx], matches, H, status)
    del images[newIdx]


# result = imutils.resize(result, width=1000)
cv2.imshow("Result", result)

cv2.waitKey(0)