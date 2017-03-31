import cv2
import numpy as np
from sklearn.cluster import KMeans
import cPickle
import time

read_path = "/Users/Ruoyu/Desktop/comp9517/project/keypoint.dat"
output_path = "/Users/Ruoyu/Desktop/comp9517/project/kmeans.dat"

class keypoint_class:
    frame = 0
    keyPoint = None
    descriptor = None
    label = -1



if __name__ == '__main__':

    start_time = time.time()
    sift_keyPointList = None
    sift_descriptorList = None

    #read from the file
    with open(read_path, 'rb') as f:
        file_content = cPickle.load(f)
        sift_keyPointList = file_content[0]
        sift_descriptorList = file_content[1]




    print len(sift_keyPointList)
    print sift_descriptorList

    kmeans = KMeans(n_clusters=500, max_iter=500, n_init=10, init='k-means++', random_state=100)
    kmeans.fit(sift_descriptorList)
    print kmeans.labels_

    #write to file
    with open(output_path, 'wb') as f:
        f.truncate()

        cPickle.dump(kmeans, f)




    print("--- %s seconds ---" % (time.time() - start_time))

