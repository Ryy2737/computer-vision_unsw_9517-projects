import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
import cPickle
import time


kmean_path = "/Users/Ruoyu/Desktop/comp9517/project/kmeans.dat"
keypoint_path = "/Users/Ruoyu/Desktop/comp9517/project/keypoint.dat"
output_path = "/Users/Ruoyu/Desktop/comp9517/project/index.dat"


nClusters = 500

def normalizeVector(vector):
    lenth = len(vector)
    total = 0.0
    for i in range(0, lenth):
        total = float(vector[i])*float(vector[i]) + total

    vecLen = math.sqrt(total)

    vec = []
    for i in range(0, lenth):
        val = float(vector[i])/vecLen
        vec.append(val)

    return np.array(vec)



class keypoint_class:
    frame = 0
    keyPoint = None
    descriptor = None
    label = -1

if __name__ == '__main__':
    start_time = time.time()

    #read the keyoint class
    keypointList = None
    with open(keypoint_path, 'rb') as f:
        file_content = cPickle.load(f)
        keypointList = file_content[0]

    #read the classifier
    kmeans = None


    with open(kmean_path, 'rb') as f:
        kmeans = cPickle.load(f)

    labelList = kmeans.labels_

    #get the label frequency

    labelTerms = []
    for i in range(0, nClusters):

        labelTerms.append(0)

    for i in range(0, len(labelList)) :
        label = labelList[i]
        labelTerms[label] += 1

    # extract

    terms = []
    for i in range(0, nClusters):
        tuple = (i, labelTerms[i])
        terms.append(tuple)

    rankTerms = sorted(terms, key=lambda tup:tup[1], reverse=True)

    # print rankTerms

    #build stop list
    stopList = []

    for i in range(0, 50):
        tuple = rankTerms[i]
        clusterNo = tuple[0]
        stopList.append(clusterNo)

    stopList.sort()


    #get frame list (frameId, [wordId])

    keypoint_len = len(keypointList)

    frameMap = {}


    for i in range(0, keypoint_len):
        frameId = keypointList[i].frame
        label = labelList[i]

        if frameId not in frameMap.keys():
            # create chile map
            frameMap[frameId] = {}

            #initialize
            for j in range(0, nClusters):
                frameMap[frameId][j] = 0

            frameMap[frameId][label] += 1

        else:
            frameMap[frameId][label] += 1


    nFrames = len(frameMap.keys())

    print "Num pf frames:", nFrames

    #create index

    index = {}

    #initialize
    for i in range(0, nClusters):
        index[i] = []

    for i in range(0, nClusters):
        for (k, v) in frameMap.items():
            count = v[i]
            if count > 0:
                tuple = (k, count)
                index[i].append(tuple)

    # print index[0]
    # print len(index[0])

    #create idf vector
    df = []
    for i in range(0, nClusters):
        if i not in stopList:
            df.append(len(index[i]))



    idf = []
    for i in range(0, len(df)):
        frequency = float(nFrames)/float(df[i])
        inverse = math.log10(frequency)
        idf.append(inverse)

    idfVector = np.array(idf)





    # create frame -- TFIDF vector
    tf_idf_vector = {}

    for (k, v) in frameMap.items():
        tf_weight = []

        for i in range (0, nClusters):
            if i not in stopList:
                frequency = float(v[i] + 1)
                tf_wt = math.log10(frequency)
                tf_weight.append(tf_wt)

        tfVector = np.array(tf_weight)


        #vector multiply
        tfidfVector = normalizeVector(tfVector*idfVector)

        tf_idf_vector[k] = tfidfVector

    # print len(tf_idf_vector)










    with open(output_path, 'wb') as f:
        f.truncate()
        file = [nFrames,index, tf_idf_vector, idfVector, stopList]
        cPickle.dump(file, f)

    print("--- %s seconds ---" % (time.time() - start_time))



