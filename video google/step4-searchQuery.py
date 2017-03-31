import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
import cPickle
import time
import imutils

kmean_path = "/Users/Ruoyu/Desktop/comp9517/project/kmeans.dat"
keypoint_path = "/Users/Ruoyu/Desktop/comp9517/project/keypoint.dat"
index_path = "/Users/Ruoyu/Desktop/comp9517/project/index.dat"
video_path = "/Users/Ruoyu/Desktop/comp9517/project/south_park.mp4"
query_path = "/Users/Ruoyu/Desktop/comp9517/project/kyle.png"

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

def calculateValue(vector1, vector2):
    lenth1 = len(vector1)
    lenth2 = len(vector2)
    if lenth1 != lenth2:
        print "error"
        return -1.0

    value = 0.0
    for i in range(0, lenth2):
        value = value + float(vector1[i])*float(vector2[i])


    return value

def extract_frame(capture, frame_no = 0):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    (ret, frame) = capture.read()
    # if ret == False:
    #     print "error reading the frame"
    return (ret, frame)

if __name__ == '__main__':

    nFrames = 0
    nClusters = 500
    index = None
    tfidfMap = None
    idfVector = None
    stopList = None
    # read the classifier
    kmeans = None

    with open(kmean_path, 'rb') as f:
        kmeans = cPickle.load(f)

    #read the index
    with open(index_path, 'rb') as f:
        file = cPickle.load(f)
        nFrames = file[0]
        index = file[1]
        tfidfMap = file[2]
        idfVector = file[3]
        stopList = file[4]

    #load the screen shot
    inputImg = cv2.imread(query_path)
    start_time = time.time()
    #detect the feature point
    siftDetector = cv2.xfeatures2d.SIFT_create()
    (keyPoint, des) = siftDetector.detectAndCompute(inputImg, None)

    labels = kmeans.predict(des)

    query = []
    for i in range (0, nClusters):
        query.append(0)

    # generate label vector
    for label in labels:
        query[label] += 1

    # print query

    tf_weight = []
    for i in range(0, nClusters):
        if i not in stopList:
            frequency = float(query[i] + 1)
            tf_wt = math.log10(frequency)
            tf_weight.append(tf_wt)

    tfVector = np.array(tf_weight)
    # print len(tfVector)


    tfidfQuery = tfVector*idfVector

    result = []

    for (k, v) in tfidfMap.items():
        frameId = k
        value = calculateValue(tfidfQuery, v)

        tuple = (frameId, value)

        result.append(tuple)

    sortedList = sorted(result, key=lambda tup:tup[1], reverse=True)
    #
    print("Retrieve time:  %s seconds" % (time.time() - start_time))
    # print sortedList
    capture = cv2.VideoCapture(video_path)
    final = []
    #display
    for i in range(0, 10):
        tuple = sortedList[i]
        frameId = tuple[0]
        score = tuple[1]
        (ret, frameImg) = extract_frame(capture, frameId)

        siftDetector = cv2.xfeatures2d.SIFT_create()
        keypoints1 = siftDetector.detect(inputImg)
        keypoints2 = siftDetector.detect(frameImg)

        # extract fecture vector
        (keypoints1, description1) = siftDetector.compute(inputImg, keypoints1)
        (keypoints2, description2) = siftDetector.compute(frameImg, keypoints2)

        matcher = cv2.DescriptorMatcher_create("BruteForce")
        # matches = []
        rawMatches = matcher.knnMatch(description1, description2, 2)
        matches = []

        min_x = None
        min_y = None
        max_x = None
        max_y = None
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                matches.append((m[0].trainIdx, m[0].queryIdx))



        len_matches = len(matches)


        if len_matches <= 5:
            continue

        # keyPointList = []
        for match in matches:
            idx = match[0]
            keyPoint = keypoints2[idx]
            x = keyPoint.pt[0]
            y = keyPoint.pt[1]

            if min_x == None or min_x > x:
                min_x = x

            if max_x == None or max_x < x:
                max_x = x

            if min_y == None or min_y > y:
                min_y = y

            if max_y == None or max_y < y:
                max_y = y


        tup = (frameId, score, score*len_matches, min_x, max_x, min_y, max_y)
        final.append(tup)

    finalList = sorted(final, key=lambda tup: tup[1], reverse=True)

    print "Find:", len(finalList), "results"
    print finalList

    for i in range(0, len(finalList)):
        tuple = finalList[i]
        frameId = tuple[0]
        score = tuple[1]
        x1 = int(tuple[3])
        y1 = int(tuple[5])
        x2 = int(tuple[4])
        y2 = int(tuple[6])

        (ret, img) = extract_frame(capture, frameId)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        name = "rank: "+ str(i) + ", frame ID: " + str(frameId) + ", score: "+ str(score)
        cv2.imshow(name, img)




    cv2.waitKey(0)
    capture.release()
