import cv2
import numpy as np
from sklearn.cluster import KMeans
import cPickle
import time
DEBUG = True
video_path = "/Users/Ruoyu/Desktop/comp9517/project/south_park.mp4"
output_path = "/Users/Ruoyu/Desktop/comp9517/project/keypoint.dat"


class keypoint_class:
    frame = 0
    keyPoint = None
    descriptor = []
    label = -1



def extract_frame(capture, frame_no = 0):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    (ret, frame) = capture.read()
    # if ret == False:
    #     print "error reading the frame"
    return (ret, frame)




if __name__ == '__main__':

    start_time = time.time()
    # read the file
    capture = cv2.VideoCapture(video_path)
    nFrame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    print "Num of Frames:", nFrame
    print "Fps:", fps

    #we deal every second



    #init lists
    sift_keyPointList = []
    sift_descriptorList = []
    # sift_frameList = []

    #deal from the second frame
    current = 1
    while(current < nFrame):

        #get continuous three frames
        (ret, cur_frame) = extract_frame(capture, current)
        if ret == False:
            break
        (ret, pre_frame) = extract_frame(capture, current - 1)
        if ret == False:
            break
        (ret, next_frame) = extract_frame(capture, current + 1)
        if ret == False:
            break

        #denoise
        # cur_frame = cv2.fastNlMeansDenoisingColored(cur_frame)
        # pre_frame = cv2.fastNlMeansDenoisingColored(pre_frame)
        # next_frame = cv2.fastNlMeansDenoisingColored(next_frame)


        #get sift keypoints
        siftDetector = cv2.xfeatures2d.SIFT_create()
        (pre_sift_kps, pre_sift_descriptor) = siftDetector.detectAndCompute(pre_frame, None)
        (cur_sift_kps, cur_sift_descriptor) = siftDetector.detectAndCompute(cur_frame, None)
        (next_sift_kps, next_sift_descriptor) = siftDetector.detectAndCompute(cur_frame, None)


        # match key points
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        firstMatches = matcher.knnMatch(cur_sift_descriptor, pre_sift_descriptor, 2)
        secondMatches = matcher.knnMatch(cur_sift_descriptor, next_sift_descriptor, 2)
        keyPoints = []
        sift_descriptor = []
        for m in firstMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                idx = m[0].queryIdx
                pre_idx = m[0].trainIdx
                next_idx = 0
                flag = False

                for n in secondMatches:
                    if idx == n[0].queryIdx and len(n) == 2 and n[0].distance < n[1].distance * 0.75:
                        flag = True
                        next_idx = n[0].trainIdx
                        break

                if flag == True:
                    kp = cur_sift_kps[idx]
                    pre_kp = pre_sift_kps[pre_idx]
                    next_kp = next_sift_kps[next_idx]
                    kp.response = (kp.response + pre_kp.response + next_kp.response)/3
                    descriptor = cur_sift_descriptor[idx]
                    # print descriptor
                    # cv2.waitKey(0)
                    keyPoints.append(kp)

                    #create class
                    kp_class = keypoint_class()
                    kp_class.frame = current
                    kp_class.keyPoint = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                    kp_class.descriptor = descriptor

                    #append to list
                    sift_keyPointList.append(kp_class)
                    sift_descriptorList.append(descriptor)




        # output = cv2.drawKeypoints(cur_frame, keyPoints, None)
        # output1 = cv2.drawKeypoints(cur_frame, cur_sift_kps, None)
        # print "before:", len(cur_sift_kps)
        # print "after:", len(keyPoints)
        # cv2.imshow("before", output)
        # cv2.imshow("after", output1)


        # cv2.waitKey(0)
        # cv2.destroyAllWindows()








        if DEBUG == True:
            print ("DEBUG: frame number is %d, keyPoints length is %d" %(current, len(keyPoints)))

        current += 100

    #end of loop

    print len(sift_keyPointList)
    # print sift_descriptorList

    # #do k-means clusters

    with open(output_path, 'wb') as f:
        f.truncate()
        saved_file = (sift_keyPointList, np.asarray(sift_descriptorList))
        cPickle.dump(saved_file, f)


    # # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #
    # # Set flags (Just to avoid line break in the code)
    # flags = cv2.KMEANS_RANDOM_CENTERS
    #
    # # Apply KMeans
    # compactness, labels, centers = cv2.kmeans(sift_descriptorList, 10, None, criteria, 10, flags)
    # print labels






    #release the video
    capture.release()
    print("--- %s seconds ---" % (time.time() - start_time))