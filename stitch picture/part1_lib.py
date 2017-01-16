import numpy as np
import cv2
import math


def findBestMatchesFirstRound(images):
    baseIdx = 0
    newIdx = 0
    ratio = 0.0
    bestMatches = None
    bestH = None
    bestStatus = None
    for i in range(0, len(images)):
        for j in range(0, len(images)):
            if i == j:
                continue
            (baseKps, baseFeatures) = detectAndDescribe(images[i])
            (newKps, newFeatures) = detectAndDescribe(images[j])

            M = matchKeyPoints(newKps, baseKps, newFeatures, baseFeatures)
            if M is None:
                continue
            (matches, H, status) = M
            inlierRatio = float(np.sum(status))/float(len(status))

            if ratio == 0.0 or inlierRatio > ratio:
                ratio = inlierRatio
                baseIdx = i
                newIdx = j
                bestMatches = matches
                bestH = H
                bestStatus = status

    return (baseIdx, newIdx, bestMatches, bestH, bestStatus)

def findBestMatches(baseImage, images):
    newIdx = 0
    ratio = 0.0
    bestMatches = None
    bestH = None
    bestStatus = None
    for i in range(0, len(images)):
        (baseKps, baseFeatures) = detectAndDescribe(baseImage)
        (newKps, newFeatures) = detectAndDescribe(images[i])

        M = matchKeyPoints(newKps, baseKps, newFeatures, baseFeatures)
        if M is None:
            continue
        (matches, H, status) = M
        inlierRatio = float(np.sum(status))/float(len(status))

        if ratio == 0.0 or inlierRatio > ratio:
            ratio = inlierRatio
            newIdx = i
            bestMatches = matches
            bestH = H
            bestStatus = status

    return (newIdx, bestMatches, bestH, bestStatus)


def detectAndDescribe(image):
    # convert image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # define image detector, using SIFT method
    siftDetector = cv2.FeatureDetector_create("SIFT")
    # get keyPoints vector
    keyPoints = siftDetector.detect(gray)

    # get image features from the image
    imageExtractor = cv2.DescriptorExtractor_create("SIFT")

    # get key points and image features
    (keyPoints, imageFeatures) = imageExtractor.compute(gray, keyPoints)

    # convert key points to float arrays
    keyPoints = np.float32([keyPoint.pt for keyPoint in keyPoints])

    # return keypoints and image features
    return (keyPoints, imageFeatures)


def matchKeyPoints(keyPointsA, keyPointsB, featuresA, featuresB, ratio=0.75, threshold=4.0):
    # compute the matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
    matches = []

    # loop over the raw matches
    for m in rawMatches:

        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:
        pointsA = np.float32([keyPointsA[i] for (i, _) in matches])
        pointsB = np.float32([keyPointsB[i] for (_, i) in matches])

        (H, status) = cv2.findHomography(pointsB, pointsA, cv2.RANSAC, threshold)
        return (matches, H, status)

    return None


def drawMatches(imageA, imageB, keyPointsA, keyPointsB, matches, status):
    # initialize
    (heightA, widthA) = imageA.shape[:2]
    (heightB, widthB) = imageB.shape[:2]

    matchImage = np.zeros((max(heightA, heightB), widthA + widthB, 3), dtype="uint8")
    matchImage[0:heightA, 0:widthA] = imageA
    matchImage[0:heightB, widthA:] = imageB

    # find matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):

        if s == 1:
            # draw the matches
            pointA = (int(keyPointsA[trainIdx][0]), int(keyPointsA[trainIdx][1]))
            pointB = (int(keyPointsB[queryIdx][0]) + widthA, int(keyPointsB[queryIdx][1]))
            cv2.line(matchImage, pointA, pointB, (0, 255, 0), 1)
    # return the matched image
    return matchImage

def findDimensions(image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0, 0]
    base_p2[:2] = [x, 0]
    base_p3[:2] = [0, y]
    base_p4[:2] = [x, y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)

        if (max_x == None or normal_pt[0, 0] > max_x):
            max_x = normal_pt[0, 0]

        if (max_y == None or normal_pt[1, 0] > max_y):
            max_y = normal_pt[1, 0]

        if (min_x == None or normal_pt[0, 0] < min_x):
            min_x = normal_pt[0, 0]

        if (min_y == None or normal_pt[1, 0] < min_y):
            min_y = normal_pt[1, 0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)



def stitch(imageA, imageB, matches, h, status, ratio=0.75, threshold=4.0):
    H = h / h[2, 2]
    H_inv = np.linalg.inv(H)


    (min_x, min_y, max_x, max_y) = findDimensions(imageB, H_inv)
    # Adjust max_x and max_y by base img size
    max_x = max(max_x, imageA.shape[1])
    max_y = max(max_y, imageA.shape[0])

    move_h = np.matrix(np.identity(3), np.float32)

    if (min_x < 0):
        move_h[0, 2] += -min_x
        max_x += -min_x

    if (min_y < 0):
        move_h[1, 2] += -min_y
        max_y += -min_y

    print "Homography: \n", H
    print "Inverse Homography: \n", H_inv
    print "Min Points: ", (min_x, min_y)

    mod_inv_h = move_h * H_inv

    img_w = int(math.ceil(max_x))
    img_h = int(math.ceil(max_y))

    print "New Dimensions: ", (img_w, img_h)

    base_img_warp = cv2.warpPerspective(imageA, move_h, (img_w, img_h))
    print "Warped base image"

    # utils.showImage(base_img_warp, scale=(0.2, 0.2), timeout=5000)
    # cv2.destroyAllWindows()

    next_img_warp = cv2.warpPerspective(imageB, mod_inv_h, (img_w, img_h))
    print "Warped next image"

    # utils.showImage(next_img_warp, scale=(0.2, 0.2), timeout=5000)
    # cv2.destroyAllWindows()

    # Put the base image on an enlarged palette
    enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

    print "Enlarged Image Shape: ", enlarged_base_img.shape
    print "Base Image Shape: ", imageA.shape
    print "Base Image Warp Shape: ", base_img_warp.shape

    # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
    # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp

    # Create a mask from the warped image for constructing masked composite
    (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY),
                                    0, 255, cv2.THRESH_BINARY)

    enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp,
                                mask=np.bitwise_not(data_map),
                                dtype=cv2.CV_8U)

    # Now add the warped image
    final_img = cv2.add(enlarged_base_img, next_img_warp,
                        dtype=cv2.CV_8U)

    # utils.showImage(final_img, scale=(0.2, 0.2), timeout=0)
    # cv2.destroyAllWindows()

    # Crop off the black edges
    final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print "Found %d contours..." % (len(contours))

    max_area = 0
    best_rect = (0, 0, 0, 0)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # print "Bounding Rectangle: ", (x,y,w,h)

        deltaHeight = h - y
        deltaWidth = w - x

        area = deltaHeight * deltaWidth

        if (area > max_area and deltaHeight > 0 and deltaWidth > 0):
            max_area = area
            best_rect = (x, y, w, h)

    if (max_area > 0):
        print "Maximum Contour: ", max_area
        print "Best Rectangle: ", best_rect

        final_img_crop = final_img[best_rect[1]:best_rect[1] + best_rect[3],
                         best_rect[0]:best_rect[0] + best_rect[2]]

        # utils.showImage(final_img_crop, scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()

        final_img = final_img_crop



    return final_img