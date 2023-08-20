# find correct points, using it to find angle
# -> find gesture, including the bicep curves. Write code to find angle between any three points
# create a method where can input three points, and gives the angle of three points
# base on the angle founded, calculate how many cur
import cv2
import numpy as np
import time
import poseDetection_Module as pm

# create object "detector"
detector = pm.poseDetector()

while True:
    img = cv2.imread("training video/1.jpg")
    img = detector.findPose(img, False)
    # get the landmarks values
    lmList = detector.findPosition(img, False)
    print(lmList)

    # make sure to have a list
    if len(lmList) != 0:
        # right arm
        detector.findAngle(img, 12, 14, 16)
        # right arm
        detector.findAngle(img, 11, 13, 15)

    cv2.imshow("Image", img)
    cv2.waitKey(1)