# find correct points, using it to find angle
# -> find gesture, including the bicep curves. Write code to find angle between any three points
# create a method where can input three points, and gives the angle of three points
# base on the angle founded, calculate how many curl
import cv2
import numpy as np
import time
import poseDetection_Module as pm

cap = cv2.VideoCapture("training video/production ID_4259068.mp4")

detector = pm.poseDetector()
count = 0 # count the curl
dir = 0
pTime = 0
while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if len(lmList) != 0:
        # detector.findAngle(img, 12, 14, 16)
        # right arm
        angle = detector.findAngle(img, 11, 13, 15)
        # tranfer the ratio from (210, 310) to (0-100) percent
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (210, 310), (650, 100))

        # check for the dumbbell curls
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1 # to know which direction we are moving
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)
        # draw bar
        cv2.rectangle(img, (1110, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1110, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 5,
                    color, 5)
        # draw curl count
        cv2.rectangle(img, (0, 600), (100, 700), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (20, 670), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50, 120), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)