import os
import time
import cv2 as cv
import mediapipe as mp
import HandTracking_module as htm
import numpy as np
############################
brushThickness = 15
eraserThickness = 50
############################

cap = cv.VideoCapture(0)

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)


detector = htm.handDetector(detectionCon=0.85)
# declare the (x, y) previous position
xp, yp = 0, 0
# create image canvas to draw in
imgCanvas = np.zeros((480, 640, 3), np.uint8) # unsigned int 8: 255 values
while True:
    # 1. Import image
    success, img = cap.read()
    img = cv.flip(img, 1)
    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        # 4. If Selection Mode - Two fingers are up
        if fingers[1] and fingers[2]:

            print("Selection Mode")
            # checking for the click
            if y1 < 80:
                if 112<x1<187:
                    header = overlayList[1]
                    drawColor = (255, 0, 255)
                elif 233<x1<316:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 360<x1<450:
                    header = overlayList[3]
                    drawColor = (255, 0, 0)
                elif 475<x1<595:
                    header = overlayList[0]
                    drawColor = (0, 0, 0)
            cv.rectangle(img, (x1, y1), (x2, y2), drawColor, cv.FILLED)

        # 5. If Drawing Mode - Index finger is up,drawing lines
        if fingers[1] and fingers[2] == False:
            # draw the circle in the tip of the finger
            cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
            print("Drawing Mode")
            # starting to draw line, need a starting point and ending point
            # at the beginning, there are no previous point so it is(0,0)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            # starting to draw at previous position (xp, yp) to the new position (x1, y1)
            # draw on a blank image
            if drawColor == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            # update the previous point value
            xp, yp = x1, y1
    # setting header image
    img[0:80, 0:636] = header
    cv.imshow('Video', img)
    cv.imshow('Canvas', imgCanvas)

    cv.waitKey(1)