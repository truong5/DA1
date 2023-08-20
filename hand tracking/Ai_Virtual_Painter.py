import os
import cv2 as cv
import mediapipe as mp
import HandTracking_module as htm
import numpy as np

############################
brushThickness = 15
eraserThickness = 100
############################

cap = cv.VideoCapture(0)

# read every image in the folder "header", import to the list "myList"
folderPath = "header"
myList = os.listdir(folderPath)
# print the list of "image" in the header folder
print(myList)
overlayList = []
for imPath in myList:
    # read every image
    image = cv.imread(f'{folderPath}/{imPath}')
    # insert image to the overlayList list, where we want to overlay
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

# create object detector
detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)
while True:
    # 1. Import image
    success, img = cap.read()
    # flip the image
    img = cv.flip(img, 1)
    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # tipID of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        # 4. If Selection Mode - Two fingers are up
        if fingers[1] and fingers[2]:
            # whenever we have a selection, set the previous values to 0, instead of random value
            xp, yp = 0, 0
            print("Selection Mode")
            # checking for the click
            if y1 < 80:
                if 112 < x1 < 187:
                    header = overlayList[1]
                    drawColor = (255, 0, 255)
                elif 233 < x1 < 316:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 360 < x1 < 450:
                    header = overlayList[3]
                    drawColor = (255, 0, 0)
                elif 475 < x1 < 595:
                    header = overlayList[0]
                    drawColor = (0, 0, 0)
            cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1
    # create grey image
    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    # convert into binary image(black and white), reversing it (color->black, black background->white)
    # -> create a mask with all this white, only the drew region is black
    # in the original video, make all the drew line -> black, merge the "imgCanvas" with the video
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    # convert back to the original element
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    # add the original image with the inverse image
    img = cv.bitwise_and(img, imgInv) # => in the video, its show the black region
    # add the original video with image canvas(which has the color)
    img = cv.bitwise_or(img, imgCanvas)


    # setting header image by slicing it
    img[0:80, 0:636] = header
    # add two image and blend them
    img = cv.addWeighted(img, 0.5, imgCanvas, 0.1, 0)
    cv.imshow('Video', img)
    cv.imshow('Canvas', imgCanvas)
    cv.imshow("Inv", imgInv)

    cv.waitKey(1)