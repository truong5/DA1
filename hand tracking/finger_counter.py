import cv2 as cv
import time
import os
import HandTracking_module as htm
wCam, hCam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
# import image, store them 1 by 1
folderPath = "finger image"
# list all the file present in the "finger image"
myList = os.listdir(folderPath)
# print all the images inside
print(myList)
# create list of image
overlayList = []
# loop through the list
for imPath in myList:
    # read all the image, import it
    image = cv.imread(f'{folderPath}/{imPath}')
    # save every image into the list
    overlayList.append(image)
# print(len(overlayList)) # print the length of the list
pTime = 0

detector =htm.handDetector(detectionCon=0.75)
# basically the top of every finger
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    print(lmList)

    if len(lmList) != 0:
        fingers = []
        # works with right hand only
        # thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 fingers
        for id in range(1, 5):
            # get the y element. ex: in a finger,according to the y axis, if the (tipId) higher than (tipId -2)-> the finger is up
            # using opencv orientation
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # get the total fingers up
        totalFingers = fingers.count(1)
        print(totalFingers)

        # display the fingers picture by overlay it in the video
        # define our new image, put in the old image(video) base on this location
        h, w, c = overlayList[totalFingers-1].shape # store the height and width of the finger image in h and w
        # target a region in the image, give in the range of height and width
        img[0:h, 0:w] = overlayList[totalFingers-1]
        # change the finger image
        cv.rectangle(img, (20,255), (170, 425), (0, 255, 0), cv.FILLED)
        cv.putText(img, str(totalFingers), (45, 400),cv.FONT_HERSHEY_SIMPLEX, 5, (255, 0 , 0), 25)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(img, f'FPS: {int(fps)}', (400, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0 ,0), 3)

    cv.imshow('Image', img)
    cv.waitKey(1)

