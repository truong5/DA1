import math

import cv2 as cv
import time
import numpy as np
import HandTracking_module as htm

# use pycaw library
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

#################################
wCam, hCam = 648, 488
#################################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

vol = 0
volBar = 400
volPer = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:

        x1, y1 = lmList[4][1], lmList[4][2] # the first element is x, second is y, 0 is the id (4 is id of top of the thump)
        x2, y2 = lmList[8][1], lmList[8][2] # (8 is id of top of the pointer)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # take the center of the line between 2 points
        cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (255, 120, 255), 3) # draw the line between 2 points
        cv.circle(img, (cx, cy), 15, (0, 255, 255), cv.FILLED)

        # calculate the length of the line
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 50:
            cv.circle(img, (cx, cy), 15, (255, 255, 255), cv.FILLED)

        # hand range 50 - 300, volume range -65 -> 0
        # convert 50-300 range to -65-0 range
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 150])

        print(vol)
        # change the volume by hand gesture
        volume.SetMasterVolumeLevel(vol, None)  # 0 set volume to max
    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)
    cv.putText(img, f'{int(volPer)}% ', (40, 450), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS:{int(fps)} ', (40, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    cv.imshow("img", img)
    cv.waitKey(1)