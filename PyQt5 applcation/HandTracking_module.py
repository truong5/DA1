import cv2 as cv
import mediapipe as mp
import time
import math


# create a class to use in all method
class handDetector():
    # create function, give in parameters that are required for this "Hands"
    def __init__(self, mode=False, maxHands = 2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        # create an object "self"
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        # initiate the object, give in the values
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        # a method from mediapipe to draw all the points(21)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    # create a method in class that will find the position -> give out the list
    def findHands(self, frame, draw=True): # put a "draw" flag,
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # calling the object hands, process the frame, give the result
        self.results = self.hands.process(imgRGB)
        # use the results and extract the information within
        # put in for loop to check it has multiple hands or not and extract one by one
        if self.results.multi_hand_landmarks:
            # get and extract for each hand landmarks in the results of "multi_hand_landmarks"
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame



    # create find positions method
    def findPosition(self, frame, handNo=0, draw=True):
        # create a landmark list to return, this list will have all the landmarks position
        self.lmList = []
        # check whether any hands, landmarks detected or not
        if self.results.multi_hand_landmarks:
            # point to a particular hand number
            myHand = self.results.multi_hand_landmarks[handNo]
            # to track on of the positions to perform a task, get the information in the hand
            for id, lm in enumerate(myHand.landmark):  # lm: landmark getting from the first element(first hand), and then within that hand it will get all the landmark, id number related to the exact index number of our finger landmark
                # for each of the hands, get the id number and landmarks information
                # for each landmarks information will gives x, y coordinates
                # use x,y coordinates to find information/location for the landmark on the hand
                h, w, c = frame.shape  # c: channel of the image
                # x,y are the ratio, multiply with the width and height and then will get the pixel value
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                # ìf true, it will draw
                if draw:
                    cv.circle(frame, (cx, cy), 7, (255, 0, 255), cv.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # totalFingers = fingers.count(1)
        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    # frame rates, write fps
    pTime = 0  # previous time
    cTime = 0  # current time
    capture = cv.VideoCapture(0)
    # create an "detector" object
    detector = handDetector() # use the default parameters above
    while True:
        success, frame = capture.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList) != 0:
            print(lmList[4])

        # display the fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)  #
        pTime = cTime
        # display the time on the screen, convert to string because its time, round the fps because we dont want decimal values, position, font, scale, color, thickness
        cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 3,
               (255, 0, 255), 3)

        cv.imshow('Video webcam', frame)
        cv.waitKey(1)

if __name__ == "__main__":
    main()