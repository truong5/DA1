import cv2 as cv
import mediapipe as mp
import time
import math

# create a class to detect pose and find all the points
class poseDetector():
    # initialization, write in the parameters that are required
    def __init__(self, mode=False,modelComplexity=1, enableSegmentation=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.enableSegmentation = enableSegmentation
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComplexity, self.enableSegmentation, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    # find the points
    def findPosition(self, img, draw=True):
        # put all the points founded in the list
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # append the id, x,y coordinates after multiply with the width and height
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 10, (255, 0 ,0), cv.FILLED)
        return self.lmList

    # get the angle
    def findAngle(self, img, p1, p2, p3, draw=True):
        # slicing, only take the coordinate values
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # calculate the angle, transfer to degree
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        # if the angle is negative
        if angle < 0:
            angle += 360
        print(angle)
        # draw the angle between elbow, shoulder, wrist landmarks
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv.line(img, (x3, y3), (x2, y2), (0, 255, 0), 3)
            cv.circle(img, (x1, y1), 10, (255, 0, 0), cv.FILLED)
            cv.circle(img, (x2, y2), 10, (255, 0, 0), cv.FILLED)
            cv.circle(img, (x3, y3), 10, (255, 0, 0), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (255, 0, 0))
            cv.circle(img, (x2, y2), 15, (255, 0, 0))
            cv.circle(img, (x3, y3), 15, (255, 0, 0))
            cv.putText(img, str(int(angle)), (x2 - 55, y2 + 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return angle

def main():
    capture = cv.VideoCapture("video/pexels-tima-miroshnichenko-6390399.mp4")
    pTime = 0
    # write the detector, give in default parameter
    detector = poseDetector()
    while True:
        success, img = capture.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        # checking if the list is filled or not, sometime the first frame were unable to detect
        if len(lmList) != 0:
            print(lmList[14])
        # draw  landmarks no.14
            cv.circle(img, (lmList[14][1], lmList[14][2]), 10, (0, 0, 255), cv.FILLED)
        # check frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (100, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
        cv.imshow('image', img)

        cv.waitKey(1)


if __name__ == "__main__":
    main()
