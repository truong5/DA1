import cv2 as cv
import mediapipe as mp
import time

# create class
class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon


        # import mediapipe function to use face detection module
        self.mpFaceDetection = mp.solutions.face_detection
        # import drawing part
        self.mpDraw = mp.solutions.drawing_utils
        # initialize to use a face detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # create new list to store faces information
        bboxs = []
        # if the detections have result, the below code detect for face by faces
        if self.results.detections:
            # store the detections landmark points in numerous
            for id, detection in enumerate(self.results.detections):
                # store all the information in a bounding box, and extract the information
                bboxC = detection.location_data.relative_bounding_box
                # image height, weight, channel
                ih, iw, ic = img.shape
                # convert to pixels value, so that bounding box will have x, y, width, height
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                       int(bboxC.width * iw), int(bboxC.height * ih)
                # send in id, bbox, score
                bboxs.append([id, bbox, detection.score])
                if draw:
                    self.fancyDraw(img, bbox)
                    # show the confidence value
                    cv.putText(img, f'{int(detection.score[0]*100)}%',
                               (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_SIMPLEX,
                               1, (255, 0, 0), 2)
        # return bounding box information, id number, score
        return img, bboxs
    # draw corner's lines, "l" is length of the line, "t" is the thickness of line, "rt" is rectangle thickness
    def fancyDraw(self, img, bbox, l=30, t=5, rt=5):
        # extract the information
        x, y, w, h = bbox
        # x1, y1 bottom right point / x ,y original point(top left)
        x1, y1 = x + w, y + h
        cv.rectangle(img, bbox, (255, 0, 255), rt)
        # top left x, y
        cv.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # top right x1, y
        cv.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # bottom left x, y
        cv.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # bottom right x1, y1
        cv.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img

def main():
    cap = cv.VideoCapture('video/video1.mp4')
    pTime = 0
    # create an object from the class
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow('image', img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()