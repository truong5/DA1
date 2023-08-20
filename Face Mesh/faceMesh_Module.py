import cv2 as cv
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, static_mode=False,
               max_faces=4,
               refineLandmarks=False,
               detection_confidence=0.5,
               tracking_confidence=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.refineLandmarks = refineLandmarks
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_mode, self.max_faces, self.refineLandmarks, self.detection_confidence, self.tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0),thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):

        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        # many faces
        # store many face
        faces = []
        if self.results.multi_face_landmarks:
            for faceid, faceLms in enumerate(self.results.multi_face_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                # for every face go through every landmark, convert it into x, y and store in "face" list
                face = []
                for id, lm in enumerate(faceLms.landmark): # 468 landmark
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # replace from landmarks point to landmarks id
                    if faceid ==1:
                        cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.18, (0, 255, 0), 1)
                    else:
                        cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.18, (0, 0, 255), 1)
                    print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces, faceid

def main():
    cap = cv.VideoCapture("video/video1.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces,faceid = detector.findFaceMesh(img)
        if len(faces)!= 0:
            print(faceid) # print all the point:print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()