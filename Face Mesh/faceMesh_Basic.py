import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture("video/pexels-artem-podrez-8088627.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=4)
drawSpec = mpDraw.DrawingSpec(color=(0, 255, 0),thickness=1, circle_radius=1)
while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark): # 468 landmark
                #print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    cv.imshow('Image', img)
    cv.waitKey(1)
