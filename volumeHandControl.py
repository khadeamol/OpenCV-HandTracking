import cv2
import time
import numpy as np
import handTrackingModule as htm
import math
import osascript
########################################################################
wCam, hCam = 720,480
########################################################################



cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)
volMin = 0
volMax = 100
volBar = 250
osascript.osascript(f"set volume output volume {volMin}")
# applescript.applescript


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        
        
        cv2.circle(img, (x1,y1), 5, (0,0,255), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2, y2), (0,255,0), 2)
        cv2.circle(img,(cx,cy), 5, (0,255,0), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)

        vol = np.interp(length, [30,200], [volMin, volMax])
        volBar = np.interp(length, [30,200],[400,150])
        volPer = np.interp(length, [30,200],[0,100])
        print(length, f"  volume is {round(vol,2)}")


        osascript.osascript(f"set volume output volume {vol}")
        # cv2.putText(img, f"Volume is: {round(vol,2)}", (10,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255,155,240), 1)
        cv2.rectangle(img, (50,150), (85,400), (255,0,0),3)
        cv2.rectangle(img, (50,int(volBar)), (85,400), (255,0,0),cv2.FILLED)
        cv2.putText(img, f"{round(vol,2)}%", (50,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

        if length < 40:
            cv2.circle(img,(cx,cy), 5, (255,0,0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS:{int(fps)}", (10,50), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,0,0), thickness=2)

    cv2.imshow("img", img)
    cv2.waitKey(1)

