"""
Hand Tracking Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
    
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode        = self.mode, 
                                        max_num_hands            = self.maxHands, 
                                        min_detection_confidence = self.detectionCon, 
                                        min_tracking_confidence  = self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.length = -1

        self.volBar = -1
        self.volPer = -1
        
        self.pTime  = -1
        self.fps    = -1        

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if len(xList) > 0 and len(yList) > 0:
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                   (bbox[2] + 20, bbox[3] + 20), 
                                   (0, 255, 0), 2)
    
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, img, draw=True):
        coords = []
        p1 = 4 
        p2 = 8

        if len(self.lmList) > 0:
            x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
            x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
            if draw:
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
    
            self.length = math.hypot(x2 - x1, y2 - y1)
            box = [x1, y1, x2, y2, cx, cy]

        return self.length, img, coords
    
    def findVolume(self, img, draw=True):
        if (self.length > -1):
            self.volBar = np.interp(self.length, [50, 300], [400, 150])
            self.volPer = np.interp(self.length, [50, 300], [0, 100])
    
            if draw:
                cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
                cv2.rectangle(img, (50, int(self.volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, 
                            f'{int(self.volPer)} %', 
                            (40, 450), 
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (255, 0, 0),
                            3)    

    def findFPS(self, img, draw=True):
        if self.pTime == -1:
            self.pTime = time.time()
        else:
            cTime = time.time()
            self.fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            if draw:
                cv2.putText(img, 
                            f'FPS: {int(self.fps)}', 
                            (40, 50), 
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, 
                            (255, 0, 0), 
                            3)    