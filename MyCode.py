import cv2 
import mediapipe 
import time
import math 
import HandTrackingModule
import VolumeControlModule 

def main():
    camera = cv2.VideoCapture(0)
    detector = HandTrackingModule.handDetector()
    volctl = VolumeControlModule.volumeControl()

    for i in range(0, 1000000000000):
        print("Writing image " + str(i) + "...")

        success, image = camera.read()

        detector.findHands(image)
        detector.findPosition(image)
        detector.findDistance(image)
        detector.findVolume(image)
        detector.findFPS(image)

        volctl.setVolume(detector.volPer)

        cv2.imshow("LiveView", image)
        cv2.waitKey(10)





main()

