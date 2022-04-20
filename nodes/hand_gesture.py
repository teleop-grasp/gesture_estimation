#!/usr/bin/env python3
import cv2
from gesture_estimation import get_gesture


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

if __name__=='__main__':
    print("hello worlds roos")
    
    if not cap.isOpened():
        print(f"cant find video feed!!!!")
        
    while True:
        # Get image frame
        success, img = cap.read()
        gesture = get_gesture('', img)
        print(f"Current gesture: {gesture}")
