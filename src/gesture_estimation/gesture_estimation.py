import cv2
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(detectionCon=0.5, maxHands=2)

LEFT_HAND = 'Left'
RIGHT_HAND = 'Right'

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    if not cap.isOpened():
        print(f"cant find video feed!!!!")
    while True:
        # Get image frame
        success, img = cap.read()

        # Find the hand and its landmarks
        hands, img = detector.findHands(img)
        
        # Display
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        hand_information = {}
        for hand in hands:
            handtype = hand['type']        
            landmarks = hand['lmList']
            bbox = hand['bbox']
            center_point = hand['center']
            
            fingers_up = detector.fingersUp(hand)

            if sum(fingers_up) > 4:
                print(f"{handtype}: Hand opened")
            else:
                print("Hand Closed.")
            # print(landmarks[1])
            # for landmark in landmarks:
                # print(f"landmark is: {landmark}")
            hand_information[handtype] = {'type': handtype, 'landmarks': landmarks, 'bbox': bbox, 'center_point': center_point, 'fingers_up': fingers_up}
            print(f"{hand_information}")
            # input()


def get_gesture(hand_type: str, img) -> bool:
    hands, img = detector.findHands(img)
    for hand in hands:
        handtype = hand['type']        
        # landmarks = hand['lmList']
        # bbox = hand['bbox']
        # center_point = hand['center']
        
        fingers_up = detector.fingersUp(hand)

        if sum(fingers_up) > 4:
            return True
        else:
            return False
