import cv2
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(detectionCon=0.5, maxHands=2)

LEFT_HAND = 'Left'
RIGHT_HAND = 'Right'


def get_gesture(hand_type: str, img) -> bool:
	hands, img = detector.findHands(img)
	for hand in hands:
		handtype = hand['type']        
		
		fingers_up = detector.fingersUp(hand)

		if sum(fingers_up) > 4:
			return True
		else:
			return False
