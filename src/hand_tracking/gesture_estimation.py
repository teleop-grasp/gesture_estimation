def fingers_up(hand):
	"""
	Inspired by  By: Computer Vision Zone
	Website: https://www.computervision.zone/

	Finds how many fingers are open and returns in a list.
	Considers left and right hands separately
	:return: List of which fingers are up
	"""

	TIP_IDS = [4, 8, 12, 16, 20]

	hand_type = hand["type"]
	lms_img = hand["lms_img"]

	fingers = []
	# thumb
	if hand_type == "right":
		if lms_img[TIP_IDS[0]][0] > lms_img[TIP_IDS[0] - 1][0]:
			fingers.append(1)
		else:
			fingers.append(0)
	else:
		if lms_img[TIP_IDS[0]][0] < lms_img[TIP_IDS[0] - 1][0]:
			fingers.append(1)
		else:
			fingers.append(0)

	# 4 fingers
	for id in range(1, 5):
		if lms_img[TIP_IDS[id]][1] < lms_img[TIP_IDS[id] - 2][1]:
			fingers.append(1)
		else:
			fingers.append(0)

	return fingers

def get_gesture(hand) -> bool:
	return sum(fingers_up(hand)) > 4