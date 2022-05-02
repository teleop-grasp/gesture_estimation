#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from numpy_ros import to_message

from hand_tracking.gesture_estimation import get_gesture
from hand_tracking.plot_utils import draw_hand_lms, Pose3DViewer
from hand_tracking.tracking import track_hands, track_body
from hand_tracking.pose_estimation import get_pose

if __name__=="__main__":

	rospy.init_node("pose_gesture")

	# read node parameters
	params = rospy.get_param("~")
	topic_grasp_state = params["topic_grasp_state"]
	topic_pose_hand = params["topic_pose_hand"]
	topic_image_hand = params["topic_image_hand"]
	topic_image_raw = params["topic_image_raw"]
	visualize_tracking = params["visualize_tracking"]

	pub_gesture = rospy.Publisher(topic_grasp_state, Bool, queue_size=1, tcp_nodelay=True)
	pub_pose = rospy.Publisher(topic_pose_hand, Pose, queue_size=1, tcp_nodelay=True)

	if visualize_tracking:
		pub_img_hand = rospy.Publisher(topic_image_hand, Image, queue_size=1)

	bridge = CvBridge()
	hand_type = "right"
	Pose3DViewer = Pose3DViewer()

	rate = rospy.Rate(30) # hz
	while not rospy.is_shutdown():
		# get image msg
		img_msg = rospy.wait_for_message(topic_image_raw, Image)

		# convert to opencv format
		cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

		hands = track_hands(cv_img)
		body = track_body(cv_img)

		if hand_type in hands:
			# get desired hand
			hand = hands[hand_type]

			gesture = get_gesture(hand)

			# add hand landmarks to img
			cv_img = draw_hand_lms(cv_img, hand)

			if body is not None:
				pose = get_pose(hand, body)
				pose_msg = to_message(Pose, pose)
				pub_pose.publish(pose_msg)

				if visualize_tracking:
					Pose3DViewer.plot_pose(pose, hand)

			pub_gesture.publish(gesture)

			if visualize_tracking:
				pub_img_hand.publish(bridge.cv2_to_imgmsg(cv_img, encoding="bgr8"))

		rate.sleep()
