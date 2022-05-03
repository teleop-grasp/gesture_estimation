#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from numpy_ros import to_message

from hand_tracking.tracking import hand_tracker, body_tracker
from hand_tracking.gesture_estimation import get_gesture
from hand_tracking.plot_utils import draw_hand_lms, Pose3DViewer
from hand_tracking.tracking import track_hands, track_body
from hand_tracking.pose_estimation import get_pose, calibrate

if __name__=="__main__":

	rospy.init_node("pose_gesture")

	# read node parameters
	params = rospy.get_param("~")
	topic_grasp_state = params["topic_grasp_state"]
	topic_pose_hand = params["topic_pose_hand"]
	topic_image_hand = params["topic_image_hand"]
	topic_image_raw = params["topic_image_raw"]
	visualize_tracking = params["visualize_tracking"]
	estimate_depth = params["estimate_depth"]
	ema_alpha_rot, ema_alpha_trans = params["ema_alpha"] # [R, t]

	# config
	LOOP_FREQ = 30 # hz
	HAND_TYPE = "right"
	hand_tracker.max_num_hands = 1
	hand_tracker.model_complexity = [0, 1][1]
	body_tracker.model_complexity = [0, 1, 2][2]

	# publishers
	pub_gesture = rospy.Publisher(topic_grasp_state, Bool, queue_size=1, tcp_nodelay=True)
	pub_pose = rospy.Publisher(topic_pose_hand, Pose, queue_size=1, tcp_nodelay=True)

	# subsriber to camera image
	img_msg = rospy.wait_for_message(topic_image_raw, Image)
	rospy.Subscriber(topic_image_raw, Image, lambda x: globals().update(img_msg=x))

	# setup visualization of hand + pose
	if visualize_tracking:

		# timer to plot 3D pose using matplotlib
		g_pose, g_hand = None, None
		plotter = Pose3DViewer()
		def callback_plot_pose(e): plotter.plot_pose(g_pose, g_hand)
		rospy.Timer(rospy.Duration(1./LOOP_FREQ), callback_plot_pose)

		# cv image w/ drawn hand
		pub_img_hand = rospy.Publisher(topic_image_hand, Image, queue_size=1, tcp_nodelay=True)

	# main loop
	bridge = CvBridge()
	rate = rospy.Rate(LOOP_FREQ) # hz
	while not rospy.is_shutdown():

		# convert to opencv format
		cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

		hands = track_hands(cv_img)
		body = track_body(cv_img)

		if HAND_TYPE in hands:

			# get desired hand and gesture
			hand = hands[HAND_TYPE]
			gesture = get_gesture(hand)

			# add hand landmarks to img
			cv_img = draw_hand_lms(cv_img, hand)

			if body is not None:
				pose = get_pose(hand, body, ema_alpha_trans, ema_alpha_rot, estimate_depth)
				pose_msg = to_message(Pose, pose)
				pub_pose.publish(pose_msg)

				if estimate_depth:
					rospy.logwarn_once("calibrating depth normalization...")
					rospy.logwarn_once("move your hand as far back and then as far forward as possible...")
					calibrate(pose, max_num_readings=100)

			if visualize_tracking:
				g_pose, g_hand = pose, hand
				pub_img_hand.publish(bridge.cv2_to_imgmsg(cv_img, encoding="bgr8"))

			pub_gesture.publish(gesture)

		rate.sleep()
