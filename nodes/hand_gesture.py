#!/usr/bin/env python3
import cv2
from gesture_estimation import get_gesture
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from numpy_ros import to_message 
import numpy as np

if __name__=="__main__":

    rospy.init_node("hand_gesture")
    params = rospy.get_param("~")
    rospy.loginfo(params)
    pub_gesture = rospy.Publisher(params["topic_grasp_state"], Bool, queue_size=10)
    bridge = CvBridge()
    
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():

        # R = np.eye(4)
        # pose = to_message(Pose, R)
        # print(pose)

        # Get image msg
        img_msg = rospy.wait_for_message(params["topic_image_raw"], Image)

        # Convert to openCV format
        cv_img = bridge.imgmsg_to_cv2(img_msg)
        gesture = get_gesture("", cv_img)

        # print(f"Current gesture: {gesture}")
        pub_gesture.publish(gesture)
        rate.sleep()