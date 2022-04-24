#!/usr/bin/env python3
import cv2
from gesture_estimation import get_gesture
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

if __name__=="__main__":

    rospy.init_node("hand_gesture")
    params = rospy.get_param("~")
    rospy.loginfo(params)
    pub_gesture = rospy.Publisher(params["topic_grasp_state"], Bool, queue_size=10)

    # if not cap.isOpened():
    #     print(f"cant find video feed!!!!")
        
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        # Get image frame
        img = rospy.wait_for_message(params["topic_image_raw"], Image)
        
        gesture = get_gesture("", img)
        print(f"Current gesture: {gesture}")
        pub_gesture.publish(gesture)
        rate.sleep()