cmake_minimum_required(VERSION 3.12)
project(hand_tracking)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find catkin and any catkin packages
find_package(
	catkin REQUIRED
	COMPONENTS
		rospy
		sensor_msgs
		cv_bridge
)

# declare a catkin package
catkin_package(
	INCLUDE_DIRS
	LIBRARIES
		hand_tracking
	CATKIN_DEPENDS
		rospy
		sensor_msgs
		cv_bridge
		python3-opencv
		python3-mediapipe-pip
	DEPENDS
)

################################################################################################

# python package
# http://www.artificialhumancompanions.com/structure-python-based-ros-package/

## process setup.py
catkin_python_setup()

## install python nodes
catkin_install_python(
	PROGRAMS
		nodes/pose_gesture.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# install(
# 	PROGRAMS
# 		nodes/hand_tracking
# 	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
