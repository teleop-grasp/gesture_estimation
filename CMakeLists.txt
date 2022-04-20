cmake_minimum_required(VERSION 3.12)
project(gesture_estimation)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find catkin and any catkin packages
find_package(
	catkin REQUIRED
	COMPONENTS
		rospy
)

# find system libraries

# declare a catkin package
catkin_package(
	INCLUDE_DIRS
		
	LIBRARIES
		gesture_estimation
	CATKIN_DEPENDS
		rospy
	DEPENDS

)

################################################################################################

# process setup.py

catkin_python_setup()

################################################################################################

# install python nodes

install(
	PROGRAMS
		nodes/hand_gesture
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
