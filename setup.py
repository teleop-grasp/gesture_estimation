#!usr/bin/env python3

# python ROS package
# http://www.artificialhumancompanions.com/structure-python-based-ros-package/

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
	packages=["gesture_estimation"],
	package_dir={"": "src"},
	install_requires=[
		"opencv-python"
		"cvzone",
	]
)

setup(**setup_args)

