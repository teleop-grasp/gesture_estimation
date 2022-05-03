import numpy as np
import scipy
from scipy.spatial.transform import Rotation

HAND_LMS_INDICES = {"left": [15, 17, 19, 21], "right": [16, 18, 20, 22]}
CORRECT_Z_AXIS = 0.0
Z_SCALE_FACTOR = 1.0
R = Rotation.from_rotvec([0, np.pi/2, 0]).as_matrix() @ Rotation.from_rotvec([0, 0, np.pi/2]).as_matrix()

def static_var(**kwargs):
	def decorate(func):
		for att_name, att_val in kwargs.items():
			setattr(func, att_name, att_val)
		return func
	return decorate

def get_hand_lms(body, hand_type: str="left") -> tuple:
	lms_pose_hand = body["lms_pose"][HAND_LMS_INDICES[hand_type.lower()]]
	lms_pose_hand_world = body["lms_pose_world"][HAND_LMS_INDICES[hand_type.lower()]]

	return lms_pose_hand, lms_pose_hand_world

@static_var(wrist_trans_prev=np.zeros(3))
def get_translation(hand, body, ema_alpha=1.0, estimate_depth=False) -> np.ndarray:

	hand_type = hand["type"]
	hand_pose_img, hand_pose_world = get_hand_lms(body, hand_type=hand_type)
	hand_pose = hand_pose_img # which one to use

	lms_img = np.array(hand["lms_img"])
	wrist_trans = np.zeros(3)
	wrist_trans[0] = lms_img[0, 0]
	wrist_trans[1] = 1 - lms_img[0, 1] # to make it have origin at lower corner instead of upper left corner

	if CORRECT_Z_AXIS: # this will be set by calibrate()
		wrist_trans[2] = 1 - (hand_pose[0, 2] + CORRECT_Z_AXIS) * Z_SCALE_FACTOR # to have approx same range as x,y
	else:
		wrist_trans[2] = hand_pose[0, 2]

	# use depth estimation?
	wrist_trans[2] = wrist_trans[2] if estimate_depth else 0.5

	# transform points to match XYZ of frame in hand
	wrist_trans = R @ wrist_trans

	# exponential moving avg filter
	if ema_alpha < 1.0:
		# ema_alpha = np.array([0.05, ema_alpha, ema_alpha]) # separate filter for X-axis (depth)
		wrist_trans = ema_alpha * wrist_trans + (1 - ema_alpha) * get_translation.wrist_trans_prev
		get_translation.wrist_trans_prev = wrist_trans

	return wrist_trans

def get_pose(hand, body, ema_alpha_trans=1.0, ema_alpha_rot=1.0, estimate_depth=False) -> np.ndarray:
	translation = get_translation(hand, body, ema_alpha_trans, estimate_depth) # t
	orientation = get_orientation(hand, ema_alpha_rot) # R
	transformation = np.eye(4, dtype=np.float32) # [R, t]
	transformation[:3, :3] = orientation
	transformation[:3, 3] = translation

	return transformation

@static_var(quat_R_prev=Rotation.from_matrix(np.eye(3)).as_quat())
def get_orientation(hand, ema_alpha=1.0) -> np.ndarray:
	# https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
	lms_world = hand["lms_world"]
	lm_wrist = lms_world[0]
	lm_index = lms_world[5]

	z = -lm_wrist / np.linalg.norm(lm_wrist)
	y = gram_schmidt(lm_index, z)
	y = y / np.linalg.norm(y)
	x = np.cross(z, y)
	x = -x / np.linalg.norm(x)
	orientation = np.array([x,y,z]).T # rotation matrix

	if ema_alpha < 1.0:
		quat_R = Rotation.from_matrix(orientation).as_quat()
		quat_R = scipy.spatial.geometric_slerp(get_orientation.quat_R_prev, quat_R, ema_alpha)
		get_orientation.quat_R_prev = quat_R
		orientation = Rotation.from_quat(quat_R).as_matrix()

	return orientation

def gram_schmidt(v, u):
	return v - ((np.dot(v, u)/np.linalg.norm(u)**2) * u)

@static_var(pose_readings=[], reading_num=0)
def calibrate(pose, max_num_readings=100):
	global CORRECT_Z_AXIS, Z_SCALE_FACTOR # z-axis is the x-axis after transformations have been applied

	if calibrate.reading_num < max_num_readings:
		calibrate.pose_readings.append(pose)
		calibrate.reading_num += 1
	elif calibrate.reading_num == max_num_readings:
		pose_readings = np.array(calibrate.pose_readings)[:,0,3]
		min = np.min(pose_readings, axis=0)
		max = np.max(pose_readings, axis=0)
		print(f"FOUND MIN: {min}")
		print(f"FOUND MAX: {max}")
		if min < 0:
			CORRECT_Z_AXIS = -1 * min
			Z_SCALE_FACTOR = 1/ (max + CORRECT_Z_AXIS)
		calibrate.reading_num += 1
	else:
		return