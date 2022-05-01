
import numpy as np

HAND_LMS_INDICES = {'left': [15, 17, 19, 21], 'right': [16, 18, 20, 22]}

def get_hand_lms(body, hand_type: str='left') -> tuple:    
    lms_pose_hand = body['lms_pose'][HAND_LMS_INDICES[hand_type.lower()]]
    lms_pose_hand_world = body['lms_pose_world'][HAND_LMS_INDICES[hand_type.lower()]]
    return lms_pose_hand, lms_pose_hand_world

def get_translation(hand, body) -> np.ndarray:
    hand_type = hand['type']
    _, hand_pose_world = get_hand_lms(body, hand_type=hand_type)
    lms_img = np.array(hand['lms_img'])
    wrist_trans = np.zeros(3)
    wrist_trans[0:2] = lms_img[0, 0:2]
    wrist_trans[2] = hand_pose_world[0, 2]
    
    # print(f"Trans: {wrist_trans}")
    return wrist_trans


def get_pose(hand, body) -> np.ndarray:
    translation = get_translation(hand, body)
    orientation = get_orientation(hand)
    transformation = np.eye(4, dtype=np.float32)
    transformation[:3, :3] = orientation
    transformation[:3, 3] = translation
    return transformation


def get_orientation(hand) -> np.ndarray:
    lms_world = hand['lms_world']
    # https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
    
    lm_wrist = lms_world[0]    
    lm_index = lms_world[5]
    z = -lm_wrist / np.linalg.norm(lm_wrist)
    y = gram_schmidt(lm_index, z)
    y = y / np.linalg.norm(y)
    x = np.cross(z, y)
    x = -x / np.linalg.norm(x)
    orientation = np.array([x,y,z]).T
    # det = np.linalg.det(orientation)
    # print(f"determinant {det}")
    return orientation


def gram_schmidt(v, u):
    return v - ((np.dot(v, u)/np.linalg.norm(u)**2) * u)
