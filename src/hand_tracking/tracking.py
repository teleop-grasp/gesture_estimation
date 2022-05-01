import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation

mp_body_pose = mp.solutions.pose

body_tracker = mp_body_pose.Pose(    
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


def track_body(img):
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = body_tracker.process(img)
      
    body = {'lms_pose': None, 'lms_pose_world': None}

    if results.pose_landmarks:
        body['lms_pose'] = np.array([(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark])
        body['lms_pose_world'] = np.array([(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_world_landmarks.landmark])
        return body
    return None




mp_hands = mp.solutions.hands
hand_tracker = mp_hands.Hands(static_image_mode=False, 
                        max_num_hands=2,
                        model_complexity=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)
        
R = Rotation.from_euler('yxz', [-np.pi/2, -np.pi/2,  0]).as_matrix()

def track_hands(img, flip_type=True) -> dict:
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hand_tracker.process(img)
  
    hands = {}
    if not results.multi_handedness:
        return hands

    for handness, hand_landmarks, hand_world_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks, results.multi_hand_world_landmarks):        
        hand_type = handness.classification[0].label.lower()
        hand_type = hand_type if not flip_type else ("left" if hand_type == "right" else "right") # Flip for mirrored view for mediapipe 
                                         
        hand_img = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
        lms_world = np.array([(landmark.x, landmark.y, landmark.z) for landmark in hand_world_landmarks.landmark])
        lms_world = (R @ lms_world.T).T
        
        hand = {}
        hand['lms_media_pipe'] = hand_landmarks
        hand['lms_img'] = hand_img
        hand['lms_world'] = lms_world
        hand['type'] = hand_type

        hands[hand_type] = hand

    return hands      