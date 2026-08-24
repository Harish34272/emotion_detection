import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# Path to the downloaded .task model file. Adjust if you put it elsewhere.
_MODEL_PATH = os.path.join(os.path.dirname(__file__),  "pose_landmarker_lite.task")

# Landmark indices are unchanged from the old mp.solutions.pose.PoseLandmark enum --
# the Tasks API kept the same 33-point ordering, it just dropped the enum wrapper.
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

_pose = None  # lazy-loaded singleton, same pattern as face_engine.get_face_app()


def get_pose_estimator():
    global _pose
    if _pose is None:
        base_options = BaseOptions(model_asset_path=_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _pose = vision.PoseLandmarker.create_from_options(options)
    return _pose


def compute_posture_features(landmarks, frame_w, frame_h):
    """
    Engineered posture features from MediaPipe Pose landmarks.
    Unchanged logic from step1b_face_and_pose.py -- crude but a starting point;
    thresholds still need tuning against real footage.
    """
    def px(lm_idx):
        lm = landmarks[lm_idx]
        return lm.x * frame_w, lm.y * frame_h

    try:
        l_shoulder = px(LEFT_SHOULDER)
        r_shoulder = px(RIGHT_SHOULDER)
        nose = px(NOSE)
        l_hip = px(LEFT_HIP)
        r_hip = px(RIGHT_HIP)
    except Exception:
        return {}

    shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
    shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
    hip_mid_y = (l_hip[1] + r_hip[1]) / 2
    hip_mid_x = (l_hip[0] + r_hip[0]) / 2

    head_drop = nose[1] - shoulder_mid_y      # smaller/negative = head lower relative to shoulders
    torso_lean = shoulder_mid_x - hip_mid_x   # horizontal shoulder-vs-hip offset
    torso_len = abs(hip_mid_y - shoulder_mid_y)  # for normalizing other features by body scale

    return {
        "head_drop_px": round(head_drop, 1),
        "torso_lean_px": round(torso_lean, 1),
        "torso_len_px": round(torso_len, 1),
    }


def get_posture_for_frame(frame, num_faces_in_frame):
    """
    Returns a posture features dict, or None if posture shouldn't be
    computed this call.

    num_faces_in_frame should be the TOTAL number of faces InsightFace
    found this cycle (matched + unmatched) -- not just matched count.
    Even one unmatched extra person in frame makes attribution ambiguous,
    so this deliberately errs conservative.

    Returns None (not {}) in three distinct skip cases -- callers don't
    need to distinguish them, but it's worth knowing while debugging:
      - more than one face in frame (ambiguous attribution, model never runs)
      - pose model ran but found no landmarks (person's body not in frame)
      - landmarks found but a required joint was missing (compute_posture_features failed)
    """
    if num_faces_in_frame != 1:
        return None

    pose = get_pose_estimator()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = pose.detect(mp_image)

    if not results.pose_landmarks:
        print("  [posture debug] pose model found no landmarks in frame")
        return None

    h, w = frame.shape[:2]
    landmarks = results.pose_landmarks[0]  # first (only) detected person
    feats = compute_posture_features(landmarks, w, h)
    if not feats:
        print("  [posture debug] landmarks found but a required joint was missing/low-confidence")
    return feats or None
