"""
posture.py

Shared posture/body-language feature extraction, pulled out of the
standalone step1b_face_and_pose.py prototype so recognize_and_log.py can
reuse it.

WHAT'S REUSED: only compute_posture_features() -- the engineered-feature
math (head_drop_px, torso_lean_px, torso_len_px). Unchanged from step1b.

WHAT'S NOT REUSED: step1b's Haar cascade face detector. recognize_and_log.py
already has face boxes from InsightFace every processed frame -- running a
second, older detector on top would duplicate work for no benefit.

IMPORTANT LIMITATION -- READ BEFORE CALLING get_posture_for_frame():
mediapipe's mp.solutions.pose detects exactly ONE person's skeleton per
frame, with no way to say which detected face that skeleton belongs to.
In a multi-person frame (which is the normal case for mess/cafeteria
scenes), there is no reliable way to know whose posture you just measured.

So get_posture_for_frame() only runs pose estimation when exactly one
face was detected in the frame this cycle -- otherwise it returns None
without running the model at all. This is a deliberate, cheap guard
against attributing posture to the wrong student, NOT a full solution.
True multi-person posture (MediaPipe's newer Pose Landmarker task API, or
running pose per face-crop) is a documented follow-up, not implemented
here.
"""

import cv2
import mediapipe as mp

_pose = None  # lazy-loaded singleton, same pattern as face_engine.get_face_app()


def get_pose_estimator():
    global _pose
    if _pose is None:
        mp_pose = mp.solutions.pose
        _pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _pose


def compute_posture_features(landmarks, frame_w, frame_h):
    """
    Engineered posture features from MediaPipe Pose landmarks.
    Unchanged from step1b_face_and_pose.py -- crude but a starting point;
    thresholds still need tuning against real footage.
    """
    mp_pose = mp.solutions.pose

    def px(lm_idx):
        lm = landmarks[lm_idx]
        return lm.x * frame_w, lm.y * frame_h

    try:
        l_shoulder = px(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        r_shoulder = px(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        nose = px(mp_pose.PoseLandmark.NOSE.value)
        l_hip = px(mp_pose.PoseLandmark.LEFT_HIP.value)
        r_hip = px(mp_pose.PoseLandmark.RIGHT_HIP.value)
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
    results = pose.process(rgb)

    if not results.pose_landmarks:
        return None

    h, w = frame.shape[:2]
    feats = compute_posture_features(results.pose_landmarks.landmark, w, h)
    return feats or None
