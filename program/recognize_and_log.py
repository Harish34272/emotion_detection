import os
from dotenv import load_dotenv
load_dotenv()
DEFAULT_CAMERA = os.environ.get("DEFAULT_CAMERA", 0)

# --- LOAD ORDER IS LOAD-BEARING: do not move these below the cv2/deepface imports ---
# TensorFlow (pulled in by deepface) statically links BoringSSL, which exports
# OpenSSL symbol names with incompatible struct layouts. libpq binds to those
# during SCRAM auth and segfaults. psycopg2 must connect before TF is loaded.
# Verified 2026-08-13: sslmode=disable, gssencmode=disable and LD_PRELOAD of
# system libssl/libcrypto 3 all still segfault. Import order is the only fix.
from sqlalchemy import text
from db import get_session
from models import Student, FaceEmbedding, Camera, DetectionEvent

_session = get_session()
_session.execute(text("SELECT 1"))  # force the real psycopg2 connect to happen NOW
_session.commit()                   # close the transaction, keep the pooled connection
# --- end load-order block ---

import argparse
import time
import cv2
from deepface import DeepFace
from face_engine import get_faces, cosine_distance, open_capture, resize_for_display
from posture import get_posture_for_frame
DISTANCE_THRESHOLD = 0.68  # tune after testing on real enrolled faces (InsightFace cosine distance)
PROCESS_EVERY_N_FRAMES = 5  # InsightFace is fast enough to sample more often than the DeepFace version
LOG_COOLDOWN_SECONDS = 60   # don't log the same student twice within this window
FACE_CROP_PADDING = 0.15    # extra margin around bbox, as a fraction of face size -- DeepFace's
                             # emotion model expects a bit of context around the face, not a tight crop


def load_known_embeddings(session):
    """Pulls all (student_id, embedding, angle_label) rows from the DB into memory."""
    rows = session.query(FaceEmbedding).all()
    known = [(r.student_id, r.embedding, r.angle_label) for r in rows]
    print(f"Loaded {len(known)} embeddings for {len(set(s for s, _, _ in known))} students.")
    return known


def match_face(embedding, known_embeddings):
    best_student_id, best_distance, best_angle = None, float("inf"), None
    for student_id, known_emb, angle_label in known_embeddings:
        d = cosine_distance(embedding, known_emb)
        if d < best_distance:
            best_distance = d
            best_student_id = student_id
            best_angle = angle_label

    if best_distance > DISTANCE_THRESHOLD:
        return None, best_distance, None
    return best_student_id, best_distance, best_angle


def crop_face_with_padding(frame, bbox):
    """
    Crops the face region out of the full frame with a small margin added,
    clipped to stay inside the frame bounds. DeepFace's emotion model was
    trained on images with a bit of context around the face (forehead/chin/
    ears visible), so a tight bbox-only crop tends to hurt its accuracy.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)

    face_w, face_h = x2 - x1, y2 - y1
    pad_x = int(face_w * FACE_CROP_PADDING)
    pad_y = int(face_h * FACE_CROP_PADDING)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None  # degenerate crop, e.g. bbox right at frame edge
    return frame[y1:y2, x1:x2]


# --- Emotion quality gate ---------------------------------------------------
# Facial emotion classifiers (DeepFace included) are trained overwhelmingly on
# frontal/near-frontal faces. Feeding them a profile or steep angle doesn't
# fail loudly -- it just returns low-confidence noise that LOOKS like a
# normal label (e.g. a 4-way near-tie that happens to rank 'sad' on top).
# This gate skips emotion scoring on faces that are too angled/low-quality to
# trust, WITHOUT affecting identity recognition or logging, which still run
# on every detected face regardless of angle -- attendance/routine is the
# primary signal and doesn't need a frontal face.
MIN_DET_SCORE_FOR_EMOTION = 0.65   # InsightFace's own detection confidence
MAX_YAW_RATIO_FOR_EMOTION = 0.35   # proxy for head turn, see is_face_frontal_enough()


def is_face_frontal_enough(face):
    """
    Decides whether a detected face is frontal/clear enough to trust for
    emotion classification. Uses two signals from InsightFace, both free
    (already computed during detection, no extra inference cost):

    1. det_score -- InsightFace's own confidence that this is a clean,
       well-formed face. Angled/partial/occluded faces tend to score lower.
    2. Yaw proxy from the 5-point landmarks (kps): eyes, nose, mouth
       corners. On a frontal face the nose sits roughly midway between the
       two eyes horizontally. As the head turns, the nose shifts toward one
       eye and away from the other -- we measure that asymmetry as a ratio
       of eye-to-eye distance, which is scale-invariant (works regardless
       of how close/far the face is from the camera).

    Returns True if the face passes both checks (safe to run emotion on),
    False otherwise (skip emotion, still log identity as normal).
    """
    det_score = getattr(face, "det_score", None)
    if det_score is not None and det_score < MIN_DET_SCORE_FOR_EMOTION:
        print(f"    [gate] det_score={det_score:.3f} -> REJECT (below {MIN_DET_SCORE_FOR_EMOTION})")
        return False

    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 3:
        # no landmarks to check yaw with -- fall back to det_score alone
        return True

    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = abs(right_eye[0] - left_eye[0])
    if eye_dist < 1e-3:
        print(f"    [gate] det_score={det_score} eye_dist~0 -> REJECT (degenerate)")
        return False  # degenerate, eyes on top of each other -- bad detection

    eye_mid_x = (left_eye[0] + right_eye[0]) / 2
    yaw_ratio = abs(nose[0] - eye_mid_x) / eye_dist

    passed = yaw_ratio <= MAX_YAW_RATIO_FOR_EMOTION
    verdict = "PASS" if passed else "REJECT"
    print(f"    [gate] det_score={det_score:.3f} yaw_ratio={yaw_ratio:.3f} -> {verdict}")
    return passed
# -----------------------------------------------------------------------------


def get_emotion(face_crop):
    """
    Runs DeepFace's emotion classifier on an already-cropped face image.
    detector_backend='skip' tells DeepFace not to re-run its own face
    detector -- we already know this crop is a face, from InsightFace.
    Returns (label, confidence, full_scores_dict) or (None, None, None) if it fails.
    full_scores_dict is the raw per-class 0-100 breakdown from DeepFace, kept
    for debugging -- e.g. checking how close 'sad' got to beating 'neutral'.
    """
    if face_crop is None or face_crop.size == 0:
        return None, None, None
    try:
        result = DeepFace.analyze(
            img_path=face_crop,
            actions=["emotion"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        dominant = result["dominant_emotion"]
        full_scores = result["emotion"]  # dict of label -> 0-100 score, all 7 classes
        confidence = full_scores[dominant] / 100.0  # normalize to 0-1
        return dominant, confidence, full_scores
    except Exception as e:
        print(f"  (emotion detection failed: {e})")
        return None, None, None


# --- DEBUG INSTRUMENTATION (temporary -- for diagnosing the sad-detection
# issue. Remove or gate behind a flag once root cause is found; this is not
# the planned source_photos/detections/ feature, just throwaway debug output) ---
DEBUG_SAVE_EMOTION_CROPS = True
DEBUG_CROP_DIR = "source_photos/detections"


def save_debug_crop(face_crop, student_label, full_scores, dominant, confidence):
    """
    Saves the exact crop that was fed to DeepFace, plus a sidecar .txt with
    the full per-class score breakdown, so we can visually check crop
    quality alongside the numbers that produced the label.
    """
    if not DEBUG_SAVE_EMOTION_CROPS or face_crop is None or face_crop.size == 0:
        return
    safe_label = str(student_label).replace("/", "_").replace(" ", "_")
    student_dir = os.path.join(DEBUG_CROP_DIR, safe_label)
    os.makedirs(student_dir, exist_ok=True)

    ts = int(time.time() * 1000)
    img_path = os.path.join(student_dir, f"{ts}.jpg")
    cv2.imwrite(img_path, face_crop)

    txt_path = os.path.join(student_dir, f"{ts}.txt")
    with open(txt_path, "w") as f:
        f.write(f"dominant: {dominant} ({confidence:.3f})\n")
        f.write("full scores:\n")
        if full_scores:
            for label, score in sorted(full_scores.items(), key=lambda kv: -kv[1]):
                f.write(f"  {label}: {score:.2f}\n")
    print(f"  [debug] saved crop + scores -> {img_path}")


STALE_EMOTION_SECONDS = 10  # beyond this, tag the label as stale rather than showing it as live


def build_label(student, last_emotion, now):
    """
    Builds the on-screen label text: name, plus the most recently known
    emotion reading for that student if one exists. Readings older than
    STALE_EMOTION_SECONDS get a "(Ns ago)" tag so a testing minute-old
    value isn't mistaken for a live one.
    """
    label_text = student.name
    cached = last_emotion.get(student.student_id)
    if cached:
        emotion_label, _, ts = cached
        age = now - ts
        if age <= STALE_EMOTION_SECONDS:
            label_text += f" - {emotion_label}"
        else:
            label_text += f" - {emotion_label} ({int(age)}s ago)"
    return label_text


def get_or_create_camera(session, location_name):
    cam = session.query(Camera).filter_by(location_name=location_name).first()
    if not cam:
        cam = Camera(location_name=location_name)
        session.add(cam)
        session.commit()
        print(f"Created camera entry for location '{location_name}' (id={cam.camera_id})")
    return cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-location", required=True, help="e.g. mess_entry, cafeteria_entry")
    parser.add_argument("--source", default=DEFAULT_CAMERA,
                         help="camera index, RTSP URL, or 'gst:<pipeline>' for a raw GStreamer pipeline")
    parser.add_argument("--show-labels", action="store_true",
                         help="overlay name+emotion on the live feed (dev/testing only — "
                              "do not use where the screen could be seen by others)")
    args = parser.parse_args()

    session = _session
    camera = get_or_create_camera(session, args.camera_location)
    known_embeddings = load_known_embeddings(session)

    if not known_embeddings:
        print("WARNING: no students enrolled yet. Run enroll_student.py or enroll_student_live.py first.")

    cap = open_capture(args.source)
    if not cap.isOpened():
        print(f"ERROR: could not open source {args.source}")
        return

    print("Press 'q' to quit.")
    frame_count = 0
    last_logged = {}   # student_id -> last logged timestamp, to avoid duplicate spam logging
    last_emotion = {}  # student_id -> (label, confidence, timestamp) -- most recent emotion reading,
                        # kept for display purposes so the label doesn't disappear between logs.
                        # NOT used for DB writes -- DetectionEvent still only gets a fresh emotion
                        # value computed at the moment of that specific log.

    faces_this_frame = []
    display_labels = []  # (bbox, text) pairs -- persists between processed frames for drawing

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed / end of stream.")
            break

        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            faces_this_frame = get_faces(frame)
            display_labels = []  # reset each processing cycle -- stale labels shouldn't linger

            for face in faces_this_frame:
                student_id, distance, matched_angle = match_face(face.embedding, known_embeddings)

                if student_id:
                    student = session.query(Student).get(student_id)

                    now = time.time()
                    last_time = last_logged.get(student_id, 0)
                    if now - last_time > LOG_COOLDOWN_SECONDS:
                        # only run the (slower) emotion model right when we're
                        # actually about to log -- not on every processed frame.
                        # Also gate on frontality: emotion classification on a
                        # steep angle produces confident-looking noise (see
                        # architecture notes), so skip it rather than trust it.
                        face_crop = crop_face_with_padding(frame, face.bbox)

                        if is_face_frontal_enough(face):
                            emotion_label, emotion_confidence, emotion_scores = get_emotion(face_crop)

                            # debug: dump crop + full score breakdown so we can see
                            # why 'sad' is/isn't winning against 'neutral'/'happy'
                            student_for_label = session.query(Student).get(student_id)
                            debug_label = student_for_label.roll_number if student_for_label else student_id
                            save_debug_crop(face_crop, debug_label, emotion_scores,
                                             emotion_label, emotion_confidence)
                        else:
                            emotion_label, emotion_confidence, emotion_scores = None, None, None
                            print("  (skipped emotion: face angle/quality below threshold)")

                        if emotion_label:
                            last_emotion[student_id] = (emotion_label, emotion_confidence, now)

                        # Posture: only safe to attribute when exactly one face is in
                        # frame this cycle -- see posture.py docstring for why. Cheap
                        # to skip; MediaPipe Pose never even runs otherwise.
                        posture_features = get_posture_for_frame(frame, len(faces_this_frame))
                        if posture_features is None and len(faces_this_frame) != 1:
                            print(f"  (skipped posture: {len(faces_this_frame)} faces in frame, ambiguous attribution)")

                        event = DetectionEvent(
                            student_id=student_id,
                            camera_id=camera.camera_id,
                            matched_confidence=1 - distance,
                            emotion_label=emotion_label,
                            emotion_confidence=emotion_confidence,
                            posture_features=posture_features,
                        )
                        session.add(event)
                        session.commit()
                        last_logged[student_id] = now

                        angle_str = f", matched via {matched_angle} angle" if matched_angle else ""
                        emotion_str = f", emotion={emotion_label} ({emotion_confidence:.2f})" if emotion_label else ""
                        print(f"[LOGGED] {student.name} at {args.camera_location} "
                              f"(dist={distance:.3f}{angle_str}{emotion_str})")
                        if emotion_scores:
                            ranked = sorted(emotion_scores.items(), key=lambda kv: -kv[1])
                            scores_str = ", ".join(f"{k}={v:.1f}" for k, v in ranked)
                            print(f"    scores: {scores_str}")

                        if args.show_labels:
                            display_labels.append((face.bbox, build_label(student, last_emotion, now)))
                    else:
                        # matched but still in cooldown -- keep showing name + last-known
                        # emotion so the box doesn't go label-less between log events
                        if args.show_labels:
                            display_labels.append((face.bbox, build_label(student, last_emotion, time.time())))
                # unmatched faces intentionally get no label, even with --show-labels

        # draw boxes for whatever we found on the last processed frame
        display = frame.copy()
        for face in faces_this_frame:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if args.show_labels:
            for bbox, text_label in display_labels:
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.putText(display, text_label, (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        display = resize_for_display(display)
        cv2.imshow(f"Recognition - {args.camera_location}", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    session.close()
if __name__ == "__main__":
    main()
