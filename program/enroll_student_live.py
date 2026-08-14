

import argparse
import os

import cv2
from dotenv import load_dotenv
load_dotenv()
DEFAULT_CAMERA = os.environ.get("DEFAULT_CAMERA", 0) 
from db import get_session
from models import Student, FaceEmbedding
from face_engine import get_faces, resize_for_display, open_capture

SAVE_DIR = "source_photos/enrollment"   # keeps a copy of captured photos, useful for re-checking/debugging later

POSES = [
    ("straight", "Look directly at the camera"),
    ("left", "Turn head ~30-45 deg to your LEFT (camera's right)"),
    ("right", "Turn head ~30-45 deg to your RIGHT (camera's left)"),
]


def capture_pose(cap, pose_name, instruction):
    """Shows a live preview with instructions until the user captures or skips."""
    print(f"\n--- Pose: {pose_name.upper()} ---")
    print(f"Instruction: {instruction}")
    print("Press SPACE to capture, 'r' to retake after capture, 'q' to quit.")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            return None

        display = resize_for_display(frame.copy())
        cv2.putText(display, f"Pose: {pose_name.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, instruction, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if captured_frame is not None:
            cv2.putText(display, "CAPTURED - press SPACE to confirm, 'r' to retake",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Enrollment", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return None
        elif key == ord("r"):
            captured_frame = None
        elif key == ord(" "):
            if captured_frame is None:
                captured_frame = frame.copy()  # keep full-res original for detection/embedding
                print("  Captured. Press SPACE again to confirm, or 'r' to retake.")
            else:
                return captured_frame


def validate_and_embed(frame):
    """
    Runs InsightFace on the full-resolution captured frame. Returns the
    embedding only if exactly one clear face is found (rejects empty or
    multi-face captures, e.g. someone walking through the background).
    """
    faces = get_faces(frame)
    if len(faces) != 1:
        return None
    return faces[0].embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--roll", required=True)
    parser.add_argument("--dept", required=True)
    parser.add_argument("--year", type=int, required=False)
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                         help="camera index, RTSP URL, or 'gst:<pipeline>' for a raw GStreamer pipeline")
    args = parser.parse_args()

    cap = open_capture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {args.camera}")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    safe_roll = args.roll.replace("/", "_")
    student_dir = os.path.join(SAVE_DIR, safe_roll)
    os.makedirs(student_dir, exist_ok=True)

    captured_photos = {}      # pose_name -> file path
    captured_embeddings = {}  # pose_name -> embedding

    for pose_name, instruction in POSES:
        while True:
            frame = capture_pose(cap, pose_name, instruction)
            if frame is None:
                print("Enrollment cancelled or camera error.")
                cap.release()
                cv2.destroyAllWindows()
                return

            embedding = validate_and_embed(frame)
            if embedding is None:
                print("  No clear single face detected in that capture -- retrying this pose.")
                continue

            photo_path = os.path.join(student_dir, f"{pose_name}.jpg")
            cv2.imwrite(photo_path, frame)
            captured_photos[pose_name] = photo_path
            captured_embeddings[pose_name] = embedding
            print(f"  Saved {photo_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(captured_embeddings) < 2:
        print("Too few valid poses captured, aborting DB save.")
        return

    # --- save to DB ---
    session = get_session()
    existing = session.query(Student).filter_by(roll_number=args.roll).first()
    if existing:
        print(f"Student with roll {args.roll} already exists (id={existing.student_id}). "
              f"Adding new embeddings to existing student.")
        student = existing
    else:
        student = Student(
            name=args.name,
            roll_number=args.roll,
            department=args.dept,
            year_of_study=args.year,
            photo_reference_path=captured_photos.get("straight"),
        )
        session.add(student)
        session.commit()
        print(f"Created student id={student.student_id}: {student.name}")

    added = 0
    for pose_name, embedding in captured_embeddings.items():
        fe = FaceEmbedding(
            student_id=student.student_id,
            embedding=embedding.tolist(),  # numpy array -> JSON-serializable list
            angle_label=pose_name,
            source_photo=captured_photos[pose_name],
        )
        session.add(fe)
        added += 1

    session.commit()
    session.close()
    print(f"\nDone. {added}/{len(captured_embeddings)} embeddings saved for {student.name}.")


if __name__ == "__main__":
    main()
