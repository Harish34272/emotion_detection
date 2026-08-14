"""
enroll_from_photos.py

Static-image equivalent of enroll_student_live.py — use this instead of
enroll_student.py for anything that needs to work with recognize_and_log.py.

WHY THIS SCRIPT EXISTS:
enroll_student.py generates embeddings via DeepFace/ArcFace. Since the
engine swap, recognize_and_log.py matches faces using InsightFace
embeddings. ArcFace and InsightFace embeddings are NOT compatible vector
spaces — cosine distance between them is meaningless. enroll_student.py
was never updated after the swap, so it's not usable for enrolling
students who need to be recognized by the current pipeline.

This script reuses the exact same validation + DB-save logic as
enroll_student_live.py, but reads frames from image files on disk
instead of capturing from a live camera. Give it up to three photos
(straight/left/right) and matching pose labels.

Usage:
  python3 enroll_from_photos.py --name "Virat Kohli" --roll 21010052 \
      --dept mech --year 3 \
      --photos "/path/straight_face.jpg" \
               "/path/Head turned left_ three-quarter view.png" \
               "/path/Head Turned Three-Quarter Right.png" \
      --angles straight left right
"""

import argparse
import os

import cv2

from db import get_session
from models import Student, FaceEmbedding
from face_engine import get_faces

VALID_ANGLES = {"straight", "left", "right"}
SAVE_DIR = "source_photos/enrollment"  # mirrors enroll_student_live.py layout


def validate_and_embed(image_path):
    """
    Same rule as enroll_student_live.py: run InsightFace on the full-res
    image, accept only if exactly one clear face is found.
    Returns (embedding, error_reason) — error_reason is None on success.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return None, f"could not read image (bad path or unsupported format): {image_path}"

    faces = get_faces(frame)
    if len(faces) == 0:
        return None, "no face detected"
    if len(faces) > 1:
        return None, f"{len(faces)} faces detected — expected exactly 1"

    return faces[0].embedding, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--roll", required=True)
    parser.add_argument("--dept", required=True)
    parser.add_argument("--year", type=int, required=False)
    parser.add_argument("--photos", nargs="+", required=True,
                         help="one or more photo paths (straight/left/right)")
    parser.add_argument("--angles", nargs="+", required=True,
                         help="pose label per photo, same order/length as --photos "
                              "(must be from: straight, left, right)")
    args = parser.parse_args()

    if len(args.angles) != len(args.photos):
        print("ERROR: --angles must have the same number of entries as --photos.")
        return

    bad = [a for a in args.angles if a not in VALID_ANGLES]
    if bad:
        print(f"ERROR: unrecognized angle label(s) {bad} -- must be one of {VALID_ANGLES}.")
        return

    safe_roll = args.roll.replace("/", "_")
    student_dir = os.path.join(SAVE_DIR, safe_roll)
    os.makedirs(student_dir, exist_ok=True)

    captured_photos = {}      # pose_name -> saved file path
    captured_embeddings = {}  # pose_name -> embedding

    for photo_path, pose_name in zip(args.photos, args.angles):
        print(f"\n--- Pose: {pose_name.upper()} ({photo_path}) ---")
        embedding, err = validate_and_embed(photo_path)
        if err:
            print(f"  SKIPPED: {err}")
            continue

        # copy into the standard enrollment layout so future review/debugging
        # matches what enroll_student_live.py produces
        frame = cv2.imread(photo_path)
        dest_path = os.path.join(student_dir, f"{pose_name}.jpg")
        cv2.imwrite(dest_path, frame)

        captured_photos[pose_name] = dest_path
        captured_embeddings[pose_name] = embedding
        print(f"  OK — saved {dest_path}")

    if len(captured_embeddings) < 2:
        print("\nToo few valid poses captured (need at least 2), aborting DB save.")
        return

    # --- save to DB (same pattern as enroll_student_live.py, with try/except
    #     since that script's missing error handling was flagged as an
    #     outstanding issue in the handoff) ---
    session = get_session()
    try:
        existing = session.query(Student).filter_by(roll_number=args.roll).first()
        if existing:
            print(f"\nStudent with roll {args.roll} already exists (id={existing.student_id}). "
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
            print(f"\nCreated student id={student.student_id}: {student.name}")

        student_name = student.name  # capture before close, avoids DetachedInstanceError
        student_id = student.student_id

        added = 0
        for pose_name, embedding in captured_embeddings.items():
            fe = FaceEmbedding(
                student_id=student_id,
                embedding=embedding.tolist(),  # numpy array -> JSON-serializable list
                angle_label=pose_name,
                source_photo=captured_photos[pose_name],
            )
            session.add(fe)
            added += 1

        session.commit()
        print(f"\nDone. {added}/{len(captured_embeddings)} embeddings saved for {student_name}.")

    except Exception as e:
        session.rollback()
        print(f"\nERROR: DB save failed, nothing committed for this run: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
