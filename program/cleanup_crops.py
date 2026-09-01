import os
import time
import argparse

CROP_DIR = "source_photos/detections"
DEFAULT_RETENTION_DAYS = 30


def cleanup(retention_days):
    cutoff = time.time() - (retention_days * 86400)  # 86400 = seconds in a day
    deleted_files = 0
    deleted_bytes = 0

    for student_dir in os.listdir(CROP_DIR):
        student_path = os.path.join(CROP_DIR, student_dir)
        if not os.path.isdir(student_path):
            continue

        for filename in os.listdir(student_path):
            filepath = os.path.join(student_path, filename)
            if os.path.getmtime(filepath) < cutoff:
                size = os.path.getsize(filepath)
                os.remove(filepath)
                deleted_files += 1
                deleted_bytes += size

        # remove the student folder too if it's now empty
        if not os.listdir(student_path):
            os.rmdir(student_path)
            print(f"  removed empty folder: {student_dir}")

    print(f"Cleanup done: {deleted_files} files removed "
          f"({deleted_bytes / (1024*1024):.1f} MB freed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=DEFAULT_RETENTION_DAYS,
                        help="Delete crops older than this many days (default: 30)")
    args = parser.parse_args()
    cleanup(args.days)