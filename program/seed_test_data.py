"""
seed_test_data.py -- generates 30 days of fake DetectionEvent rows for testing
------------------------------------------------------------------------------
Simulates a student's normal routine for 25 days, then introduces
suspicious behaviour (missed meals, negative emotions) on days 26-30
so that routine_baseline.py raises flags.

Usage:
    python seed_test_data.py              # seed + run baseline for all 30 days
    python seed_test_data.py --clear      # wipe existing events/summaries/flags first
    python seed_test_data.py --clear --dry-run   # just show what would be inserted
"""

import argparse
import random
from datetime import datetime, timedelta, timezone

from db import get_session
from models import Student, Camera, DetectionEvent, DailyActivitySummary, Flag

# ---- config ---------------------------------------------------------------

MEAL_WINDOWS = {
    "breakfast": {"start": (7, 30), "end": (9, 0)},   # normal arrival window
    "lunch":     {"start": (12, 15), "end": (13, 30)},
    "dinner":    {"start": (19, 15), "end": (20, 30)},
}

CAMERA_LOCATION = "mess_entry"   # must match what's in your cameras table

EMOTIONS_NORMAL   = ["neutral", "happy", "neutral", "neutral", "happy"]
EMOTIONS_NEGATIVE = ["sad", "sad", "angry", "fear", "sad", "neutral"]

TOTAL_DAYS = 30
NORMAL_DAYS = 25      # days 1-25: healthy routine
DECLINE_START = 26    # days 26-30: start missing meals + negative emotions


# ---- helpers ---------------------------------------------------------------

def rand_time(base_date, hour, minute, jitter_minutes=15):
    from datetime import timezone, timedelta
    IST = timezone(timedelta(hours=5, minutes=30))
    base = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0,
                             tzinfo=IST)
    jitter = random.randint(-jitter_minutes, jitter_minutes)
    return base + timedelta(minutes=jitter)


def make_event(student_id, camera_id, timestamp, emotion, head_drop):
    return DetectionEvent(
        student_id=student_id,
        camera_id=camera_id,
        timestamp=timestamp,
        matched_confidence=round(random.uniform(0.75, 0.98), 4),
        emotion_label=emotion,
        emotion_confidence=round(random.uniform(0.55, 0.92), 4),
        posture_features={
            "head_drop_px": round(head_drop + random.uniform(-20, 20), 1),
            "torso_lean_px": round(random.uniform(-30, 30), 1),
            "torso_len_px": round(random.uniform(580, 660), 1),
        }
    )


# ---- main seeder -----------------------------------------------------------

def seed(session, student, camera, dry_run=False):
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # day 1 is 30 days ago
    start_day = today - timedelta(days=TOTAL_DAYS)

    total_events = 0

    for day_offset in range(TOTAL_DAYS):
        day = start_day + timedelta(days=day_offset)
        day_number = day_offset + 1          # 1-indexed for readability
        is_decline = day_number >= DECLINE_START

        print(f"  Day {day_number:02d}  {day.date()}  "
              f"{'[ DECLINE ]' if is_decline else '[ normal  ]'}")

        # decide which meals to attend
        if not is_decline:
            # normal: attend all 3 meals reliably (skip randomly ~10%)
            attend_breakfast = random.random() > 0.05
            attend_lunch     = random.random() > 0.05
            attend_dinner    = random.random() > 0.05
        else:
            # decline phase: start skipping dinner, then lunch too
            attend_breakfast = random.random() > 0.20
            attend_lunch     = random.random() > 0.60   # often misses lunch
            attend_dinner    = False                     # always misses dinner

        emotion_pool = EMOTIONS_NEGATIVE if is_decline else EMOTIONS_NORMAL
        head_drop_base = -180 if is_decline else -50   # posture worsens too

        # insert 2-4 detection events per attended meal
        for meal, window in MEAL_WINDOWS.items():
            attend = {"breakfast": attend_breakfast,
                      "lunch":     attend_lunch,
                      "dinner":    attend_dinner}[meal]

            if not attend:
                print(f"           {meal}: MISSED")
                continue

            h, m = window["start"]
            arrival = rand_time(day, h, m, jitter_minutes=20)
            num_sightings = random.randint(2, 4)

            print(f"           {meal}: {num_sightings} sightings @ ~{arrival.strftime('%H:%M')}")

            for i in range(num_sightings):
                ts = arrival + timedelta(minutes=i * random.randint(1, 4))
                emotion = random.choice(emotion_pool)
                event = make_event(student.student_id, camera.camera_id,
                                   ts, emotion, head_drop_base)
                if not dry_run:
                    session.add(event)
                total_events += 1

        if not dry_run:
            session.commit()

    print(f"\n  Total events inserted: {total_events}")


def clear_data(session):
    print("Clearing existing flags, summaries, and detection events...")
    session.query(Flag).delete()
    session.query(DailyActivitySummary).delete()
    session.query(DetectionEvent).delete()
    session.commit()
    print("Done.\n")


def run_baseline_for_all_days(session, student):
    """
    After seeding, run baseline+deviation for each day from day 8 onwards
    (need at least 7 days of history before baseline kicks in).
    """
    import routine_baseline as rb

    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_day = today - timedelta(days=TOTAL_DAYS)

    print("\n=== Running routine_baseline for each seeded day ===\n")

    for day_offset in range(TOTAL_DAYS):
        target = start_day + timedelta(days=day_offset)
        print(f"--- {target.date()} ---")
        rb.aggregate_day(session, target)
        rb.evaluate_student(session, student, target)

    print("\n=== Baseline run complete ===")


# ---- entry -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing events/summaries/flags before seeding")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be inserted without writing to DB")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip running routine_baseline after seeding")
    args = parser.parse_args()

    session = get_session()

    # verify student exists
    student = session.query(Student).first()
    if not student:
        print("ERROR: No students found in DB. Please enrol a student first.")
        return
    print(f"Using student: {student.name} ({student.roll_number})\n")

    # verify camera exists
    camera = session.query(Camera).filter_by(location_name=CAMERA_LOCATION).first()
    if not camera:
        print(f"ERROR: Camera '{CAMERA_LOCATION}' not found in DB.")
        print("Run recognize_and_log.py once first to auto-create it.")
        return
    print(f"Using camera: {camera.location_name}\n")

    if args.clear and not args.dry_run:
        clear_data(session)

    print(f"=== Seeding {TOTAL_DAYS} days of fake data ===")
    print(f"    Normal days  : 1 - {NORMAL_DAYS}")
    print(f"    Decline days : {DECLINE_START} - {TOTAL_DAYS}")
    print(f"    Dry run      : {args.dry_run}\n")

    seed(session, student, camera, dry_run=args.dry_run)

    if not args.dry_run and not args.no_baseline:
        run_baseline_for_all_days(session, student)

        # show flags raised
        flags = session.query(Flag).filter_by(student_id=student.student_id).all()
        print(f"\n=== Flags raised for {student.name}: {len(flags)} ===")
        for f in flags:
            print(f"  [{f.signal_type.upper()}] score={f.score:.2f} | {f.reason}")

    session.close()


if __name__ == "__main__":
    random.seed(42)   
    main()