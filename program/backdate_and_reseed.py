"""
backdate_and_reseed.py
----------------------
Works with an ALREADY ENROLLED student (photos + embeddings untouched).

What it does:
  1. Finds the student by roll number
  2. Backdates enrolled_at to 20 days ago
  3. Clears only DetectionEvent / DailyActivitySummary / Flag / WellnessScore
     for that student  (FaceEmbedding rows are NOT touched)
  4. Seeds 20 days of detection events (normal days 1-14, decline days 15-20)
  5. Runs routine_baseline for each day → raises flags + wellness scores

Usage:
    python backdate_and_reseed.py --roll 21011153
    python backdate_and_reseed.py --roll 21011153 --days 15   # use 15 days instead
"""

import argparse
import random
from datetime import datetime, timedelta, timezone

from db import get_session
from models import (
    Student, Camera, DetectionEvent,
    DailyActivitySummary, Flag, WellnessScore
)

# ── defaults ──────────────────────────────────────────────────────────────────

CAMERA_LOCATION = "mess_entry"
TOTAL_DAYS      = 20    # how many days of history to create
NORMAL_DAYS     = 14   # days 1-14: healthy routine
DECLINE_START   = 15   # days 15-20: missed meals + negative emotions

MEAL_WINDOWS = {
    "breakfast": {"ist_h": 7,  "ist_m": 45},
    "lunch":     {"ist_h": 12, "ist_m": 30},
    "dinner":    {"ist_h": 19, "ist_m": 30},
}

EMOTIONS_NORMAL   = ["neutral", "happy", "neutral", "neutral", "happy"]
EMOTIONS_NEGATIVE = ["sad", "sad", "angry", "fear", "sad"]

IST_OFFSET = timedelta(hours=5, minutes=30)


# ── helpers ───────────────────────────────────────────────────────────────────

def ist_to_utc(naive_ist_dt):
    return naive_ist_dt - IST_OFFSET


def rand_arrival(base_day_utc, ist_h, ist_m, jitter_minutes=15):
    """UTC datetime for an IST meal arrival, with jitter."""
    ist_midnight = (base_day_utc + IST_OFFSET).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    ist_arrival = ist_midnight.replace(hour=ist_h, minute=ist_m)
    jitter = timedelta(minutes=random.randint(-jitter_minutes, jitter_minutes))
    return ist_to_utc(ist_arrival + jitter).replace(tzinfo=timezone.utc)


def make_event(student_id, camera_id, timestamp, emotion, is_decline):
    head_drop = round(random.uniform(-220, -150), 1) if is_decline else round(random.uniform(-80, -30), 1)
    return DetectionEvent(
        student_id=student_id,
        camera_id=camera_id,
        timestamp=timestamp,
        matched_confidence=round(random.uniform(0.75, 0.98), 4),
        emotion_label=emotion,
        emotion_confidence=round(random.uniform(0.55, 0.92), 4),
        posture_features={
            "head_drop_px":  head_drop,
            "torso_lean_px": round(random.uniform(-30, 30), 1),
            "torso_len_px":  round(random.uniform(580, 660), 1),
        },
    )


# ── core steps ────────────────────────────────────────────────────────────────

def backdate_student(session, student, total_days):
    new_enrolled = datetime.now(timezone.utc) - timedelta(days=total_days + 1)
    old = student.enrolled_at
    student.enrolled_at = new_enrolled
    session.commit()
    print(f"  enrolled_at: {old} → {new_enrolled.date()}")


def clear_detection_data(session, student_id):
    """Clears only detection-derived data. Embeddings and the student row are kept."""
    f = session.query(Flag).filter_by(student_id=student_id).delete()
    w = session.query(WellnessScore).filter_by(student_id=student_id).delete()
    d = session.query(DailyActivitySummary).filter_by(student_id=student_id).delete()
    e = session.query(DetectionEvent).filter_by(student_id=student_id).delete()
    session.commit()
    print(f"  Cleared: {e} events, {d} summaries, {f} flags, {w} wellness rows")


def seed_events(session, student, camera, total_days, normal_days, decline_start):
    today_utc = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_day = today_utc - timedelta(days=total_days)
    total_events = 0

    for day_offset in range(total_days):
        day_utc    = start_day + timedelta(days=day_offset)
        day_number = day_offset + 1
        is_decline = day_number >= decline_start

        print(f"  Day {day_number:02d}  {(day_utc + IST_OFFSET).date()}  "
              f"{'[ DECLINE ]' if is_decline else '[ normal  ]'}")

        if not is_decline:
            attend_b = random.random() > 0.05
            attend_l = random.random() > 0.05
            attend_d = random.random() > 0.05
        else:
            attend_b = random.random() > 0.30   # misses sometimes
            attend_l = random.random() > 0.65   # misses often
            attend_d = False                     # always misses dinner

        emotion_pool = EMOTIONS_NEGATIVE if is_decline else EMOTIONS_NORMAL

        for meal, window in MEAL_WINDOWS.items():
            attend = {"breakfast": attend_b, "lunch": attend_l, "dinner": attend_d}[meal]
            if not attend:
                print(f"           {meal}: MISSED")
                continue

            arrival      = rand_arrival(day_utc, window["ist_h"], window["ist_m"])
            num_sightings = random.randint(2, 4)
            print(f"           {meal}: {num_sightings} sightings "
                  f"@ ~{(arrival + IST_OFFSET).strftime('%H:%M')} IST")

            for i in range(num_sightings):
                ts = arrival + timedelta(minutes=i * random.randint(1, 4))
                event = make_event(
                    student.student_id, camera.camera_id, ts,
                    random.choice(emotion_pool), is_decline
                )
                session.add(event)
                total_events += 1

        session.commit()

    print(f"\n  Total events inserted: {total_events}")


def run_baseline(session, student, total_days):
    import routine_baseline as rb

    today_utc = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_day = today_utc - timedelta(days=total_days)

    print("\n=== Running routine_baseline for each seeded day ===\n")
    for day_offset in range(total_days):
        target = start_day + timedelta(days=day_offset)
        print(f"--- {target.date()} ---")
        rb.aggregate_day(session, target)
        rb.evaluate_student(session, student, target)
        rb.compute_wellness_score(session, student, target)

    # summary
    flags = (
        session.query(Flag)
        .filter_by(student_id=student.student_id)
        .order_by(Flag.created_at.desc())
        .all()
    )
    print(f"\n=== Flags raised for {student.name}: {len(flags)} ===")
    for f in flags:
        print(f"  [{f.signal_type.upper()}] score={f.score:.2f} | {f.reason}")

    scores = (
        session.query(WellnessScore)
        .filter_by(student_id=student.student_id)
        .order_by(WellnessScore.date.desc())
        .limit(7)
        .all()
    )
    print(f"\n=== Latest wellness scores for {student.name} ===")
    for ws in scores:
        print(f"  {ws.date.date()}  score={ws.score}  factors_used={ws.factors_used}")


# ── entry ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roll", required=True,
                        help="Roll number of the already-enrolled student, e.g. 21011153")
    parser.add_argument("--days", type=int, default=TOTAL_DAYS,
                        help=f"Total days of history to create (default {TOTAL_DAYS})")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip running routine_baseline after seeding")
    args = parser.parse_args()

    random.seed(42)
    session = get_session()

    # ── find student ──────────────────────────────────────────────────────────
    student = session.query(Student).filter_by(roll_number=args.roll).first()
    if not student:
        print(f"ERROR: No student found with roll number '{args.roll}'")
        session.close()
        return

    print(f"\nFound: {student.name}  (id={student.student_id}, roll={student.roll_number})")
    print(f"  Dept={student.department}, Year={student.year_of_study}, Active={student.is_active}")

    from models import FaceEmbedding
    emb_count = session.query(FaceEmbedding).filter_by(student_id=student.student_id).count()
    print(f"  Embeddings (kept intact): {emb_count}")

    # ── camera ────────────────────────────────────────────────────────────────
    camera = session.query(Camera).filter_by(location_name=CAMERA_LOCATION).first()
    if not camera:
        camera = Camera(location_name=CAMERA_LOCATION)
        session.add(camera)
        session.commit()
        print(f"  Created camera '{CAMERA_LOCATION}' (id={camera.camera_id})")
    else:
        print(f"  Using camera '{CAMERA_LOCATION}' (id={camera.camera_id})")

    # ── step 1: backdate enrollment ───────────────────────────────────────────
    total_days    = args.days
    normal_days   = max(1, total_days - 6)   # last 6 days are decline
    decline_start = normal_days + 1

    print(f"\n── Step 1: Backdating enrolled_at ──")
    backdate_student(session, student, total_days)

    # ── step 2: clear old detection data ─────────────────────────────────────
    print(f"\n── Step 2: Clearing old detection data (embeddings kept) ──")
    clear_detection_data(session, student.student_id)

    # ── step 3: seed events ───────────────────────────────────────────────────
    print(f"\n── Step 3: Seeding {total_days} days  "
          f"(normal: 1–{normal_days}, decline: {decline_start}–{total_days}) ──\n")
    seed_events(session, student, camera, total_days, normal_days, decline_start)

    # ── step 4: baseline + wellness ───────────────────────────────────────────
    if not args.no_baseline:
        run_baseline(session, student, total_days)

    session.close()
    print("\nDone. Refresh the dashboard to see wellness scores and flags.")


if __name__ == "__main__":
    main()