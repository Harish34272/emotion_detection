"""
routine_baseline.py -- the behavioral/routine analysis batch job
---------------------------------------------------------------------
Run this once a day (e.g. via cron, after the day's mess/cafeteria/veranda
detections are in). It does three things, in order:

  1. AGGREGATE   raw detection_events (yesterday) -> one daily_activity_summary row per student
  2. BASELINE    compute each student's personal rolling baseline from their
                 own history (mean/std of meal times, typical veranda frequency)
  3. DEVIATE     compare yesterday's summary against the baseline, and if it
                 crosses a threshold with sustained history, insert a Flag

No automated action is taken. A Flag is just a row a human will review.

Run:
    python routine_baseline.py --date 2026-08-10
    python routine_baseline.py                     # defaults to yesterday
"""

import argparse
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from db import get_session
from models import Student, DetectionEvent, Camera, DailyActivitySummary, Flag

# ---- configuration -------------------------------------------------------

# map camera location_name -> which "meal window" it belongs to
# (edit this to match your actual camera location_names)
MEAL_WINDOWS = {
    "breakfast": (7, 0, 9, 30),   # (start_hour, start_min, end_hour, end_min)
    "lunch": (12, 0, 14, 30),
    "dinner": (19, 0, 21, 30),
}
MEAL_LOCATIONS = {"mess_entry", "cafeteria_entry"}   # combined as one "meal" signal
VERANDA_LOCATIONS = {"hostel_veranda"}

BASELINE_WINDOW_DAYS = 21          # how much history to build a baseline from
MIN_HISTORY_DAYS_FOR_BASELINE = 7  # don't judge deviation until we know enough about this student

MISSED_MEAL_STREAK_THRESHOLD = 3   # flag if a normally-attended meal is missed this many days running
TIME_DRIFT_STD_MULTIPLIER = 2.0    # flag if check-in time is this many std-devs from personal baseline
VERANDA_DROP_RATIO = 0.4           # flag if veranda sightings fall below 40% of baseline average

NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}


# ---- step 1: aggregate ----------------------------------------------------

def which_meal(dt):
    for meal, (sh, sm, eh, em) in MEAL_WINDOWS.items():
        start = dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end = dt.replace(hour=eh, minute=em, second=0, microsecond=0)
        if start <= dt <= end:
            return meal
    return None


def aggregate_day(session, target_date):
    """Builds one DailyActivitySummary row per student for target_date."""
    day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    events = (
        session.query(DetectionEvent, Camera)
        .join(Camera, DetectionEvent.camera_id == Camera.camera_id)
        .filter(DetectionEvent.timestamp >= day_start, DetectionEvent.timestamp < day_end)
        .filter(DetectionEvent.student_id.isnot(None))
        .all()
    )

    per_student = defaultdict(lambda: {
        "meals": {}, "veranda_count": 0, "emotions": [], "head_drops": []
    })

    for event, camera in events:
        s = per_student[event.student_id]
        loc = camera.location_name

        if loc in MEAL_LOCATIONS:
            meal = which_meal(event.timestamp)
            if meal and meal not in s["meals"]:  # first sighting for that meal window wins
                s["meals"][meal] = event.timestamp
        elif loc in VERANDA_LOCATIONS:
            s["veranda_count"] += 1

        if event.emotion_label:
            s["emotions"].append(event.emotion_label)
        if event.posture_features and "head_drop_px" in event.posture_features:
            s["head_drops"].append(event.posture_features["head_drop_px"])

    written = 0
    for student_id, data in per_student.items():
        neg_ratio = None
        if data["emotions"]:
            neg = sum(1 for e in data["emotions"] if e in NEGATIVE_EMOTIONS)
            neg_ratio = neg / len(data["emotions"])

        avg_head_drop = None
        if data["head_drops"]:
            avg_head_drop = sum(data["head_drops"]) / len(data["head_drops"])

        existing = session.query(DailyActivitySummary).filter_by(
            student_id=student_id, date=day_start
        ).first()
        if existing:
            summary = existing
        else:
            summary = DailyActivitySummary(student_id=student_id, date=day_start)
            session.add(summary)

        summary.breakfast_attended = "breakfast" in data["meals"]
        summary.breakfast_time = data["meals"].get("breakfast")
        summary.lunch_attended = "lunch" in data["meals"]
        summary.lunch_time = data["meals"].get("lunch")
        summary.dinner_attended = "dinner" in data["meals"]
        summary.dinner_time = data["meals"].get("dinner")
        summary.veranda_sightings = data["veranda_count"]
        summary.avg_emotion_negative_ratio = neg_ratio
        summary.avg_head_drop = avg_head_drop
        written += 1

    session.commit()
    print(f"Aggregated {written} student-days for {day_start.date()}")


# ---- step 2 + 3: baseline + deviation -------------------------------------

def time_to_minutes(dt):
    return dt.hour * 60 + dt.minute if dt else None


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, variance ** 0.5


def build_baseline(history_rows, meal):
    """Returns (attendance_rate, mean_minutes, std_minutes) for a meal, from history rows."""
    attended = [r for r in history_rows if getattr(r, f"{meal}_attended")]
    attendance_rate = len(attended) / len(history_rows) if history_rows else 0
    times = [time_to_minutes(getattr(r, f"{meal}_time")) for r in attended]
    times = [t for t in times if t is not None]
    mean_t, std_t = mean_std(times)
    return attendance_rate, mean_t, std_t


def check_missed_meal_streak(session, student_id, meal, upto_date, history_rows, attendance_rate):
    """Flags if a normally-attended meal has been missed for N days running."""
    if attendance_rate < 0.5:
        return None  # this student doesn't reliably attend this meal anyway -- not a signal

    recent = sorted(history_rows, key=lambda r: r.date, reverse=True)[:MISSED_MEAL_STREAK_THRESHOLD]
    if len(recent) < MISSED_MEAL_STREAK_THRESHOLD:
        return None

    all_missed = all(not getattr(r, f"{meal}_attended") for r in recent)
    if all_missed:
        return f"Missed {meal} for {MISSED_MEAL_STREAK_THRESHOLD} consecutive days " \
               f"(normally attends ~{attendance_rate:.0%} of the time)"
    return None


def check_time_drift(today_row, meal, mean_t, std_t):
    if mean_t is None or std_t in (None, 0):
        return None
    today_time = getattr(today_row, f"{meal}_time")
    if not today_time or not getattr(today_row, f"{meal}_attended"):
        return None
    today_minutes = time_to_minutes(today_time)
    drift = abs(today_minutes - mean_t)
    if drift > TIME_DRIFT_STD_MULTIPLIER * std_t and std_t > 5:  # ignore near-zero-variance noise
        return f"{meal.capitalize()} check-in time drifted {drift:.0f} min from personal baseline " \
               f"(usual ~{int(mean_t // 60):02d}:{int(mean_t % 60):02d})"
    return None


def check_veranda_dropoff(today_row, history_rows):
    counts = [r.veranda_sightings for r in history_rows]
    mean_c, _ = mean_std(counts)
    if mean_c is None or mean_c < 1:
        return None  # not a meaningful baseline to compare against
    if today_row.veranda_sightings < mean_c * VERANDA_DROP_RATIO:
        return f"Veranda sightings dropped to {today_row.veranda_sightings} " \
               f"(baseline avg ~{mean_c:.1f}/day)"
    return None


def evaluate_student(session, student, target_date):
    day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    window_start = day_start - timedelta(days=BASELINE_WINDOW_DAYS)

    history_rows = (
        session.query(DailyActivitySummary)
        .filter(
            DailyActivitySummary.student_id == student.student_id,
            DailyActivitySummary.date >= window_start,
            DailyActivitySummary.date < day_start,  # history excludes today
        )
        .all()
    )

    if len(history_rows) < MIN_HISTORY_DAYS_FOR_BASELINE:
        return  # not enough history yet to judge this student fairly

    today_row = session.query(DailyActivitySummary).filter_by(
        student_id=student.student_id, date=day_start
    ).first()
    if not today_row:
        return  # student wasn't seen at all today -- could itself be worth a softer flag later

    reasons = []

    for meal in ("breakfast", "lunch", "dinner"):
        attendance_rate, mean_t, std_t = build_baseline(history_rows, meal)

        streak_reason = check_missed_meal_streak(
            session, student.student_id, meal, day_start, history_rows, attendance_rate
        )
        if streak_reason:
            reasons.append(("routine", streak_reason))

        drift_reason = check_time_drift(today_row, meal, mean_t, std_t)
        if drift_reason:
            reasons.append(("routine", drift_reason))

    veranda_reason = check_veranda_dropoff(today_row, history_rows)
    if veranda_reason:
        reasons.append(("routine", veranda_reason))

    # weak supplementary signal: sustained negative emotion trend
    recent_neg = [r.avg_emotion_negative_ratio for r in history_rows[-5:] if r.avg_emotion_negative_ratio is not None]
    if today_row.avg_emotion_negative_ratio and recent_neg:
        if today_row.avg_emotion_negative_ratio > 0.6 and sum(recent_neg) / len(recent_neg) > 0.5:
            reasons.append(("emotion", "Sustained negative emotion trend over recent visits (supplementary signal)"))

    if not reasons:
        return

    # avoid re-flagging a student who already has a pending/reviewed flag from the last 7 days
    recent_flag = (
        session.query(Flag)
        .filter(Flag.student_id == student.student_id, Flag.created_at >= day_start - timedelta(days=7))
        .first()
    )
    if recent_flag:
        print(f"  (skipping flag for {student.name} -- already has a recent flag, status={recent_flag.status})")
        return

    combined_reason = "; ".join(r for _, r in reasons)
    signal_types = set(t for t, _ in reasons)
    signal_type = "combined" if len(signal_types) > 1 else next(iter(signal_types))
    score = min(1.0, 0.3 * len(reasons))  # simple starting scoring -- tune once you see real outcomes

    flag = Flag(
        student_id=student.student_id,
        reason=combined_reason,
        score=score,
        signal_type=signal_type,
        status="pending",
    )
    session.add(flag)
    session.commit()
    print(f"  FLAGGED {student.name} ({student.roll_number}): {combined_reason}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD, defaults to yesterday")
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        target_date = datetime.now(timezone.utc) - timedelta(days=1)

    session = get_session()

    print(f"=== Step 1: Aggregating detections for {target_date.date()} ===")
    aggregate_day(session, target_date)

    print(f"=== Step 2+3: Baseline + deviation check for {target_date.date()} ===")
    students = session.query(Student).all()
    for student in students:
        evaluate_student(session, student, target_date)

    session.close()
    print("Done.")


if __name__ == "__main__":
    main()
