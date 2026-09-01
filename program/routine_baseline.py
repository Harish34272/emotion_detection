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
    "breakfast": (1, 30, 4, 0),    # 07:00–09:30 IST = 01:30–04:00 UTC
    "lunch":     (6, 30, 9, 0),    # 12:00–14:30 IST = 06:30–09:00 UTC
    "dinner":    (13, 30, 16, 0),  # 19:00–21:30 IST = 13:30–16:00 UTC
}
MEAL_LOCATIONS = {"mess_entry", "cafeteria_entry"}   # combined as one "meal" signal
VERANDA_LOCATIONS = {"hostel_veranda"}

BASELINE_WINDOW_DAYS = 21          # how much history to build a baseline from
MIN_HISTORY_DAYS_FOR_BASELINE = 7  # don't judge deviation until we know enough about this student

MISSED_MEAL_STREAK_THRESHOLD = 3   # flag if a normally-attended meal is missed this many days running
TIME_DRIFT_STD_MULTIPLIER = 2.0    # flag if check-in time is this many std-devs from personal baseline
VERANDA_DROP_RATIO = 0.4           # flag if veranda sightings fall below 40% of baseline average
MIN_DRIFT_MINUTES = 90
NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}
# ---- scoring weights (0.0 - 1.0) -----------------------------------------
SCORE_MISSED_MEAL_STREAK   = 0.4   # missing meals is a strong signal
SCORE_TIME_DRIFT           = 0.2   # timing shift is a weak signal
SCORE_VERANDA_DROPOFF      = 0.3   # social withdrawal is moderate
SCORE_EMOTION_TREND        = 0.3   # emotion alone is supplementary
EMOTION_MIN_DAYS = 3             # ← new: minimum days of data before flagging
EMOTION_NEG_RATIO_THRESHOLD = 0.5  # ← new: flag if avg negative ratio exceeds this
SCORE_ESCALATION_MULTIPLIER = 1.5
SCORE_MAX = 1.0 
FLAG_COOLDOWN_DAYS = 7 
# ---- step 1: aggregate ----------------------------------------------------

def which_meal(dt):
    # normalize to UTC so hour comparison is consistent regardless of DB timezone
    from datetime import timezone
    dt_utc = dt.astimezone(timezone.utc)
    for meal, (sh, sm, eh, em) in MEAL_WINDOWS.items():
        start = dt_utc.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end   = dt_utc.replace(hour=eh, minute=em, second=0, microsecond=0)
        if start <= dt_utc <= end:
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
    print(f"  Total events fetched: {len(events)}")
    for event, camera in events[:3]:
        print(f"    ts={event.timestamp} loc={camera.location_name} meal={which_meal(event.timestamp)}")
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


def check_missed_meal_streak(session, student_id, meal, upto_date, history_rows, attendance_rate, today_row):
    if attendance_rate < 0.5:
        return None

    # build a date->row lookup, missing dates treated as "missed"
    row_by_date = {r.date.date(): r for r in history_rows}

    # check today + last 2 calendar days (not just last 2 DB rows)
    days_to_check = [
        upto_date.date() - timedelta(days=i)
        for i in range(MISSED_MEAL_STREAK_THRESHOLD)
    ]

    attended_flags = []
    for d in days_to_check:
        if d == upto_date.date():
            attended_flags.append(bool(getattr(today_row, f"{meal}_attended")))
        elif d in row_by_date:
            attended_flags.append(bool(getattr(row_by_date[d], f"{meal}_attended")))
        else:
            attended_flags.append(False)  # no row = not seen = missed

    print(f"    [streak] meal={meal} attendance_rate={attendance_rate:.2f} upto={upto_date.date()}")
    print(f"    [streak] days_to_check: {days_to_check}")
    print(f"    [streak] attended flags: {attended_flags}")

    if all(not f for f in attended_flags):
        return (f"Missed {meal} for {MISSED_MEAL_STREAK_THRESHOLD} consecutive days "
                f"(normally attends ~{attendance_rate:.0%} of the time)")
    return None


def check_time_drift(today_row, meal, mean_t, std_t, history_rows):
    attended = [r for r in history_rows if getattr(r, f"{meal}_attended")]
    if len(attended) < 5:
        return None
    if mean_t is None or std_t in (None, 0):
        return None
    today_time = getattr(today_row, f"{meal}_time")
    if not today_time or not getattr(today_row, f"{meal}_attended"):
        return None
    today_minutes = time_to_minutes(today_time)
    drift = abs(today_minutes - mean_t)
    if drift > TIME_DRIFT_STD_MULTIPLIER * std_t and std_t > 5 and drift > MIN_DRIFT_MINUTES:
        return (f"{meal.capitalize()} check-in time drifted {drift:.0f} min from personal baseline "
                f"(usual ~{int(mean_t // 60):02d}:{int(mean_t % 60):02d})")
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
            session, student.student_id, meal, day_start,
            history_rows, attendance_rate, today_row   # ← add today_row
        )
        if streak_reason:
            reasons.append(("routine", streak_reason, SCORE_MISSED_MEAL_STREAK))

        drift_reason = check_time_drift(today_row, meal, mean_t, std_t, history_rows)
        if drift_reason:
            reasons.append(("routine", drift_reason, SCORE_TIME_DRIFT))

    veranda_reason = check_veranda_dropoff(today_row, history_rows)
    if veranda_reason:
        reasons.append(("routine", veranda_reason, SCORE_VERANDA_DROPOFF))

    # weak supplementary signal: sustained negative emotion trend
    recent_rows_sorted = sorted(history_rows, key=lambda r: r.date, reverse=True)[:4]
    neg_ratios = [r.avg_emotion_negative_ratio for r in recent_rows_sorted
                if r.avg_emotion_negative_ratio is not None]
    if today_row.avg_emotion_negative_ratio is not None:
        neg_ratios = [today_row.avg_emotion_negative_ratio] + neg_ratios

    if len(neg_ratios) >= EMOTION_MIN_DAYS:
        avg_neg = sum(neg_ratios) / len(neg_ratios)
        if avg_neg > EMOTION_NEG_RATIO_THRESHOLD:
            reason_text = (
                f"Sustained negative emotion trend over recent days "
                f"(avg {avg_neg:.0%} negative across last {len(neg_ratios)} days)"
            )
            reasons.append(("emotion", reason_text, SCORE_EMOTION_TREND))

    if not reasons:
        return

    for signal_type, reason, base_score in reasons:
        # count previous similar flags to escalate score
        previous_flags = (
            session.query(Flag)
            .filter(
                Flag.student_id == student.student_id,
                Flag.signal_type == signal_type,
                Flag.reason.contains(reason[:30]),
            )
            .count()
        )

        score = min(SCORE_MAX, base_score * (SCORE_ESCALATION_MULTIPLIER ** previous_flags))

        # cooldown check
        recent_flag = (
            session.query(Flag)
            .filter(
                Flag.student_id == student.student_id,
                Flag.created_at >= day_start - timedelta(days=FLAG_COOLDOWN_DAYS),
                Flag.signal_type == signal_type,
                Flag.reason.contains(reason[:30]),
            )
            .first()
        )
        if recent_flag:
            print(f"  (skipping — recent similar flag exists)")
            continue

        flag = Flag(
            student_id=student.student_id,
            reason=reason,
            score=score,
            signal_type=signal_type,
            status="pending",
        )
        session.add(flag)
        print(f"  FLAGGED {student.name} ({student.roll_number}): {reason} [score={score:.2f}]")

    session.commit()

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
