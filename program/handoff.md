# Project Handoff: Student Wellness Monitoring System

## Project Goal
Building a student wellness monitoring system for a college client. Goal is
**early detection of student distress signals**, routed to human counselors
for a real check-in — not automated action. Success = a reliable,
privacy-conscious pipeline that supports counselor intervention.

## Context & Constraints
- Sanctioned institutional deployment, done with proper permissions from
  faculty and students (not covert surveillance)
- Cameras placed at public campus locations: **mess, cafeteria, hostel veranda**
  (no classroom/attendance-log cameras)
- Clear signage at camera locations recommended as a protective measure
- **No automated action ever taken on a flagged student.** System only
  surfaces flags to a human (counselor/warden); a real person decides next steps
- No training data / gated-dataset access available yet — architecture is
  built entirely on **pretrained models** for now; custom training is an
  upgrade path, not a current requirement

## Architecture Overview
Three layers, documented in full in `architecture.md` (already delivered):

1. **Perception layer** — face detection, face recognition (identity),
   emotion classification, MediaPipe Pose skeleton + engineered posture features
2. **Behavioral/routine layer** — per-student baseline deviation detection
   from checkpoint logs (this is the STRONGEST, most reliable signal —
   weighted highest in fusion)
3. **Fusion layer** — combines signals into a risk score → routed to
   human counselors, never automated

## Key Decisions Made (with reasoning)
- **PostgreSQL over SQLite** — needed for concurrent multi-camera writes
  and time-series queries over weeks of data
- **SQLAlchemy (`models.py`) + Alembic for migrations** — chosen to mimic
  a Prisma-style workflow: edit the model class, autogenerate migration,
  review, apply. No hand-written ALTER TABLE.
- **student_id ≠ roll_number** — kept as separate fields; student_id is an
  internal immutable PK, roll_number is the college's external identifier
  (can theoretically change; never used as a FK target)
- **department, year_of_study** added to `students` table per explicit request
- **face_embeddings is a separate table**, not a column on students —
  supports multiple reference photos per student (accuracy) and
  re-enrollment without data loss
- **No model training needed for detection/recognition/pose** — RetinaFace/
  Haar (detection), ArcFace via DeepFace (recognition), MediaPipe Pose are
  all pretrained and sufficient. Training only becomes relevant later if
  DeepFace's emotion classifier underperforms on real footage, or for a
  learned posture-emotion model (using BOLD/EMOTIC datasets) as a future upgrade
- **YOLO vs ArcFace clarified**: YOLO = detection only (finds a face).
  ArcFace = recognition (identifies whose face). They're sequential, not
  alternatives — you need both.
- **Body language/posture matters, not just face** — faces get occluded/
  angled away in a cafeteria; posture (slouch, head-down duration) is a
  real complementary signal. Datasets referenced for future upgrade: BOLD,
  EMOTIC, CAER-S (posture-in-context emotion), MPII/COCO-Pose (skeleton training)
- **Routine/behavioral signal is the primary driver**, emotion and posture
  are weaker supplementary signals — deliberately rules/statistics-based
  (not ML) for interpretability and explainability to the institution
- **Mess + cafeteria treated as one combined "meal attendance" signal**;
  hostel veranda kept separate (measures general presence/activity, not meals)
- Reviewed an alternative architecture doc (uploaded by user) proposing full
  PyTorch training on AffectNet/RAF-DB + LSTM posture models. **Decision:
  not adopted as primary approach** (too resource-heavy for current stage,
  gated dataset access, no GPU infra) but two ideas from it were folded in:
  **per-session aggregation** (smooth readings over a visit, not per-frame)
  and **multi-person tracking** (ByteTrack) as an open question for crowded
  camera views

## Database Schema (current, in `models.py`)
Five tables:
- `students` — student_id (PK), name, roll_number, department, year_of_study,
  photo_reference_path, enrolled_at
- `face_embeddings` — id, student_id (FK), embedding (JSON, 512-dim vector),
  source_photo, created_at — multiple rows per student supported
- `cameras` — camera_id (PK), location_name (e.g. "mess_entry"), description
- `detection_events` — id, student_id (FK, nullable if unmatched), camera_id
  (FK), timestamp, matched_confidence, emotion_label, emotion_confidence,
  posture_features (JSON)
- `flags` — id, student_id (FK), reason (text), score, signal_type
  (routine/emotion/posture/combined), created_at, notified, notified_at,
  status (pending/reviewed/dismissed/handled), reviewed_by, reviewed_at,
  review_notes
- `daily_activity_summary` — id, student_id (FK), date, breakfast/lunch/
  dinner attended + time, veranda_sightings, avg_emotion_negative_ratio,
  avg_head_drop — built by nightly aggregation job, baseline logic reads
  from this instead of raw events

Migration workflow documented in `README_schema.md` (delivered).

## Scripts Delivered So Far
1. `step1_face_detection.py` — OpenCV Haar cascade face detection on
   webcam/video, FPS overlay, snapshot saving
2. `step1b_face_and_pose.py` — extends Step 1 with MediaPipe Pose skeleton
   overlay + engineered posture features (`head_drop_px`, `torso_lean_px`)
3. `models.py` — full SQLAlchemy schema (5 tables above)
4. `db.py` — DB connection/session setup (env-var based credentials)
5. `alembic.ini`, `alembic/env.py`, `alembic/script.py.mako` — migration tooling
6. `enroll_student.py` — CLI script to enroll a student (name, roll, dept,
   year, reference photos) → generates ArcFace embeddings → inserts into DB
7. `step2b_recognize_and_log.py` — loads known embeddings from DB, runs
   live recognition against camera feed, logs `detection_events` with
   cooldown to avoid duplicate spam logging
8. `routine_baseline.py` — the daily batch job: (1) aggregates raw
   detections into `daily_activity_summary`, (2) computes per-student
   rolling baseline (21-day window, needs min 7 days history), (3) checks
   deviation rules (missed meal streaks, time drift, veranda drop-off,
   sustained negative emotion trend) and creates `Flag` rows, with
   duplicate-flag suppression for 7 days
9. `architecture.md` — full system architecture writeup (pipeline diagram,
   layer-by-layer breakdown, build order, open questions)

All files are in a project folder structure: `emotion_project/` with an
`alembic/` subfolder for migrations.

## Deviation Rules Currently Implemented (in routine_baseline.py)
- Missed meal streak: 3+ consecutive days missing a meal the student
  normally attends (≥50% historical attendance rate)
- Time drift: check-in time >2 std-devs from personal baseline (with
  minimum variance floor to avoid noise-triggered flags on very
  regular students)
- Veranda drop-off: daily sightings fall below 40% of personal baseline average
- Sustained negative emotion trend: supplementary signal only, requires
  routine signals to have already fired or a clear sustained negative
  trend across recent visits — never a standalone trigger

## Open Questions (not yet resolved)
1. Emotion taxonomy: keep DeepFace's 7-class output, or simplify to
   3-class (positive/neutral/negative) for more robustness in real,
   noisy footage?
2. Per-visit aggregation window: exact definition of a "session" for
   smoothing emotion/posture readings (entry-to-exit vs. fixed time window)
   — not yet implemented, flagged as next improvement
3. Multi-person-in-frame handling: does the entry-point framing avoid
   crowd misattribution, or is lightweight tracking (ByteTrack) needed?
   — not yet decided or implemented
4. Notification delivery channel: dashboard-only, email, SMS, or combination
   — not yet built (explicitly deferred by user to "work on later")
5. `MEAL_WINDOWS` and camera `location_name` values in `routine_baseline.py`
   are placeholders — need to be set to match the actual college's real
   camera names and meal timings before this runs correctly

## Not Yet Built / Next Steps
- Notification delivery (email/dashboard alert when a Flag is created) —
  explicitly deferred by user ("we can work on notification at last")
- Per-session aggregation refinement
- Multi-person tracking decision + implementation if needed
- Testing `DISTANCE_THRESHOLD` (currently 0.68 cosine distance, a starting
  guess) against real enrolled student photos to tune accuracy
- Cron/scheduling setup for `routine_baseline.py` to run nightly
- Real-world testing of camera mounting height/angle (recommended: face
  height ~5.5–6ft, 15–20° downward angle, choke-point/entry framing not
  overhead, 1080p+ minimum, 1–3m subject distance)

## User's Working Style / Preferences (for the new assistant to know)
- Prefers direct, working code over lengthy explanation-only answers
- Wants schema/migration workflow to feel like Prisma (`schema.prisma` →
  `models.py`, `prisma migrate dev` → Alembic autogenerate)
- Asks clarifying/challenging questions often ("are you sure we don't need
  to train a model?", "does YOLO recognize too?") — expects direct,
  confident technical answers, not hedging
- Comfortable working file-by-file iteratively rather than one giant dump