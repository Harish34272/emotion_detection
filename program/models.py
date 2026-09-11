"""models.py"""
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float, Boolean, Date
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Student(Base):
    __tablename__ = "students"

    student_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    roll_number = Column(String(50), unique=True, nullable=False)
    department = Column(String(80), nullable=False)
    year_of_study = Column(Integer, nullable=True)  # 1/2/3/4
    photo_reference_path = Column(Text, nullable=True)
    enrolled_at = Column(DateTime(timezone=True), server_default=func.now())

    # Soft-delete flag: when False, the student is excluded from face-recognition
    # matching (recognize_and_log.py) and from the default enrolled-students view,
    # but every historical detection/flag/summary row is kept intact. Used for
    # students who've graduated or moved out, instead of a hard DB delete, since
    # a hard delete would either orphan Flag rows (FK is non-nullable there) or
    # silently discard behavioral history that's the whole point of this system.
    is_active = Column(Boolean, nullable=False, default=True, server_default="true")
    phone_number = Column(String(20), nullable=True)

    # --- add new fields here later, e.g.: ---
    # hostel_block = Column(String(50), nullable=True)
    # phone_number = Column(String(20), nullable=True)

    embeddings = relationship("FaceEmbedding", back_populates="student", cascade="all, delete-orphan")
    detections = relationship("DetectionEvent", back_populates="student")
    flags = relationship("Flag", back_populates="student")
    leaves = relationship("StudentLeave", back_populates="student", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)
    embedding = Column(JSON, nullable=False)  # list of floats (512-dim vector), stored as JSON for portability
    angle_label = Column(String(20), nullable=True)  # "straight" / "left" / "right" / null for unstructured enrollment
    source_photo = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    student = relationship("Student", back_populates="embeddings")


class Camera(Base):
    __tablename__ = "cameras"

    camera_id = Column(Integer, primary_key=True, autoincrement=True)
    location_name = Column(String(80), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    mounted_at = Column(DateTime(timezone=True), server_default=func.now())

    detections = relationship("DetectionEvent", back_populates="camera")


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=True)
    camera_id = Column(Integer, ForeignKey("cameras.camera_id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    matched_confidence = Column(Float, nullable=True)
    emotion_label = Column(String(30), nullable=True)
    emotion_confidence = Column(Float, nullable=True)
    posture_features = Column(JSON, nullable=True)

    student = relationship("Student", back_populates="detections")
    camera = relationship("Camera", back_populates="detections")


class Flag(Base):
    """
    Raised by the batch analysis job when a student's routine/emotion/posture
    trend crosses a threshold. Purely informational -- the system NEVER acts
    on a flag itself. A human (counselor/warden) reviews it and updates status.
    """
    __tablename__ = "flags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)

    reason = Column(Text, nullable=False)             # human-readable: "missed 4 lunches in a row"
    score = Column(Float, nullable=False)              # computed risk score
    signal_type = Column(String(30), nullable=False)   # "routine" / "emotion" / "posture" / "combined"

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notified = Column(Boolean, default=False)
    notified_at = Column(DateTime(timezone=True), nullable=True)

    status = Column(String(20), default="pending")     # pending / reviewed / dismissed / handled
    reviewed_by = Column(String(120), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    review_notes = Column(Text, nullable=True)

    student = relationship("Student", back_populates="flags")


class DailyActivitySummary(Base):
    __tablename__ = "daily_activity_summary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)  # date-truncated (midnight) timestamp

    breakfast_attended = Column(Boolean, default=False)
    breakfast_time = Column(DateTime(timezone=True), nullable=True)
    lunch_attended = Column(Boolean, default=False)
    lunch_time = Column(DateTime(timezone=True), nullable=True)
    dinner_attended = Column(Boolean, default=False)
    dinner_time = Column(DateTime(timezone=True), nullable=True)

    veranda_sightings = Column(Integer, default=0)

    avg_emotion_negative_ratio = Column(Float, nullable=True)
    avg_head_drop = Column(Float, nullable=True)

    student = relationship("Student")


class WellnessScore(Base):
    """
    Daily composite wellness indicator (1.0 bad -- 5.0 good), combining meal
    attendance, emotion, and veranda presence into one number for quick
    scanning. Stored per student per day, independent of DailyActivitySummary
    -- a day with zero detections still gets a score (that's itself a real
    signal for the meal factor), whereas DailyActivitySummary rows only get
    created for students who appeared in at least one event that day.

    Individual factor values are kept alongside the final score so the number
    is never a black box -- same "interpretable over sophisticated" principle
    used for the routine_baseline z-score/streak flags. A None factor means
    that signal wasn't available that week (not that it was bad) and was
    excluded from the weighted average rather than penalizing the student.
    """
    __tablename__ = "wellness_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)  # date-truncated (midnight), same convention as DailyActivitySummary

    score = Column(Float, nullable=True)          # 1.0-5.0, None if no factors were available at all
    factors_used = Column(Integer, nullable=False, default=0)  # how many of the 3 factors contributed (0-3)

    meal_factor = Column(Float, nullable=True)     # 0.0-1.0, always available (absence itself is the signal)
    emotion_factor = Column(Float, nullable=True)  # 0.0-1.0, None if no emotion data in the window
    veranda_factor = Column(Float, nullable=True)  # 0.0-1.0, None if no meaningful baseline to compare against

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    student = relationship("Student")


class StudentLeave(Base):
    """
    Approved leave for a specific student (home visit, medical, etc.).
    While a leave is active, routine_baseline.py skips flagging that student.
    Registered by the warden via the review app.
    """
    __tablename__ = "student_leaves"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)

    start_date = Column(Date, nullable=False)   # inclusive
    end_date = Column(Date, nullable=False)     # inclusive

    reason = Column(String(200), nullable=True)       # e.g. "home visit", "medical"
    approved_by = Column(String(120), nullable=True)  # warden name / staff ID
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    student = relationship("Student", back_populates="leaves")

    def covers(self, d):
        """Return True if date d (a datetime.date) falls within this leave."""
        return self.start_date <= d <= self.end_date


class HostelClosure(Base):
    """
    Hostel-wide closure (holidays, maintenance, etc.).
    While a closure is active, routine_baseline.py skips flagging ALL students.
    Registered by the warden via the review app.
    """
    __tablename__ = "hostel_closures"

    id = Column(Integer, primary_key=True, autoincrement=True)

    start_date = Column(Date, nullable=False)   # inclusive
    end_date = Column(Date, nullable=False)     # inclusive

    reason = Column(String(200), nullable=True)       # e.g. "Diwali holidays"
    registered_by = Column(String(120), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def covers(self, d):
        """Return True if date d (a datetime.date) falls within this closure."""
        return self.start_date <= d <= self.end_date