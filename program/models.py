
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float, Boolean
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

    # --- add new fields here later, e.g.: ---
    # hostel_block = Column(String(50), nullable=True)
    # phone_number = Column(String(20), nullable=True)

    embeddings = relationship("FaceEmbedding", back_populates="student", cascade="all, delete-orphan")
    detections = relationship("DetectionEvent", back_populates="student")
    flags = relationship("Flag", back_populates="student")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)
    embedding = Column(JSON, nullable=False)  # list of floats (512-dim vector), stored as JSON for portability
    angle_label = Column(String(20), nullable=True)  # "straight" / "left" / "right" / null for unstructured enrollment
    source_photo = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    student = relationship("Student", back_populates="embeddings")

    # NOTE: once you're ready for production-scale similarity search,
    # switch `embedding` to pgvector's VECTOR(512) type + add an index.
    # That's a schema change like any other -- just edit + migrate.


class Camera(Base):
    __tablename__ = "cameras"

    camera_id = Column(Integer, primary_key=True, autoincrement=True)
    location_name = Column(String(80), nullable=False, unique=True)  # e.g. "mess_entry"
    description = Column(Text, nullable=True)
    mounted_at = Column(DateTime(timezone=True), server_default=func.now())

    detections = relationship("DetectionEvent", back_populates="camera")


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=True)  # nullable: face seen but not matched
    camera_id = Column(Integer, ForeignKey("cameras.camera_id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    matched_confidence = Column(Float, nullable=True)   # how confident the face-match was
    emotion_label = Column(String(30), nullable=True)   # e.g. "neutral", "sad"
    emotion_confidence = Column(Float, nullable=True)
    posture_features = Column(JSON, nullable=True)      # head_drop, torso_lean, etc.

    # --- add new fields here later, e.g.: ---
    # sitting_alone = Column(Integer, nullable=True)  # 1/0

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
    """
    One row per student per day, built by the nightly aggregation job from
    raw detection_events. The baseline/deviation logic reads from this --
    much cheaper than re-scanning raw events every time.
    """
    __tablename__ = "daily_activity_summary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.student_id"), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)  # date-truncated (midnight) timestamp

    # meal attendance -- mess + cafeteria treated as one combined "meal" signal
    breakfast_attended = Column(Boolean, default=False)
    breakfast_time = Column(DateTime(timezone=True), nullable=True)
    lunch_attended = Column(Boolean, default=False)
    lunch_time = Column(DateTime(timezone=True), nullable=True)
    dinner_attended = Column(Boolean, default=False)
    dinner_time = Column(DateTime(timezone=True), nullable=True)

    # veranda / general presence signal (kept separate -- different meaning than meals)
    veranda_sightings = Column(Integer, default=0)

    # affect signals, averaged across all detections that day
    avg_emotion_negative_ratio = Column(Float, nullable=True)  # fraction of sad/angry/fear readings
    avg_head_drop = Column(Float, nullable=True)               # posture signal average

    student = relationship("Student")
