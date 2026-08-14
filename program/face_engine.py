
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

DET_SIZE = (320, 320)

PROVIDERS = ["CPUExecutionProvider"]

_app = None  # lazy-loaded singleton so the model only loads once per process


def get_face_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(providers=PROVIDERS)
        _app.prepare(ctx_id=0, det_size=DET_SIZE)
    return _app


def get_faces(frame):
    """Returns a list of InsightFace face objects (bbox, kps, embedding, etc.)."""
    app = get_face_app()
    return app.get(frame)


def cosine_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def open_capture(source):
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, str) and source.startswith("gst:"):
        pipeline = source[len("gst:"):]
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def resize_for_display(frame, max_width=960):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))
