import platform
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

DET_SIZE = (320, 320)
PROVIDERS = ["CPUExecutionProvider"]

_app = None


def get_face_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(providers=PROVIDERS)
        _app.prepare(ctx_id=0, det_size=DET_SIZE)
    return _app


def get_faces(frame):
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

    # Pick the right backend for the current OS.
    # Only apply V4L2 on Linux with an integer index / device path;
    # RTSP URLs and other string sources should use OpenCV's default backend.
    system = platform.system()
    if isinstance(source, int) or (isinstance(source, str) and source.startswith("/dev/")):
        backend = cv2.CAP_V4L2 if system == "Linux" else cv2.CAP_DSHOW if system == "Windows" else 0
    else:
        backend = 0  # let OpenCV auto-select for RTSP/URLs

    cap = cv2.VideoCapture(source, backend)
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