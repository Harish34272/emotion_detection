"""review_app.py"""
import os
import re
from datetime import date, datetime, timezone

import cv2
import numpy as np
from flask import Flask, redirect, url_for, request, send_from_directory, abort, jsonify
from db import get_session
from models import Flag, Student, FaceEmbedding, DetectionEvent, StudentLeave, HostelClosure
from face_engine import get_faces
from generate_dashboard import (
    build_detections_section,
    build_html,
    build_student_detail_data,
    get_latest_wellness_scores,
    wellness_badge,
)

app = Flask(__name__)

VALID_STATUSES = {"pending", "reviewed", "dismissed", "handled"}

CROPS_ROOT = os.path.abspath("source_photos/detections")

# ---- enrollment config -----------------------------------------------------
# Mirrors enroll_student.py's layout/logic so photos captured via the warden
# UI land in the exact same place and DB shape as the CLI enrollment tools.
ENROLL_POSES = ("straight", "left", "right")
ENROLL_SAVE_DIR = os.path.abspath("source_photos/enrollment")
MIN_VALID_POSES = 2


# ---- static file serving -------------------------------------------------

@app.route("/crops/<roll_number>/<filename>")
def serve_crop(roll_number, filename):
    student_dir = os.path.join(CROPS_ROOT, roll_number)
    if not os.path.abspath(student_dir).startswith(CROPS_ROOT):
        abort(403)
    return send_from_directory(student_dir, filename)


@app.route("/student/<int:student_id>/photos")
def student_photos(student_id):
    session = get_session()
    try:
        student = session.query(Student).filter_by(student_id=student_id).first()
        if not student:
            return jsonify([])
        student_dir = os.path.join(CROPS_ROOT, student.roll_number)
        if not os.path.isdir(student_dir):
            return jsonify([])
        files = [f for f in os.listdir(student_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(student_dir, f)), reverse=True)
        return jsonify([
            {"filename": f, "url": f"/crops/{student.roll_number}/{f}"}
            for f in files[:8]
        ])
    finally:
        session.close()


# ---- enrollment ------------------------------------------------------------

def _validate_and_embed_frame(frame):
    """
    Same rule as enroll_student.py/enroll_student_live.py: run InsightFace,
    accept only if exactly one clear face is found. Returns (embedding, error).
    """
    faces = get_faces(frame)
    if len(faces) == 0:
        return None, "no face detected"
    if len(faces) > 1:
        return None, f"{len(faces)} faces detected — expected exactly 1"
    return faces[0].embedding, None


@app.route("/enroll")
def enroll_page():
    return ENROLL_PAGE_HTML


@app.route("/enroll/submit", methods=["POST"])
def enroll_submit():
    name = request.form.get("name", "").strip()
    roll = request.form.get("roll", "").strip()
    dept = request.form.get("dept", "").strip()
    year_raw = request.form.get("year", "").strip()
    year = int(year_raw) if year_raw.isdigit() else None
    phone = request.form.get("phone", "").strip() or None

    if not name or not roll or not dept:
        return jsonify({"ok": False, "error": "Name, roll number, and department are required."}), 400

    safe_roll = roll.replace("/", "_")
    student_dir = os.path.join(ENROLL_SAVE_DIR, safe_roll)
    os.makedirs(student_dir, exist_ok=True)

    pose_results = {}
    captured_photos = {}
    captured_embeddings = {}

    for pose in ENROLL_POSES:
        file = request.files.get(pose)
        if not file or file.filename == "":
            pose_results[pose] = {"ok": False, "detail": "no photo provided"}
            continue

        data = file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            pose_results[pose] = {"ok": False, "detail": "could not decode image"}
            continue

        embedding, err = _validate_and_embed_frame(frame)
        if err:
            pose_results[pose] = {"ok": False, "detail": err}
            continue

        dest_path = os.path.join(student_dir, f"{pose}.jpg")
        cv2.imwrite(dest_path, frame)
        captured_photos[pose] = dest_path
        captured_embeddings[pose] = embedding
        pose_results[pose] = {"ok": True, "detail": "saved"}

    if len(captured_embeddings) < MIN_VALID_POSES:
        return jsonify({
            "ok": False,
            "error": f"Need at least {MIN_VALID_POSES} valid poses (got {len(captured_embeddings)}).",
            "poses": pose_results,
        }), 400

    session = get_session()
    try:
        existing = session.query(Student).filter_by(roll_number=roll).first()
        if existing:
            student = existing
            created_new = False
        else:
            student = Student(
                name=name,
                roll_number=roll,
                department=dept,
                year_of_study=year,
                phone_number=phone,
                photo_reference_path=captured_photos.get("straight"),
            )
            session.add(student)
            session.commit()
            created_new = True

        student_id = student.student_id
        student_name = student.name

        added = 0
        for pose, embedding in captured_embeddings.items():
            fe = FaceEmbedding(
                student_id=student_id,
                embedding=embedding.tolist(),
                angle_label=pose,
                source_photo=captured_photos[pose],
            )
            session.add(fe)
            added += 1
        session.commit()

        return jsonify({
            "ok": True,
            "student_id": student_id,
            "student_name": student_name,
            "created_new": created_new,
            "embeddings_added": added,
            "poses": pose_results,
        })
    except Exception as e:
        session.rollback()
        return jsonify({"ok": False, "error": f"DB save failed: {e}", "poses": pose_results}), 500
    finally:
        session.close()


ENROLL_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Enroll Student — Wellness Monitoring</title>
<style>
  :root {
    --bg: #F5F6F8; --panel: #FFFFFF; --ink: #1F2933; --muted: #6B7280;
    --border: #E2E5EA; --accent: #3E5C76; --accent-soft: #E9EEF3;
    --sans: 'IBM Plex Sans', 'Inter', -apple-system, sans-serif;
    --mono: 'IBM Plex Mono', 'SF Mono', Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans); line-height:1.5; }
  header { background:var(--panel); border-bottom:1px solid var(--border); padding:24px 40px;
           display:flex; align-items:center; justify-content:space-between; }
  header h1 { margin:0; font-size:1.3rem; font-weight:600; }
  header a { color:var(--accent); font-size:0.85rem; text-decoration:none; font-family:var(--mono); }
  header a:hover { text-decoration:underline; }
  main { max-width:820px; margin:0 auto; padding:32px 40px 80px; }
  section { background:var(--panel); border:1px solid var(--border); border-radius:6px;
            margin-bottom:24px; padding:24px; }
  section h2 { margin:0 0 16px; font-size:0.85rem; font-weight:600; text-transform:uppercase;
               letter-spacing:0.04em; color:var(--accent); }
  .field-row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:14px; }
  label.field { display:flex; flex-direction:column; gap:4px; font-size:0.78rem; color:var(--muted); flex:1; min-width:160px; }
  input[type=text], input[type=number] { padding:8px 10px; border:1px solid var(--border); border-radius:4px; font-size:0.9rem; }
  .pose-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px,1fr)); gap:16px; }
  .pose-card { border:1px dashed var(--border); border-radius:6px; padding:14px; text-align:center; }
  .pose-card h3 { margin:0 0 10px; font-size:0.8rem; text-transform:uppercase; color:var(--muted); letter-spacing:0.04em; }
  .pose-preview { width:100%; height:150px; object-fit:cover; border-radius:4px; background:var(--bg);
                   display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:0.8rem;
                   border:1px solid var(--border); margin-bottom:10px; }
  .pose-preview img { width:100%; height:100%; object-fit:cover; border-radius:4px; }
  .pose-buttons { display:flex; gap:8px; justify-content:center; }
  .btn { background:var(--accent); color:white; border:none; padding:6px 12px; border-radius:4px;
         font-size:0.78rem; cursor:pointer; }
  .btn:hover { opacity:0.85; }
  .btn.secondary { background:var(--accent-soft); color:var(--accent); }
  .btn.submit { padding:10px 24px; font-size:0.9rem; }
  .pose-status { margin-top:8px; font-size:0.75rem; font-family:var(--mono); color:var(--muted); }
  .pose-status.ok { color:#1F7A4D; }
  .pose-status.err { color:#B45309; }
  input[type=file] { display:none; }
  #resultBox { margin-top:16px; padding:12px 16px; border-radius:4px; font-size:0.85rem; display:none; }
  #resultBox.ok { background:#EAF6EF; color:#1F7A4D; display:block; }
  #resultBox.err { background:#FBEAEA; color:#B4231E; display:block; }
  .camera-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.6);
                     align-items:center; justify-content:center; z-index:200; }
  .camera-panel { background:var(--panel); border-radius:8px; padding:20px; text-align:center; }
  .camera-panel video, .camera-panel canvas { width:360px; height:270px; border-radius:6px; background:#000; }
  .camera-actions { margin-top:12px; display:flex; gap:10px; justify-content:center; }
</style>
</head>
<body>
<header>
  <h1>Enroll New Student</h1>
  <a href="/">&larr; Back to dashboard</a>
</header>
<main>
  <section>
    <h2>Student Details</h2>
    <div class="field-row">
      <label class="field">Full name
        <input type="text" id="f-name" required>
      </label>
      <label class="field">Roll number
        <input type="text" id="f-roll" required>
      </label>
    </div>
    <div class="field-row">
      <label class="field">Department
        <input type="text" id="f-dept" required>
      </label>
      <label class="field">Year of study
        <input type="number" id="f-year" min="1" max="6">
      </label>
    </div>
    <div class="field-row">
      <label class="field">Phone number
        <input type="text" id="f-phone">
      </label>
    </div>
  </section>

  <section>
    <h2>Face Photos (need at least 2 of 3)</h2>
    <div class="pose-grid" id="poseGrid"></div>
  </section>

  <button class="btn submit" id="submitBtn">Enroll Student</button>
  <div id="resultBox"></div>
</main>

<div class="camera-overlay" id="cameraOverlay">
  <div class="camera-panel">
    <video id="cameraVideo" autoplay playsinline></video>
    <canvas id="cameraCanvas" style="display:none;"></canvas>
    <div class="camera-actions">
      <button class="btn" id="captureBtn">Capture</button>
      <button class="btn secondary" id="cancelCameraBtn">Cancel</button>
    </div>
  </div>
</div>

<script>
const POSES = [
  {key: "straight", label: "Straight", hint: "Look directly at camera"},
  {key: "left", label: "Left", hint: "Turn ~60-85° to your left"},
  {key: "right", label: "Right", hint: "Turn ~60-85° to your right"},
];

const photos = {}; // pose -> Blob
let activePose = null;
let cameraStream = null;

const poseGrid = document.getElementById("poseGrid");
POSES.forEach(p => {
  const card = document.createElement("div");
  card.className = "pose-card";
  card.innerHTML = `
    <h3>${p.label}</h3>
    <div class="pose-preview" id="preview-${p.key}">${p.hint}</div>
    <div class="pose-buttons">
      <label class="btn secondary" style="margin:0;">
        Upload
        <input type="file" accept="image/*" id="file-${p.key}">
      </label>
      <button class="btn" data-pose="${p.key}" data-action="camera">Camera</button>
    </div>
    <div class="pose-status" id="status-${p.key}"></div>
  `;
  poseGrid.appendChild(card);

  document.getElementById(`file-${p.key}`).addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) setPosePhoto(p.key, file);
  });
});

poseGrid.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-action='camera']");
  if (btn) openCamera(btn.dataset.pose);
});

function setPosePhoto(pose, blob) {
  photos[pose] = blob;
  const preview = document.getElementById(`preview-${pose}`);
  const url = URL.createObjectURL(blob);
  preview.innerHTML = `<img src="${url}">`;
  const status = document.getElementById(`status-${pose}`);
  status.textContent = "Ready";
  status.className = "pose-status";
}

async function openCamera(pose) {
  activePose = pose;
  document.getElementById("cameraOverlay").style.display = "flex";
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    document.getElementById("cameraVideo").srcObject = cameraStream;
  } catch (err) {
    alert("Could not access camera: " + err.message);
    closeCamera();
  }
}

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  document.getElementById("cameraOverlay").style.display = "none";
  activePose = null;
}

document.getElementById("cancelCameraBtn").addEventListener("click", closeCamera);

document.getElementById("captureBtn").addEventListener("click", () => {
  const video = document.getElementById("cameraVideo");
  const canvas = document.getElementById("cameraCanvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    if (blob && activePose) setPosePhoto(activePose, blob);
    closeCamera();
  }, "image/jpeg", 0.92);
});

document.getElementById("submitBtn").addEventListener("click", async () => {
  const name = document.getElementById("f-name").value.trim();
  const roll = document.getElementById("f-roll").value.trim();
  const dept = document.getElementById("f-dept").value.trim();
  const year = document.getElementById("f-year").value.trim();
  const phone = document.getElementById("f-phone").value.trim();
  const resultBox = document.getElementById("resultBox");

  if (!name || !roll || !dept) {
    resultBox.className = "err";
    resultBox.textContent = "Name, roll number, and department are required.";
    return;
  }
  const providedCount = Object.keys(photos).length;
  if (providedCount < 2) {
    resultBox.className = "err";
    resultBox.textContent = "Please provide at least 2 pose photos.";
    return;
  }

  const fd = new FormData();
  fd.append("name", name);
  fd.append("roll", roll);
  fd.append("dept", dept);
  if (year) fd.append("year", year);
  if (phone) fd.append("phone", phone);
  for (const [pose, blob] of Object.entries(photos)) {
    fd.append(pose, blob, `${pose}.jpg`);
  }

  const submitBtn = document.getElementById("submitBtn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Enrolling…";

  try {
    const res = await fetch("/enroll/submit", { method: "POST", body: fd });
    const data = await res.json();

    for (const pose of Object.keys(photos)) {
      const st = document.getElementById(`status-${pose}`);
      const r = data.poses && data.poses[pose];
      if (r) {
        st.textContent = r.detail;
        st.className = "pose-status " + (r.ok ? "ok" : "err");
      }
    }

    if (data.ok) {
      resultBox.className = "ok";
      resultBox.textContent = `Success — ${data.student_name} (${data.created_new ? "new student" : "existing student"}), ${data.embeddings_added} embedding(s) saved.`;
    } else {
      resultBox.className = "err";
      resultBox.textContent = data.error || "Enrollment failed.";
    }
  } catch (err) {
    resultBox.className = "err";
    resultBox.textContent = "Request failed: " + err.message;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Enroll Student";
  }
});
</script>
</body>
</html>"""


# ---- students section (editable — deactivate / reactivate) ----------------

def build_students_section_editable(session):
    """
    Same data as generate_dashboard.build_students_section, but shows both
    active and inactive students with a Deactivate/Reactivate action per row.
    This is the only place inactive students are visible day-to-day -- the
    static dashboard snapshot hides them by default.
    """
    from html import escape
    students = (
        session.query(Student)
        .order_by(Student.is_active.desc(), Student.name)
        .all()
    )
    wellness = get_latest_wellness_scores(session)
    rows = []
    for s in students:
        emb_count = session.query(FaceEmbedding).filter_by(student_id=s.student_id).count()
        det_count = session.query(DetectionEvent).filter_by(student_id=s.student_id).count()

        if s.is_active:
            status_badge = '<span class="status-pill" style="background:#1F7A4D">active</span>'
            action = f"""
            <form method="POST" action="/students/{s.student_id}/deactivate"
                  onsubmit="return confirm('Deactivate {escape(s.name)}? They will stop being matched by recognition. All history is kept and this can be undone.')">
              <button type="submit" class="btn-danger">Deactivate</button>
            </form>"""
        else:
            status_badge = '<span class="status-pill" style="background:#6B7280">inactive</span>'
            action = f"""
            <form method="POST" action="/students/{s.student_id}/reactivate">
              <button type="submit" class="btn">Reactivate</button>
            </form>"""

        rows.append(f"""
        <tr class="student-row">
          <td onclick="showStudentDetail({s.student_id})" style="cursor:pointer">{s.student_id}</td>
          <td onclick="showStudentDetail({s.student_id})" style="cursor:pointer">{escape(s.name)}</td>
          <td onclick="showStudentDetail({s.student_id})" style="cursor:pointer">{escape(s.roll_number)}</td>
          <td onclick="showStudentDetail({s.student_id})" style="cursor:pointer">{escape(s.department)}</td>
          <td onclick="showStudentDetail({s.student_id})" style="cursor:pointer">{s.year_of_study or '-'}</td>
          <td>{wellness_badge(wellness.get(s.student_id))}</td>
          <td>{emb_count}</td>
          <td>{det_count}</td>
          <td>{status_badge}</td>
          <td>{action}</td>
        </tr>""")

    if not rows:
        return '<p class="empty">No students enrolled yet.</p>'
    return f"""
    <table>
      <thead><tr>
        <th>ID</th><th>Name</th><th>Roll No.</th><th>Dept</th><th>Year</th>
        <th>Wellness</th><th>Embeddings</th><th>Detections</th><th>Status</th><th></th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


@app.route("/students/<int:student_id>/deactivate", methods=["POST"])
def deactivate_student(student_id):
    session = get_session()
    try:
        student = session.query(Student).filter_by(student_id=student_id).first()
        if student:
            student.is_active = False
            session.commit()
    finally:
        session.close()
    return redirect(url_for("index"))


@app.route("/students/<int:student_id>/reactivate", methods=["POST"])
def reactivate_student(student_id):
    session = get_session()
    try:
        student = session.query(Student).filter_by(student_id=student_id).first()
        if student:
            student.is_active = True
            session.commit()
    finally:
        session.close()
    return redirect(url_for("index"))


# ---- bulk actions (department / year / roll-number range) -----------------

def _natural_sort_key(s):
    """
    Splits a roll number into (type_flag, value) chunks so ranges compare
    correctly regardless of format -- "9" vs "10" won't misorder the way
    plain string comparison would, and mixed formats (e.g. a department
    prefix) won't raise a TypeError, since every chunk is a (int, value)
    tuple and the leading int always makes two chunks comparable.
    """
    tokens = [t for t in re.split(r"(\d+)", s or "") if t != ""]
    key = []
    for t in tokens:
        if t.isdigit():
            key.append((0, int(t)))
        else:
            key.append((1, t.lower()))
    return key


def get_matching_students(session, filter_type, department=None, year=None, roll_from=None, roll_to=None):
    """
    Returns active students matching the given bulk filter. Only active
    students are considered -- there's no reason to bulk-deactivate or
    bulk-register-leave for someone already inactive.
    """
    base = session.query(Student).filter(Student.is_active.is_(True))

    if filter_type == "department":
        if not department:
            return []
        return base.filter(Student.department == department).order_by(Student.name).all()

    if filter_type == "year":
        if not year:
            return []
        return base.filter(Student.year_of_study == year).order_by(Student.name).all()

    if filter_type == "dept_year":
        if not department or not year:
            return []
        return (base.filter(Student.department == department, Student.year_of_study == year)
                .order_by(Student.name).all())

    if filter_type == "roll_range":
        if not roll_from or not roll_to:
            return []
        key_from, key_to = _natural_sort_key(roll_from), _natural_sort_key(roll_to)
        if key_to < key_from:
            key_from, key_to = key_to, key_from
        matched = [s for s in base.all() if key_from <= _natural_sort_key(s.roll_number) <= key_to]
        matched.sort(key=lambda s: _natural_sort_key(s.roll_number))
        return matched

    return []


def _matches_from_request(session, data):
    return get_matching_students(
        session,
        data.get("filter_type"),
        department=(data.get("department") or "").strip() or None,
        year=int(data["year"]) if str(data.get("year") or "").strip().isdigit() else None,
        roll_from=(data.get("roll_from") or "").strip() or None,
        roll_to=(data.get("roll_to") or "").strip() or None,
    )


@app.route("/bulk")
def bulk_page():
    from html import escape
    session = get_session()
    try:
        departments = [
            d[0] for d in session.query(Student.department)
            .filter(Student.is_active.is_(True)).distinct().order_by(Student.department).all()
        ]
        years = sorted({
            y[0] for y in session.query(Student.year_of_study)
            .filter(Student.is_active.is_(True), Student.year_of_study.isnot(None)).distinct().all()
        })
    finally:
        session.close()

    dept_options = "".join(f'<option value="{escape(d)}">{escape(d)}</option>' for d in departments)
    year_options = "".join(f'<option value="{y}">Year {y}</option>' for y in years)
    return (BULK_PAGE_HTML
            .replace("{{DEPT_OPTIONS}}", dept_options)
            .replace("{{YEAR_OPTIONS}}", year_options))


@app.route("/bulk/preview", methods=["POST"])
def bulk_preview():
    data = request.get_json(silent=True) or {}
    session = get_session()
    try:
        students = _matches_from_request(session, data)
        return jsonify({
            "ok": True,
            "count": len(students),
            "students": [
                {"id": s.student_id, "name": s.name, "roll_number": s.roll_number,
                 "department": s.department, "year_of_study": s.year_of_study}
                for s in students
            ],
        })
    finally:
        session.close()


@app.route("/bulk/deactivate", methods=["POST"])
def bulk_deactivate():
    data = request.get_json(silent=True) or {}
    session = get_session()
    try:
        students = _matches_from_request(session, data)
        for s in students:
            s.is_active = False
        session.commit()
        return jsonify({"ok": True, "deactivated": len(students)})
    finally:
        session.close()


@app.route("/bulk/leave", methods=["POST"])
def bulk_leave():
    data = request.get_json(silent=True) or {}
    try:
        start_date = date.fromisoformat(data["start_date"])
        end_date = date.fromisoformat(data["end_date"])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"ok": False, "error": f"Invalid dates: {e}"}), 400
    if end_date < start_date:
        return jsonify({"ok": False, "error": "End date must be on or after start date."}), 400

    reason = (data.get("reason") or "").strip() or None
    approved_by = (data.get("approved_by") or "").strip() or None

    session = get_session()
    try:
        students = _matches_from_request(session, data)
        for s in students:
            session.add(StudentLeave(
                student_id=s.student_id, start_date=start_date, end_date=end_date,
                reason=reason, approved_by=approved_by,
            ))
        session.commit()
        return jsonify({"ok": True, "leaves_created": len(students)})
    finally:
        session.close()


BULK_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bulk Actions — Wellness Monitoring</title>
<style>
  :root {
    --bg: #F5F6F8; --panel: #FFFFFF; --ink: #1F2933; --muted: #6B7280;
    --border: #E2E5EA; --accent: #3E5C76; --accent-soft: #E9EEF3;
    --sans: 'IBM Plex Sans', 'Inter', -apple-system, sans-serif;
    --mono: 'IBM Plex Mono', 'SF Mono', Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans); line-height:1.5; }
  header { background:var(--panel); border-bottom:1px solid var(--border); padding:24px 40px;
           display:flex; align-items:center; justify-content:space-between; }
  header h1 { margin:0; font-size:1.3rem; font-weight:600; }
  header a { color:var(--accent); font-size:0.85rem; text-decoration:none; font-family:var(--mono); }
  header a:hover { text-decoration:underline; }
  main { max-width:820px; margin:0 auto; padding:32px 40px 80px; }
  section { background:var(--panel); border:1px solid var(--border); border-radius:6px;
            margin-bottom:24px; padding:24px; }
  section h2 { margin:0 0 16px; font-size:0.85rem; font-weight:600; text-transform:uppercase;
               letter-spacing:0.04em; color:var(--accent); }
  .field-row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:14px; align-items:flex-end; }
  label.field { display:flex; flex-direction:column; gap:4px; font-size:0.78rem; color:var(--muted); flex:1; min-width:160px; }
  input[type=text], input[type=date], select { padding:8px 10px; border:1px solid var(--border); border-radius:4px; font-size:0.9rem; }
  .btn { background:var(--accent); color:white; border:none; padding:8px 16px; border-radius:4px;
         font-size:0.85rem; cursor:pointer; }
  .btn:hover { opacity:0.85; }
  .btn.danger { background:#DC2626; }
  .btn:disabled { opacity:0.5; cursor:not-allowed; }
  .hint { color:var(--muted); font-size:0.78rem; margin-top:-6px; margin-bottom:12px; }
  table { width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:12px; }
  th { text-align:left; padding:8px 12px; background:var(--bg); color:var(--muted); font-weight:600;
       font-size:0.7rem; text-transform:uppercase; border-bottom:1px solid var(--border); }
  td { padding:8px 12px; border-bottom:1px solid var(--border); }
  #resultBox { margin-top:16px; padding:12px 16px; border-radius:4px; font-size:0.85rem; display:none; }
  #resultBox.ok { background:#EAF6EF; color:#1F7A4D; display:block; }
  #resultBox.err { background:#FBEAEA; color:#B4231E; display:block; }
  #actionSection, #previewSection { display:none; }
  #matchCount { font-weight:600; }
</style>
</head>
<body>
<header>
  <h1>Bulk Actions</h1>
  <a href="/">&larr; Back to dashboard</a>
</header>
<main>
  <section>
    <h2>Select Students</h2>
    <div class="field-row">
      <label class="field">Filter by
        <select id="filterType">
          <option value="department">Department</option>
          <option value="year">Year of study</option>
          <option value="dept_year">Department + Year</option>
          <option value="roll_range">Roll number range</option>
        </select>
      </label>
    </div>
    <div class="field-row" id="deptField">
      <label class="field">Department
        <select id="f-department">{{DEPT_OPTIONS}}</select>
      </label>
    </div>
    <div class="field-row" id="yearField">
      <label class="field">Year
        <select id="f-year">{{YEAR_OPTIONS}}</select>
      </label>
    </div>
    <div class="field-row" id="rollField">
      <label class="field">From roll no.
        <input type="text" id="f-roll-from" placeholder="e.g. 21011050">
      </label>
      <label class="field">To roll no.
        <input type="text" id="f-roll-to" placeholder="e.g. 21011090">
      </label>
    </div>
    <p class="hint" id="rollHint">Range is matched numerically where possible, not as plain text, so order won't matter and mismatched formats are handled safely — always double-check the preview list below.</p>
    <button class="btn" id="previewBtn">Preview Matches</button>
  </section>

  <section id="previewSection">
    <h2>Matched Students — <span id="matchCount">0</span></h2>
    <div id="previewTable"></div>
  </section>

  <section id="actionSection">
    <h2>Apply Action</h2>
    <div class="field-row">
      <button class="btn danger" id="deactivateBtn">Deactivate All Matched</button>
    </div>
    <hr style="border:none; border-top:1px solid var(--border); margin:20px 0;">
    <h2 style="margin-top:0;">Or Register Leave for All Matched</h2>
    <div class="field-row">
      <label class="field">From <input type="date" id="leave-start" required></label>
      <label class="field">To <input type="date" id="leave-end" required></label>
      <label class="field">Reason <input type="text" id="leave-reason" placeholder="e.g. department tour"></label>
      <label class="field">Approved by <input type="text" id="leave-approved" placeholder="Warden name"></label>
    </div>
    <button class="btn" id="leaveBtn">Register Leave for All Matched</button>
  </section>

  <div id="resultBox"></div>
</main>

<script>
const filterType = document.getElementById("filterType");
const deptField = document.getElementById("deptField");
const yearField = document.getElementById("yearField");
const rollField = document.getElementById("rollField");
let lastMatches = [];

function syncFields() {
  const t = filterType.value;
  deptField.style.display = (t === "department" || t === "dept_year") ? "flex" : "none";
  yearField.style.display = (t === "year" || t === "dept_year") ? "flex" : "none";
  rollField.style.display = (t === "roll_range") ? "flex" : "none";
}
filterType.addEventListener("change", syncFields);
syncFields();

function currentFilterPayload() {
  return {
    filter_type: filterType.value,
    department: document.getElementById("f-department").value,
    year: document.getElementById("f-year").value,
    roll_from: document.getElementById("f-roll-from").value.trim(),
    roll_to: document.getElementById("f-roll-to").value.trim(),
  };
}

document.getElementById("previewBtn").addEventListener("click", async () => {
  const resultBox = document.getElementById("resultBox");
  resultBox.className = ""; resultBox.style.display = "none";
  const res = await fetch("/bulk/preview", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(currentFilterPayload()),
  });
  const data = await res.json();
  lastMatches = data.students || [];
  document.getElementById("matchCount").textContent = data.count;
  const previewSection = document.getElementById("previewSection");
  const actionSection = document.getElementById("actionSection");

  if (data.count === 0) {
    document.getElementById("previewTable").innerHTML = '<p class="hint">No active students matched this filter.</p>';
    previewSection.style.display = "block";
    actionSection.style.display = "none";
    return;
  }

  const rows = lastMatches.map(s => `
    <tr><td>${s.name}</td><td>${s.roll_number}</td><td>${s.department}</td><td>${s.year_of_study || '-'}</td></tr>
  `).join("");
  document.getElementById("previewTable").innerHTML = `
    <table><thead><tr><th>Name</th><th>Roll No.</th><th>Dept</th><th>Year</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
  previewSection.style.display = "block";
  actionSection.style.display = "block";
});

document.getElementById("deactivateBtn").addEventListener("click", async () => {
  if (lastMatches.length === 0) return;
  if (!confirm(`Deactivate ${lastMatches.length} student(s)? They will stop being matched by recognition. This is reversible per-student from the dashboard.`)) return;

  const res = await fetch("/bulk/deactivate", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(currentFilterPayload()),
  });
  const data = await res.json();
  const resultBox = document.getElementById("resultBox");
  if (data.ok) {
    resultBox.className = "ok";
    resultBox.textContent = `Deactivated ${data.deactivated} student(s).`;
  } else {
    resultBox.className = "err";
    resultBox.textContent = data.error || "Bulk deactivate failed.";
  }
});

document.getElementById("leaveBtn").addEventListener("click", async () => {
  if (lastMatches.length === 0) return;
  const start = document.getElementById("leave-start").value;
  const end = document.getElementById("leave-end").value;
  const resultBox = document.getElementById("resultBox");
  if (!start || !end) {
    resultBox.className = "err";
    resultBox.textContent = "From and To dates are required.";
    return;
  }
  if (!confirm(`Register leave (${start} to ${end}) for ${lastMatches.length} student(s)?`)) return;

  const payload = currentFilterPayload();
  payload.start_date = start;
  payload.end_date = end;
  payload.reason = document.getElementById("leave-reason").value.trim();
  payload.approved_by = document.getElementById("leave-approved").value.trim();

  const res = await fetch("/bulk/leave", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (data.ok) {
    resultBox.className = "ok";
    resultBox.textContent = `Registered leave for ${data.leaves_created} student(s).`;
  } else {
    resultBox.className = "err";
    resultBox.textContent = data.error || "Bulk leave registration failed.";
  }
});
</script>
</body>
</html>"""


# ---- flag section (editable) ---------------------------------------------

def build_flags_section_editable(session):
    from html import escape
    flags = (
        session.query(Flag, Student)
        .join(Student, Flag.student_id == Student.student_id)
        .order_by(Flag.created_at.desc())
        .all()
    )
    rows = []
    for flag, student in flags:
        created = flag.created_at.strftime('%Y-%m-%d %H:%M') if flag.created_at else "-"
        options = "".join(
            f'<option value="{s}" {"selected" if s == flag.status else ""}>{s}</option>'
            for s in VALID_STATUSES
        )
        reviewed_note = ""
        if flag.reviewed_by:
            reviewed_when = flag.reviewed_at.strftime('%Y-%m-%d %H:%M') if flag.reviewed_at else "-"
            reviewed_note = (
                f'<div class="muted" style="margin-top:4px; font-size:0.75rem;">'
                f'Last reviewed by {escape(flag.reviewed_by)} on {reviewed_when}'
                f'{" — " + escape(flag.review_notes) if flag.review_notes else ""}</div>'
            )
        rows.append(f"""
        <tr>
          <td class="muted" onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{created}</td>
          <td onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{escape(student.name)}</td>
          <td onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{escape(student.roll_number)}</td>
          <td onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{escape(flag.signal_type)}</td>
          <td onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{escape(flag.reason)}</td>
          <td onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{flag.score:.2f}</td>
          <td>
            <form method="POST" action="/flags/{flag.id}/update" class="status-form" style="flex-wrap:wrap;">
              <select name="status">{options}</select>
              <input type="text" name="reviewed_by" placeholder="Your name" required
                     value="{escape(flag.reviewed_by or '')}" style="width:100px;">
              <input type="text" name="review_notes" placeholder="Notes (optional)"
                     value="{escape(flag.review_notes or '')}" style="width:140px;">
              <button type="submit">Save</button>
            </form>
            {reviewed_note}
          </td>
        </tr>""")
    if not rows:
        return '<p class="empty">No flags raised yet. This is a good thing.</p>'
    return f"""
    <table>
      <thead><tr>
        <th>Raised</th><th>Student</th><th>Roll No.</th><th>Signal</th>
        <th>Reason</th><th>Score</th><th>Status / Review</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


# ---- leave + closure section builders ------------------------------------

def build_leave_section(session):
    """Renders the student leave list + registration form."""
    from html import escape
    students = session.query(Student).order_by(Student.name).all()
    leaves = (
        session.query(StudentLeave, Student)
        .join(Student, StudentLeave.student_id == Student.student_id)
        .order_by(StudentLeave.start_date.desc())
        .all()
    )

    student_options = "".join(
        f'<option value="{s.student_id}">{escape(s.name)} ({escape(s.roll_number)})</option>'
        for s in students
    )

    rows = []
    for leave, student in leaves:
        active = leave.start_date <= date.today() <= leave.end_date
        badge = '<span class="status-pill" style="background:#1F7A4D">active</span>' if active else ""
        rows.append(f"""
        <tr>
          <td>{escape(student.name)}</td>
          <td class="muted">{escape(student.roll_number)}</td>
          <td class="muted">{leave.start_date}</td>
          <td class="muted">{leave.end_date}</td>
          <td>{escape(leave.reason or "—")}</td>
          <td>{escape(leave.approved_by or "—")}</td>
          <td>{badge}</td>
          <td>
            <form method="POST" action="/leaves/{leave.id}/delete"
                  onsubmit="return confirm('Delete this leave record?')">
              <button type="submit" class="btn-danger">Delete</button>
            </form>
          </td>
        </tr>""")

    table = f"""
    <table>
      <thead><tr>
        <th>Student</th><th>Roll No.</th><th>From</th><th>To</th>
        <th>Reason</th><th>Approved by</th><th></th><th></th>
      </tr></thead>
      <tbody>{''.join(rows) if rows else '<tr><td colspan="8" class="empty">No leave records.</td></tr>'}</tbody>
    </table>""" if True else ""

    form = f"""
    <div class="sub-form">
      <h3>Register Student Leave</h3>
      <form method="POST" action="/leaves/add" class="inline-form">
        <label>Student
          <select name="student_id" required>{student_options}</select>
        </label>
        <label>From <input type="date" name="start_date" required></label>
        <label>To   <input type="date" name="end_date" required></label>
        <label>Reason <input type="text" name="reason" placeholder="home visit / medical…" maxlength="200"></label>
        <label>Approved by <input type="text" name="approved_by" placeholder="Warden name" maxlength="120"></label>
        <button type="submit">Add Leave</button>
      </form>
    </div>"""

    return table + form


def build_closure_section(session):
    """Renders the hostel closure list + registration form."""
    from html import escape
    closures = session.query(HostelClosure).order_by(HostelClosure.start_date.desc()).all()

    rows = []
    for c in closures:
        active = c.start_date <= date.today() <= c.end_date
        badge = '<span class="status-pill" style="background:#1F7A4D">active</span>' if active else ""
        rows.append(f"""
        <tr>
          <td class="muted">{c.start_date}</td>
          <td class="muted">{c.end_date}</td>
          <td>{escape(c.reason or "—")}</td>
          <td>{escape(c.registered_by or "—")}</td>
          <td>{badge}</td>
          <td>
            <form method="POST" action="/closures/{c.id}/delete"
                  onsubmit="return confirm('Delete this closure record?')">
              <button type="submit" class="btn-danger">Delete</button>
            </form>
          </td>
        </tr>""")

    table = f"""
    <table>
      <thead><tr>
        <th>From</th><th>To</th><th>Reason</th><th>Registered by</th><th></th><th></th>
      </tr></thead>
      <tbody>{''.join(rows) if rows else '<tr><td colspan="6" class="empty">No closure records.</td></tr>'}</tbody>
    </table>"""

    form = """
    <div class="sub-form">
      <h3>Register Hostel Closure</h3>
      <form method="POST" action="/closures/add" class="inline-form">
        <label>From <input type="date" name="start_date" required></label>
        <label>To   <input type="date" name="end_date" required></label>
        <label>Reason <input type="text" name="reason" placeholder="Diwali holidays…" maxlength="200"></label>
        <label>Registered by <input type="text" name="registered_by" placeholder="Warden name" maxlength="120"></label>
        <button type="submit">Add Closure</button>
      </form>
    </div>"""

    return table + form


# ---- page builder --------------------------------------------------------

def build_full_page(session):
    """Assembles the complete dashboard HTML, injecting leave/closure sections."""
    students_html   = build_students_section_editable(session)
    detections_html = build_detections_section(session)
    flags_html      = build_flags_section_editable(session)
    student_data    = build_student_detail_data(session)
    pending_count   = session.query(Flag).filter_by(status="pending").count()
    leave_html      = build_leave_section(session)
    closure_html    = build_closure_section(session)

    base_html = build_html(students_html, detections_html, flags_html, pending_count, student_data)

    # Inject leave + closure sections before </main>, and add extra CSS
    extra_css = """
    <style>
      .sub-form { padding: 16px 20px 20px; border-top: 1px solid var(--border); }
      .sub-form h3 { margin: 0 0 12px; font-size: 0.85rem; color: var(--accent);
                     text-transform: uppercase; letter-spacing: 0.04em; }
      .inline-form { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end; }
      .inline-form label { display: flex; flex-direction: column;
                           font-size: 0.78rem; color: var(--muted); gap: 3px; }
      .inline-form input, .inline-form select {
        padding: 5px 8px; border: 1px solid var(--border);
        border-radius: 4px; font-size: 0.85rem; }
      .inline-form button {
        background: var(--accent); color: white; border: none;
        padding: 6px 14px; border-radius: 4px; font-size: 0.82rem; cursor: pointer; }
      .inline-form button:hover { opacity: 0.85; }
      .btn-danger {
        background: #DC2626; color: white; border: none;
        padding: 3px 10px; border-radius: 4px; font-size: 0.75rem; cursor: pointer; }
      .btn-danger:hover { opacity: 0.85; }
      .btn {
        background: var(--accent); color: white; border: none;
        padding: 3px 10px; border-radius: 4px; font-size: 0.75rem; cursor: pointer; }
      .btn:hover { opacity: 0.85; }
    </style>"""

    leave_section = f"""
  <section>
    <div class="section-head"><h2>Student Leaves</h2></div>
    {leave_html}
  </section>"""

    closure_section = f"""
  <section>
    <div class="section-head"><h2>Hostel Closures</h2></div>
    {closure_html}
  </section>"""

    # insert CSS into <head> and sections before </main>
    base_html = base_html.replace("</head>", extra_css + "\n</head>", 1)
    base_html = base_html.replace("</main>", leave_section + closure_section + "\n</main>", 1)

    # add an "Enroll New Student" link into the header
    base_html = base_html.replace(
        "<h1>Wellness Monitoring — Dashboard</h1>",
        '<h1>Wellness Monitoring — Dashboard</h1>'
        '<a href="/bulk" style="float:right; color:#3E5C76; font-size:0.85rem; '
        'text-decoration:none; font-family:\'IBM Plex Mono\',monospace; margin-right:20px;">Bulk Actions</a>'
        '<a href="/enroll" style="float:right; color:#3E5C76; font-size:0.85rem; '
        'text-decoration:none; font-family:\'IBM Plex Mono\',monospace;">+ Enroll New Student</a>',
        1,
    )
    return base_html


# ---- routes --------------------------------------------------------------

@app.route("/")
def index():
    session = get_session()
    try:
        return build_full_page(session)
    finally:
        session.close()


@app.route("/flags/<int:flag_id>/update", methods=["POST"])
def update_flag(flag_id):
    new_status = request.form.get("status")
    reviewed_by = request.form.get("reviewed_by", "").strip()
    review_notes = request.form.get("review_notes", "").strip() or None

    if new_status not in VALID_STATUSES:
        return "Invalid status", 400
    if not reviewed_by:
        return "Reviewer name is required.", 400

    session = get_session()
    try:
        flag = session.query(Flag).filter_by(id=flag_id).first()
        if flag:
            flag.status = new_status
            flag.reviewed_by = reviewed_by
            flag.review_notes = review_notes
            flag.reviewed_at = datetime.now(timezone.utc)
            session.commit()
    finally:
        session.close()
    return redirect(url_for("index"))


# ---- leave routes --------------------------------------------------------

@app.route("/leaves/add", methods=["POST"])
def add_leave():
    try:
        student_id  = int(request.form["student_id"])
        start_date  = date.fromisoformat(request.form["start_date"])
        end_date    = date.fromisoformat(request.form["end_date"])
        reason      = request.form.get("reason", "").strip() or None
        approved_by = request.form.get("approved_by", "").strip() or None
    except (KeyError, ValueError) as e:
        return f"Invalid form data: {e}", 400

    if end_date < start_date:
        return "End date must be on or after start date.", 400

    session = get_session()
    try:
        leave = StudentLeave(
            student_id=student_id,
            start_date=start_date,
            end_date=end_date,
            reason=reason,
            approved_by=approved_by,
        )
        session.add(leave)
        session.commit()
    finally:
        session.close()
    return redirect(url_for("index"))


@app.route("/leaves/<int:leave_id>/delete", methods=["POST"])
def delete_leave(leave_id):
    session = get_session()
    try:
        leave = session.query(StudentLeave).filter_by(id=leave_id).first()
        if leave:
            session.delete(leave)
            session.commit()
    finally:
        session.close()
    return redirect(url_for("index"))


# ---- closure routes ------------------------------------------------------

@app.route("/closures/add", methods=["POST"])
def add_closure():
    try:
        start_date    = date.fromisoformat(request.form["start_date"])
        end_date      = date.fromisoformat(request.form["end_date"])
        reason        = request.form.get("reason", "").strip() or None
        registered_by = request.form.get("registered_by", "").strip() or None
    except (KeyError, ValueError) as e:
        return f"Invalid form data: {e}", 400

    if end_date < start_date:
        return "End date must be on or after start date.", 400

    session = get_session()
    try:
        closure = HostelClosure(
            start_date=start_date,
            end_date=end_date,
            reason=reason,
            registered_by=registered_by,
        )
        session.add(closure)
        session.commit()
    finally:
        session.close()
    return redirect(url_for("index"))


@app.route("/closures/<int:closure_id>/delete", methods=["POST"])
def delete_closure(closure_id):
    session = get_session()
    try:
        closure = session.query(HostelClosure).filter_by(id=closure_id).first()
        if closure:
            session.delete(closure)
            session.commit()
    finally:
        session.close()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)