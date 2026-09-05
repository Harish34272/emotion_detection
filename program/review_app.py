import os
from datetime import date

import cv2
import numpy as np
from flask import Flask, redirect, url_for, request, send_from_directory, abort, jsonify
from db import get_session
from models import Flag, Student, FaceEmbedding, StudentLeave, HostelClosure
from face_engine import get_faces
from generate_dashboard import (
    build_students_section,
    build_detections_section,
    build_html,
    build_student_detail_data,
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
  {key: "left", label: "Left", hint: "Turn ~30-45° to your left"},
  {key: "right", label: "Right", hint: "Turn ~30-45° to your right"},
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
        rows.append(f"""
        <tr>
          <td class="muted" onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{created}</td>
          <td onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{escape(student.name)}</td>
          <td onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{escape(student.roll_number)}</td>
          <td onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{escape(flag.signal_type)}</td>
          <td onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{escape(flag.reason)}</td>
          <td onclick="showStudentDetail({student.student_id})" style="cursor:pointer">{flag.score:.2f}</td>
          <td>
            <form method="POST" action="/flags/{flag.id}/update" class="status-form">
              <select name="status">{options}</select>
              <button type="submit">Save</button>
            </form>
          </td>
        </tr>""")
    if not rows:
        return '<p class="empty">No flags raised yet. This is a good thing.</p>'
    return f"""
    <table>
      <thead><tr>
        <th>Raised</th><th>Student</th><th>Roll No.</th><th>Signal</th>
        <th>Reason</th><th>Score</th><th>Status</th>
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
    students_html   = build_students_section(session)
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
    if new_status not in VALID_STATUSES:
        return "Invalid status", 400
    session = get_session()
    try:
        flag = session.query(Flag).filter_by(id=flag_id).first()
        if flag:
            flag.status = new_status
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
    app.run(host="127.0.0.1", port=5000, debug=False)