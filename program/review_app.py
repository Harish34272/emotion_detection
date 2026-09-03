import os
from datetime import date
from flask import Flask, redirect, url_for, request, send_from_directory, abort, jsonify
from db import get_session
from models import Flag, Student, StudentLeave, HostelClosure
from generate_dashboard import (
    build_students_section,
    build_detections_section,
    build_html,
    build_student_detail_data,
)

app = Flask(__name__)

VALID_STATUSES = {"pending", "reviewed", "dismissed", "handled"}

CROPS_ROOT = os.path.abspath("source_photos/detections")


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