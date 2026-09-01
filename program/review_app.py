
import os
from flask import Flask, redirect, url_for, request,send_from_directory, abort
from db import get_session
from models import Flag, Student
from generate_dashboard import (
    build_students_section,
    build_detections_section,
    build_html,
    build_student_detail_data,
)
import json

app = Flask(__name__)

VALID_STATUSES = {"pending", "reviewed", "dismissed", "handled"}




CROPS_ROOT = os.path.abspath("source_photos/detections")

@app.route("/crops/<roll_number>/<filename>")
def serve_crop(roll_number, filename):
    student_dir = os.path.join(CROPS_ROOT, roll_number)
    # send_from_directory already guards against path traversal (../),
    # but double-check the resolved dir is really inside CROPS_ROOT
    if not os.path.abspath(student_dir).startswith(CROPS_ROOT):
        abort(403)
    return send_from_directory(student_dir, filename)
@app.route("/student/<int:student_id>/photos")
def student_photos(student_id):
    from flask import jsonify
    session = get_session()
    student = session.query(Student).filter_by(student_id=student_id).first()
    session.close()
    if not student:
        return jsonify([])

    student_dir = os.path.join(CROPS_ROOT, student.roll_number)
    if not os.path.isdir(student_dir):
        return jsonify([])

    # only image files, sorted newest-first by modified time
    files = [f for f in os.listdir(student_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(student_dir, f)), reverse=True)
    recent = files[:8]

    return jsonify([
        {"filename": f, "url": f"/crops/{student.roll_number}/{f}"}
        for f in recent
    ])

def build_flags_section_editable(session):
    """Same as generate_dashboard.build_flags_section, but each row
    has a dropdown + submit button instead of a static status pill."""
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


@app.route("/")
def index():
    session = get_session()
    students_html = build_students_section(session)
    detections_html = build_detections_section(session)
    flags_html = build_flags_section_editable(session)
    student_data = build_student_detail_data(session)
    pending_count = session.query(Flag).filter_by(status="pending").count()
    html = build_html(students_html, detections_html, flags_html, pending_count, student_data)
    session.close()
    return html


@app.route("/flags/<int:flag_id>/update", methods=["POST"])
def update_flag(flag_id):
    new_status = request.form.get("status")
    if new_status not in VALID_STATUSES:
        return "Invalid status", 400

    session = get_session()
    flag = session.query(Flag).filter_by(id=flag_id).first()   # ← was flag_id=flag_id
    if flag:
        flag.status = new_status
        session.commit()
    session.close()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)