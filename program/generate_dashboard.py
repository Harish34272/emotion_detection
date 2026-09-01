
"""generate_dashboard.py"""
import os
from datetime import datetime, timezone
from html import escape

from db import get_session
from models import Student, FaceEmbedding, Camera, DetectionEvent, Flag, DailyActivitySummary

OUTPUT_FILE = "emotional_detection_dashboard.html"
RECENT_DETECTIONS_LIMIT = 30

STATUS_COLORS = {
    "pending": "#B45309",   # amber
    "reviewed": "#2563A8",  # blue
    "dismissed": "#6B7280", # gray
    "handled": "#1F7A4D",   # green
}


def build_students_section(session):
    students = session.query(Student).order_by(Student.name).all()
    rows = []
    for s in students:
        emb_count = session.query(FaceEmbedding).filter_by(student_id=s.student_id).count()
        det_count = session.query(DetectionEvent).filter_by(student_id=s.student_id).count()
        rows.append(f"""
        <tr onclick="showStudentDetail({s.student_id})" style="cursor:pointer" class="student-row">
          <td>{s.student_id}</td>
          <td>{escape(s.name)}</td>
          <td>{escape(s.roll_number)}</td>
          <td>{escape(s.department)}</td>
          <td>{s.year_of_study or '-'}</td>
          <td>{emb_count}</td>
          <td>{det_count}</td>
          <td class="muted">{s.enrolled_at.strftime('%Y-%m-%d') if s.enrolled_at else '-'}</td>
        </tr>""")
    if not rows:
        return '<p class="empty">No students enrolled yet.</p>'
    return f"""
    <table>
      <thead><tr>
        <th>ID</th><th>Name</th><th>Roll No.</th><th>Dept</th><th>Year</th>
        <th>Embeddings</th><th>Detections</th><th>Enrolled</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def build_detections_section(session):
    events = (
        session.query(DetectionEvent, Student, Camera)
        .join(Student, DetectionEvent.student_id == Student.student_id)
        .join(Camera, DetectionEvent.camera_id == Camera.camera_id)
        .order_by(DetectionEvent.timestamp.desc())
        .limit(RECENT_DETECTIONS_LIMIT)
        .all()
    )
    rows = []
    for event, student, camera in events:
        conf = f"{event.matched_confidence:.2f}" if event.matched_confidence is not None else "-"
        emotion = escape(event.emotion_label) if event.emotion_label else "-"
        ts = event.timestamp.strftime('%Y-%m-%d %H:%M:%S') if event.timestamp else "-"
        rows.append(f"""
        <tr>
          <td class="muted">{ts}</td>
          <td>{escape(student.name)}</td>
          <td>{escape(camera.location_name)}</td>
          <td>{conf}</td>
          <td>{emotion}</td>
        </tr>""")
    if not rows:
        return '<p class="empty">No detections logged yet.</p>'
    return f"""
    <table>
      <thead><tr>
        <th>Timestamp</th><th>Student</th><th>Camera</th><th>Match Conf.</th><th>Emotion</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""

import json

def build_student_detail_data(session):
    """
    Returns { student_id: {...} } with everything needed for the
    expandable detail panel — recent detections, daily meal summary,
    and flags — so the dashboard stays a single static HTML file.
    """
    students = session.query(Student).all()
    data = {}

    for s in students:
        # last 10 daily summaries, most recent first
        summaries = (
            session.query(DailyActivitySummary)
            .filter_by(student_id=s.student_id)
            .order_by(DailyActivitySummary.date.desc())
            .limit(10)
            .all()
        )
        summary_rows = [{
            "date": row.date.strftime("%Y-%m-%d"),
            "breakfast": row.breakfast_attended,
            "lunch": row.lunch_attended,
            "dinner": row.dinner_attended,
            "veranda_sightings": row.veranda_sightings,
        } for row in summaries]

        # last 15 raw detections
        recent = (
            session.query(DetectionEvent, Camera)
            .join(Camera, DetectionEvent.camera_id == Camera.camera_id)
            .filter(DetectionEvent.student_id == s.student_id)
            .order_by(DetectionEvent.timestamp.desc())
            .limit(15)
            .all()
        )
        recent_rows = [{
            "timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M"),
            "camera": camera.location_name,
            "emotion": event.emotion_label,
        } for event, camera in recent]

        # open flags for this student
        flags = (
            session.query(Flag)
            .filter_by(student_id=s.student_id)
            .order_by(Flag.created_at.desc())
            .limit(5)
            .all()
        )
        flag_rows = [{
            "created_at": f.created_at.strftime("%Y-%m-%d %H:%M") if f.created_at else "-",
            "reason": f.reason,
            "status": f.status,
            "score": f.score,
        } for f in flags]

        # quick summary flags for the UI: was today/most recent day fully missed?
        last_day = summary_rows[0] if summary_rows else None
        missed_today = None
        if last_day:
            missed_today = [m for m in ("breakfast", "lunch", "dinner") if not last_day[m]]

        data[s.student_id] = {
            "name": s.name,
            "roll_number": s.roll_number,
            "summaries": summary_rows,
            "recent_detections": recent_rows,
            "flags": flag_rows,
            "missed_today": missed_today,
        }

    return data
def build_flags_section(session):
    flags = (
        session.query(Flag, Student)
        .join(Student, Flag.student_id == Student.student_id)
        .order_by(Flag.created_at.desc())
        .all()
    )
    rows = []
    for flag, student in flags:
        color = STATUS_COLORS.get(flag.status, "#6B7280")
        created = flag.created_at.strftime('%Y-%m-%d %H:%M') if flag.created_at else "-"
        rows.append(f"""
        <tr onclick="showStudentDetail({student.student_id})" style="cursor:pointer" class="student-row">
          <td class="muted">{created}</td>
          <td>{escape(student.name)}</td>
          <td>{escape(student.roll_number)}</td>
          <td>{escape(flag.signal_type)}</td>
          <td>{escape(flag.reason)}</td>
          <td>{flag.score:.2f}</td>
          <td><span class="status-pill" style="background:{color}">{escape(flag.status)}</span></td>
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


def build_html(students_html, detections_html, flags_html, pending_count, student_data):
    generated_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wellness Monitoring — Dashboard</title>
<style>
  :root {{
    --bg: #F5F6F8;
    --panel: #FFFFFF;
    --ink: #1F2933;
    --muted: #6B7280;
    --border: #E2E5EA;
    --accent: #3E5C76;
    --accent-soft: #E9EEF3;
    --mono: 'IBM Plex Mono', 'SF Mono', Consolas, monospace;
    --sans: 'IBM Plex Sans', 'Inter', -apple-system, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: var(--sans);
    line-height: 1.5;
  }}
  header {{
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px;
  }}
  header h1 {{
    margin: 0 0 4px 0;
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: -0.01em;
  }}
  header .meta {{
    color: var(--muted);
    font-size: 0.85rem;
    font-family: var(--mono);
  }}
  .banner {{
    background: var(--accent-soft);
    border-bottom: 1px solid var(--border);
    padding: 10px 40px;
    font-size: 0.82rem;
    color: var(--accent);
  }}
  main {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 40px 80px;
  }}
  section {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 28px;
    overflow: hidden;
  }}
  section .section-head {{
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }}
  section h2 {{
    margin: 0;
    font-size: 0.95rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--accent);
  }}
  section .count {{
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--muted);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }}
  th {{
    text-align: left;
    padding: 10px 20px;
    background: var(--bg);
    color: var(--muted);
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 10px 20px;
    border-bottom: 1px solid var(--border);
  }}
  tr:last-child td {{ border-bottom: none; }}
  .muted {{ color: var(--muted); font-family: var(--mono); font-size: 0.82rem; }}
  .empty {{
    padding: 24px 20px;
    color: var(--muted);
    font-style: italic;
    margin: 0;
  }}
  .status-pill {{
    color: white;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  .status-form {{ display: flex; gap: 6px; align-items: center; }}
  .status-form select {{ padding: 4px 6px; border: 1px solid var(--border); border-radius: 4px; font-size: 0.8rem; }}
  .status-form button {{
    background: var(--accent); color: white; border: none;
    padding: 4px 10px; border-radius: 4px; font-size: 0.78rem; cursor: pointer;
  }}
  .status-form button:hover {{ opacity: 0.85; }}
  .detail-overlay {{
  display: none;
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.4);
  align-items: center; justify-content: center;
  z-index: 100;
}}
.detail-panel {{
  background: var(--panel);
  border-radius: 8px;
  max-width: 700px; width: 90%;
  max-height: 85vh; overflow-y: auto;
  padding: 28px 32px;
  position: relative;
}}
.detail-close {{
  position: absolute; top: 16px; right: 16px;
  border: none; background: none; font-size: 1.5rem;
  cursor: pointer; color: var(--muted);
}}
.detail-panel h3 {{ margin-top: 24px; color: var(--accent); font-size: 0.85rem; text-transform: uppercase; }}
.missed-banner {{ color: #B45309; font-weight: 600; }}
.ok-banner {{ color: #1F7A4D; font-weight: 600; }}
.student-row:hover {{ background: var(--accent-soft); }}
#photoStrip {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
.crop-thumb {{
  width: 80px; height: 80px; object-fit: cover;
  border-radius: 4px; border: 1px solid var(--border);
  cursor: pointer;
}}
.crop-thumb:hover {{ opacity: 0.8; }}
</style>
</head>
<body>
<header>
  <h1>Wellness Monitoring — Dashboard</h1>
  <div class="meta">Generated {generated_at} · re-run generate_dashboard.py to refresh</div>
</header>
<div class="banner">
  {pending_count} flag(s) pending review. This dashboard is read-only reference — resolve flags directly in the database or a future review tool.
</div>
<main>
  <section>
    <div class="section-head"><h2>Flags</h2></div>
    {flags_html}
  </section>
  <div id="detailOverlay" class="detail-overlay" onclick="closeStudentDetail(event)">
    <div class="detail-panel" onclick="event.stopPropagation()">
      <button class="detail-close" onclick="closeStudentDetail()">×</button>
      <div id="detailContent"></div>
    </div>
  </div>
  <section>
    <div class="section-head"><h2>Enrolled Students</h2></div>
    {students_html}
  </section>
  <section>
    <div class="section-head"><h2>Recent Detections</h2><span class="count">last {RECENT_DETECTIONS_LIMIT}</span></div>
    {detections_html}
  </section>
  
</main>
<script>
const STUDENT_DATA = {json.dumps(student_data)};
async function showStudentDetail(id) {{
  const d = STUDENT_DATA[id];
  if (!d) return;

  let missedHtml = '';
  if (d.missed_today && d.missed_today.length) {{
    missedHtml = `<p class="missed-banner">⚠ Missed today: ${{d.missed_today.join(', ')}}</p>`;
  }} else if (d.missed_today) {{
    missedHtml = `<p class="ok-banner">✓ All meals attended (most recent day)</p>`;
  }}

  const summaryRows = d.summaries.map(r => `
    <tr>
      <td class="muted">${{r.date}}</td>
      <td>${{r.breakfast ? '✓' : '✗'}}</td>
      <td>${{r.lunch ? '✓' : '✗'}}</td>
      <td>${{r.dinner ? '✓' : '✗'}}</td>
      <td>${{r.veranda_sightings}}</td>
    </tr>`).join('');

  const detectionRows = d.recent_detections.map(r => `
    <tr>
      <td class="muted">${{r.timestamp}}</td>
      <td>${{r.camera}}</td>
      <td>${{r.emotion || '-'}}</td>
    </tr>`).join('');

  const flagRows = d.flags.map(f => `
    <tr>
      <td class="muted">${{f.created_at}}</td>
      <td>${{f.reason}}</td>
      <td>${{f.status}}</td>
    </tr>`).join('');

  document.getElementById('detailContent').innerHTML = `
    <h2>${{d.name}} <span class="muted">(${{d.roll_number}})</span></h2>
    ${{missedHtml}}
    <h3>Recent Photos</h3>
    <div id="photoStrip"><p class="muted">Loading photos…</p></div>
    <h3>Last 10 Days — Meal Attendance</h3>
    <table><thead><tr><th>Date</th><th>B</th><th>L</th><th>D</th><th>Veranda</th></tr></thead>
    <tbody>${{summaryRows || '<tr><td colspan=5 class="empty">No data</td></tr>'}}</tbody></table>
    <h3>Recent Detections</h3>
    <table><thead><tr><th>Timestamp</th><th>Camera</th><th>Emotion</th></tr></thead>
    <tbody>${{detectionRows || '<tr><td colspan=3 class="empty">No detections</td></tr>'}}</tbody></table>
    <h3>Flags</h3>
    <table><thead><tr><th>Raised</th><th>Reason</th><th>Status</th></tr></thead>
    <tbody>${{flagRows || '<tr><td colspan=3 class="empty">No flags</td></tr>'}}</tbody></table>
  `;
  document.getElementById('detailOverlay').style.display = 'flex';

  try {{
    const res = await fetch(`/student/${{id}}/photos`);
    const photos = await res.json();
    const strip = document.getElementById('photoStrip');
    if (photos.length === 0) {{
      strip.innerHTML = '<p class="empty">No photos available</p>';
    }} else {{
      strip.innerHTML = photos.map(p =>
        `<img src="${{p.url}}" class="crop-thumb" onclick="window.open('${{p.url}}', '_blank')">`
      ).join('');
    }}
  }} catch (e) {{
    document.getElementById('photoStrip').innerHTML = '<p class="empty">Could not load photos</p>';
  }}
}}

function closeStudentDetail(e) {{
  document.getElementById('detailOverlay').style.display = 'none';
}}

</script>
</body>
</html>"""


def main():
    session = get_session()

    students_html = build_students_section(session)
    detections_html = build_detections_section(session)
    flags_html = build_flags_section(session)
    student_data = build_student_detail_data(session)
    pending_count = session.query(Flag).filter_by(status="pending").count()

    html = build_html(students_html, detections_html, flags_html, pending_count, student_data)
    # with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    #     f.write(html)


    session.close()
    print(f"Dashboard written to {os.path.abspath(OUTPUT_FILE)}")
    print(f"Open it in a browser, or run: python3 -m http.server 8080")

main()

