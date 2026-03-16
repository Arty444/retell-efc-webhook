import os
import logging
import sqlite3
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from contextlib import contextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

ZAPIER_WEBHOOK_URL = os.environ.get("ZAPIER_WEBHOOK_URL", "https://hooks.zapier.com/hooks/catch/20502458/uxcnw9q/")
DB_PATH = os.environ.get("DB_PATH", "leads.db")
MATCH_WINDOW_DAYS = int(os.environ.get("MATCH_WINDOW_DAYS", "30"))
NAME_MATCH_THRESHOLD = float(os.environ.get("NAME_MATCH_THRESHOLD", "0.7"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("retell-efc")

app = FastAPI(title="Retell → EFC Tracking")


# ── Database ────────────────────────────────────────────────────────────────

def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caller_name TEXT,
                caller_phone TEXT,
                trial_day TEXT,
                trial_time TEXT,
                program TEXT,
                call_date TEXT,
                call_summary TEXT,
                stage TEXT DEFAULT 'trial_booked',
                enrolled_date TEXT,
                membership_amount REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@app.on_event("startup")
async def startup():
    init_db()


# ── Helpers ─────────────────────────────────────────────────────────────────

def normalize_phone(phone: str) -> str:
    """Strip to digits only for consistent matching."""
    return "".join(c for c in phone if c.isdigit())


def name_similarity(a: str, b: str) -> float:
    """Fuzzy name match ratio (0-1)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def find_matching_lead(db, phone: str, name: str, program: str):
    """Find a lead by phone first, then fall back to name+program fuzzy match."""
    cutoff = (datetime.utcnow() - timedelta(days=MATCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
    norm_phone = normalize_phone(phone)

    # Try phone match first
    if norm_phone:
        rows = db.execute(
            "SELECT * FROM leads WHERE stage = 'trial_booked' AND call_date >= ? ORDER BY created_at DESC",
            (cutoff,)
        ).fetchall()
        for row in rows:
            if normalize_phone(row["caller_phone"]) == norm_phone:
                return row

    # Fall back to name + program match
    if name:
        rows = db.execute(
            "SELECT * FROM leads WHERE stage = 'trial_booked' AND call_date >= ? ORDER BY created_at DESC",
            (cutoff,)
        ).fetchall()
        for row in rows:
            sim = name_similarity(row["caller_name"], name)
            program_match = row["program"].lower().strip() == program.lower().strip() if (row["program"] and program) else False
            if sim >= NAME_MATCH_THRESHOLD and program_match:
                log.info("Fuzzy name match: '%s' ↔ '%s' (%.0f%%), program: %s", row["caller_name"], name, sim * 100, program)
                return row

    return None


# ── Retell Webhook (AI call → trial booked) ─────────────────────────────────

@app.post("/webhook/retell")
async def retell_webhook(request: Request):
    body = await request.json()
    log.info("Received Retell event: %s", body.get("event", ""))

    event = body.get("event", "")
    if event != "call_analyzed":
        return {"status": "ignored", "event": event}

    call = body.get("call", {})
    analysis = call.get("call_analysis", {})
    custom = analysis.get("custom_analysis_data", {})

    payload = {
        "caller_name": custom.get("caller_name", ""),
        "caller_phone": call.get("from_number", ""),
        "trial_day": custom.get("trial_day", ""),
        "trial_time": custom.get("trial_time", ""),
        "program": custom.get("program", ""),
        "call_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "call_summary": analysis.get("call_summary", ""),
    }

    # Store in database
    with get_db() as db:
        db.execute(
            """INSERT INTO leads (caller_name, caller_phone, trial_day, trial_time, program, call_date, call_summary, stage)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'trial_booked')""",
            (payload["caller_name"], payload["caller_phone"], payload["trial_day"],
             payload["trial_time"], payload["program"], payload["call_date"], payload["call_summary"])
        )

    log.info("Lead saved: %s", payload["caller_name"])

    # Still forward to Zapier
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(ZAPIER_WEBHOOK_URL, json=payload)
        log.info("Zapier response: %s", resp.status_code)

    return {"status": "success", "payload": payload}


# ── Aquila Webhook (enrollment from Zapier) ──────────────────────────────────

@app.post("/webhook/aquila")
async def aquila_webhook(request: Request):
    body = await request.json()
    log.info("Received Aquila enrollment: %s", body)

    name = body.get("member_name", "")
    phone = body.get("member_phone", "")
    program = body.get("program", "")
    amount = body.get("membership_amount", 0)

    with get_db() as db:
        lead = find_matching_lead(db, phone, name, program)

        if lead:
            db.execute(
                "UPDATE leads SET stage = 'enrolled', enrolled_date = ?, membership_amount = ? WHERE id = ?",
                (datetime.utcnow().strftime("%Y-%m-%d"), amount, lead["id"])
            )
            log.info("Lead #%d matched and enrolled: %s", lead["id"], name)
            return {"status": "matched", "lead_id": lead["id"], "name": lead["caller_name"]}
        else:
            log.info("No matching lead found for: %s / %s", name, phone)
            return {"status": "no_match", "name": name, "phone": phone}


# ── Dashboard ────────────────────────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    period = request.query_params.get("period", "all")

    with get_db() as db:
        where = ""
        params = ()
        if period == "7d":
            where = "WHERE call_date >= ?"
            params = ((datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),)
        elif period == "30d":
            where = "WHERE call_date >= ?"
            params = ((datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"),)
        elif period == "90d":
            where = "WHERE call_date >= ?"
            params = ((datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d"),)

        total_calls = db.execute(f"SELECT COUNT(*) FROM leads {where}", params).fetchone()[0]
        enrolled_where = f"{where} AND stage = 'enrolled'" if where else "WHERE stage = 'enrolled'"
        total_enrolled = db.execute(f"SELECT COUNT(*) FROM leads {enrolled_where}", params).fetchone()[0]
        total_revenue = db.execute(f"SELECT COALESCE(SUM(membership_amount), 0) FROM leads {enrolled_where}", params).fetchone()[0]
        pending = total_calls - total_enrolled

        conv_rate = (total_enrolled / total_calls * 100) if total_calls > 0 else 0

        # Recent leads
        recent = db.execute(
            "SELECT * FROM leads ORDER BY created_at DESC LIMIT 20"
        ).fetchall()

        # Per-program breakdown
        programs = db.execute(f"""
            SELECT program,
                   COUNT(*) as total,
                   SUM(CASE WHEN stage = 'enrolled' THEN 1 ELSE 0 END) as enrolled,
                   COALESCE(SUM(membership_amount), 0) as revenue
            FROM leads {where}
            GROUP BY program
            ORDER BY total DESC
        """, params).fetchall()

    leads_html = ""
    for r in recent:
        stage_color = "#22c55e" if r["stage"] == "enrolled" else "#eab308"
        stage_label = "Enrolled" if r["stage"] == "enrolled" else "Trial Booked"
        revenue_str = f"${r['membership_amount']:.0f}" if r["membership_amount"] else "-"
        leads_html += f"""
        <tr>
            <td>{r['caller_name'] or 'Unknown'}</td>
            <td>{r['caller_phone'] or '-'}</td>
            <td>{r['program'] or '-'}</td>
            <td>{r['trial_day'] or '-'} {r['trial_time'] or ''}</td>
            <td>{r['call_date'] or '-'}</td>
            <td><span style="color:{stage_color};font-weight:700">{stage_label}</span></td>
            <td>{revenue_str}</td>
        </tr>"""

    programs_html = ""
    for p in programs:
        p_rate = (p["enrolled"] / p["total"] * 100) if p["total"] > 0 else 0
        programs_html += f"""
        <tr>
            <td>{p['program'] or 'Unknown'}</td>
            <td>{p['total']}</td>
            <td>{p['enrolled']}</td>
            <td>{p_rate:.0f}%</td>
            <td>${p['revenue']:.0f}</td>
        </tr>"""

    active = lambda p: "font-weight:700;text-decoration:underline" if period == p else ""

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Agent ROI Dashboard</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,system-ui,sans-serif; background:#0f172a; color:#e2e8f0; padding:20px; }}
  h1 {{ font-size:1.5rem; margin-bottom:8px; }}
  .subtitle {{ color:#94a3b8; margin-bottom:20px; }}
  .filters {{ margin-bottom:20px; }}
  .filters a {{ color:#94a3b8; text-decoration:none; margin-right:16px; }}
  .filters a:hover {{ color:#e2e8f0; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; margin-bottom:32px; }}
  .card {{ background:#1e293b; border-radius:12px; padding:20px; }}
  .card .label {{ font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; }}
  .card .value {{ font-size:2rem; font-weight:700; margin-top:4px; }}
  .card .value.green {{ color:#22c55e; }}
  .card .value.yellow {{ color:#eab308; }}
  .card .value.blue {{ color:#3b82f6; }}
  table {{ width:100%; border-collapse:collapse; background:#1e293b; border-radius:12px; overflow:hidden; margin-bottom:32px; }}
  th {{ text-align:left; padding:12px 16px; background:#334155; font-size:0.75rem; text-transform:uppercase; color:#94a3b8; letter-spacing:0.05em; }}
  td {{ padding:12px 16px; border-top:1px solid #334155; font-size:0.9rem; }}
  h2 {{ font-size:1.1rem; margin-bottom:12px; }}
</style>
</head><body>
<h1>AI Call Agent — ROI Dashboard</h1>
<p class="subtitle">Tracking Retell AI calls through to enrollment</p>

<div class="filters">
  <a href="?period=7d" style="{active('7d')}">7 days</a>
  <a href="?period=30d" style="{active('30d')}">30 days</a>
  <a href="?period=90d" style="{active('90d')}">90 days</a>
  <a href="?period=all" style="{active('all')}">All time</a>
</div>

<div class="cards">
  <div class="card">
    <div class="label">Trials Booked</div>
    <div class="value">{total_calls}</div>
  </div>
  <div class="card">
    <div class="label">Enrolled</div>
    <div class="value green">{total_enrolled}</div>
  </div>
  <div class="card">
    <div class="label">Pending</div>
    <div class="value yellow">{pending}</div>
  </div>
  <div class="card">
    <div class="label">Conversion Rate</div>
    <div class="value blue">{conv_rate:.0f}%</div>
  </div>
  <div class="card">
    <div class="label">Revenue from AI Leads</div>
    <div class="value green">${total_revenue:,.0f}</div>
  </div>
</div>

<h2>By Program</h2>
<table>
  <tr><th>Program</th><th>Trials</th><th>Enrolled</th><th>Conv %</th><th>Revenue</th></tr>
  {programs_html}
</table>

<h2>Recent Leads</h2>
<table>
  <tr><th>Name</th><th>Phone</th><th>Program</th><th>Trial</th><th>Call Date</th><th>Stage</th><th>Revenue</th></tr>
  {leads_html}
</table>
</body></html>"""


@app.get("/")
async def health():
    return {"status": "ok", "service": "retell-efc-webhook"}
