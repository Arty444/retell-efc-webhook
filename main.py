import os
import logging
import json
import re
from datetime import datetime
 
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = FastAPI(title="Retell Voice Agent Webhook Server")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://*.vercel.app",
        os.environ.get("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
ZAPIER_WEBHOOK_URL = os.environ.get("ZAPIER_WEBHOOK_URL")
RETELL_API_KEY = os.environ.get("RETELL_API_KEY")
 
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
 
supabase_client = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    logger.info("Supabase client initialized")
else:
    logger.warning("Supabase credentials not set")
 
AGENT_CLIENT_MAP = {
    "agent_27efcd8d33e3d52313d74a74a2": "6d047c8a-bedf-4feb-9223-803c57a8ce1a",
}

BOOKED_OUTCOMES = {"booked", "rescheduled"}
EMPTY_VALUES = {"", "N/A", "NA", "NONE", "NULL"}
VALID_PROGRAMS = {
    "Adult Jiu Jitsu",
    "Tiny Sharks",
    "Little Sharks",
    "Junior Sharks",
    "Multiple",
    "N/A",
}
 
 
def get_client_id(agent_id):
    return AGENT_CLIENT_MAP.get(agent_id)


def clean_text(value, default=""):
    if value is None:
        return default
    return str(value).strip()


def is_empty_value(value):
    return clean_text(value).upper() in EMPTY_VALUES


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("true", "yes", "1")


def is_valid_date(value):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", clean_text(value)):
        return False
    try:
        datetime.strptime(clean_text(value), "%Y-%m-%d")
        return True
    except ValueError:
        return False


def normalize_final_outcome(value):
    outcome = clean_text(value).lower().replace(" ", "_")
    if outcome in {"book", "booked", "booking"}:
        return "booked"
    if outcome in {"reschedule", "rescheduled"}:
        return "rescheduled"
    if outcome in {"cancel", "canceled", "cancelled", "cancellation"}:
        return "cancelled"
    if outcome in {"info", "info_only", "general_question", "schedule"}:
        return "info_only"
    if outcome in {"hangup", "hang_up", "abandoned"}:
        return "abandoned"
    if outcome in {"spam", "spam_or_hangup"}:
        return "spam"
    if outcome == "message":
        return "message"
    return ""


def normalize_program(value):
    program = clean_text(value, "N/A")
    program_map = {
        "Adult": "Adult Jiu Jitsu",
        "Tiny Shark": "Tiny Sharks",
        "Little Shark": "Little Sharks",
        "Jr Shark": "Junior Sharks",
        "Jr Sharks": "Junior Sharks",
    }
    return program if program in VALID_PROGRAMS else program_map.get(program, program)


def parse_bookings(value):
    if is_empty_value(value):
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        logger.warning("Could not parse bookings JSON: %s", value)
        return []

    if not isinstance(parsed, list):
        return []

    bookings = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        status = clean_text(item.get("status"), "booked").lower()
        booking = {
            "student_name": clean_text(item.get("student_name"), "N/A"),
            "student_type": clean_text(item.get("student_type"), "N/A"),
            "program": normalize_program(item.get("program")),
            "trial_date": clean_text(item.get("trial_date")),
            "trial_time": clean_text(item.get("trial_time")),
            "status": status,
        }
        if status in BOOKED_OUTCOMES and is_valid_date(booking["trial_date"]):
            bookings.append(booking)
    return bookings


def build_legacy_booking(custom):
    trial_day = clean_text(custom.get("trial_day"))
    trial_time = clean_text(custom.get("trial_time"))
    if not is_valid_date(trial_day) or is_empty_value(trial_time):
        return []
    return [{
        "student_name": "N/A",
        "student_type": "N/A",
        "program": normalize_program(custom.get("program")),
        "trial_date": trial_day,
        "trial_time": trial_time,
        "status": "booked",
    }]
 
 
def extract_call_data(body):
    call = body.get("call", {})
    analysis = call.get("call_analysis", {})
    custom = analysis.get("custom_analysis_data", {})
    final_outcome = normalize_final_outcome(custom.get("final_outcome"))
    bookings = parse_bookings(custom.get("bookings"))
    if final_outcome in BOOKED_OUTCOMES and not bookings:
        bookings = build_legacy_booking(custom)

    first_booking = bookings[0] if bookings else {}
    trial_booked = (
        final_outcome in BOOKED_OUTCOMES
        and bool(bookings)
        and not parse_bool(custom.get("trial_cancelled"))
    )

    return {
        "agent_id": call.get("agent_id", ""),
        "call_id": call.get("call_id", ""),
        "from_number": call.get("from_number", ""),
        "to_number": call.get("to_number", ""),
        "direction": call.get("direction", ""),
        "duration_ms": call.get("call_duration_ms"),
        "caller_name": custom.get("caller_name", ""),
        "caller_phone": custom.get("caller_phone", ""),
        "program": first_booking.get("program") or normalize_program(custom.get("program")),
        "trial_day": first_booking.get("trial_date") or "N/A",
        "trial_time": first_booking.get("trial_time") or "N/A",
        "call_type": custom.get("Call_Type", ""),
        "call_successful": analysis.get("call_successful", custom.get("Call_Successful", False)),
        "is_spam": str(custom.get("Spam", "No")).lower() == "yes",
        "final_outcome": final_outcome,
        "trial_booked": trial_booked,
        "trial_cancelled": parse_bool(custom.get("trial_cancelled")),
        "needs_follow_up": parse_bool(custom.get("needs_follow_up")),
        "follow_up_reason": clean_text(custom.get("follow_up_reason"), "N/A"),
        "bookings": bookings,
        "sentiment": analysis.get("user_sentiment", ""),
        "summary": analysis.get("call_summary", ""),
        "transcript": call.get("transcript", ""),
    }
 
 
async def forward_to_zapier(payload):
    if not ZAPIER_WEBHOOK_URL:
        return
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(ZAPIER_WEBHOOK_URL, json=payload, timeout=10)
            logger.info("Zapier response: %s", resp.status_code)
    except Exception as e:
        logger.error("Zapier forward failed: %s", e)
 
 
def write_to_supabase(data, client_id):
    if not supabase_client:
        return
 
    duration_seconds = None
    if data.get("duration_ms"):
        duration_seconds = int(data["duration_ms"]) // 1000
 
    is_lead = data.get("trial_booked", False)
 
    call_record = {
        "client_id": client_id,
        "call_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "caller_phone": data.get("caller_phone") or data.get("from_number", ""),
        "caller_name": data.get("caller_name", ""),
        "program": data.get("program", ""),
        "trial_day": data.get("trial_day", ""),
        "trial_time": data.get("trial_time", ""),
        "call_type": data.get("call_type", ""),
        "call_successful": data.get("call_successful", False),
        "is_spam": data.get("is_spam", False),
        "sentiment": data.get("sentiment", ""),
        "duration_seconds": duration_seconds,
        "summary": data.get("summary", ""),
        "transcript": data.get("transcript", ""),
        "is_lead": is_lead,
        "lead_converted": False,
    }
 
    try:
        call_result = supabase_client.table("calls").insert(call_record).execute()
        inserted_call_id = call_result.data[0]["id"] if call_result.data else None
        logger.info("Call written to Supabase: %s", inserted_call_id)
 
        if is_lead and inserted_call_id:
            lead_records = []
            for booking in data.get("bookings", []):
                lead_records.append({
                    "client_id": client_id,
                    "call_id": inserted_call_id,
                    "caller_name": call_record["caller_name"],
                    "caller_phone": call_record["caller_phone"],
                    "program": booking.get("program") or call_record["program"],
                    "trial_day": booking.get("trial_date") or call_record["trial_day"],
                    "trial_time": booking.get("trial_time") or call_record["trial_time"],
                    "status": "Booked",
                })
            if lead_records:
                supabase_client.table("leads").insert(lead_records).execute()
                logger.info("Created %s lead(s) for call %s", len(lead_records), inserted_call_id)
 
    except Exception as e:
        logger.error("Supabase write failed: %s", e)
 
 
@app.get("/")
async def health():
    return {"status": "ok", "service": "voice-agent-webhook"}
 
 
@app.post("/webhook/retell")
async def retell_webhook(request: Request):
    body = await request.json()
    event = body.get("event", "")
 
    if event != "call_analyzed":
        return {"status": "ignored", "event": event}
 
    data = extract_call_data(body)
    logger.info("Processing call from %s agent %s", data["from_number"], data["agent_id"])
 
    zapier_payload = {
        "caller_name": data["caller_name"],
        "caller_phone": data["caller_phone"] or data["from_number"],
        "trial_day": data["trial_day"],
        "trial_time": data["trial_time"],
        "program": data["program"],
        "call_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "call_summary": data["summary"],
        "call_type": data["call_type"],
        "final_outcome": data["final_outcome"],
        "trial_booked": data["trial_booked"],
        "trial_cancelled": data["trial_cancelled"],
        "needs_follow_up": data["needs_follow_up"],
        "follow_up_reason": data["follow_up_reason"],
        "bookings": data["bookings"],
    }
    await forward_to_zapier(zapier_payload)
 
    client_id = get_client_id(data["agent_id"])
    if client_id:
        write_to_supabase(data, client_id)
    else:
        logger.warning("No client_id mapping for agent %s", data["agent_id"])
 
    return {"status": "ok", "event": event}
 
 
@app.post("/webhook/crm")
async def crm_webhook(request: Request):
    if not supabase_client:
        return {"status": "error", "message": "Supabase not configured"}
 
    body = await request.json()
    logger.info("CRM webhook received: %s", body.get("event_type", "unknown"))
 
    client_id = body.get("client_id")
    event_type = body.get("event_type", "")
    contact_name = body.get("contact_name", "")
    contact_phone = body.get("contact_phone", "")
    event_data = body.get("event_data", {})
 
    if not client_id:
        return {"status": "error", "message": "client_id required"}
 
    try:
        supabase_client.table("crm_events").insert({
            "client_id": client_id,
            "event_type": event_type,
            "contact_name": contact_name,
            "contact_phone": contact_phone,
            "event_data": event_data,
        }).execute()
        logger.info("CRM event stored: %s for %s", event_type, contact_name)
 
        if event_type in ("member_joined", "converted", "payment_received") and contact_phone:
            result = supabase_client.table("leads").update({"status": "Converted"}).eq("client_id", client_id).eq("caller_phone", contact_phone).eq("status", "Booked").execute()
 
            if result.data:
                logger.info("Lead auto-converted for %s", contact_phone)
                supabase_client.table("calls").update({"lead_converted": True}).eq("client_id", client_id).eq("caller_phone", contact_phone).eq("is_lead", True).execute()
 
    except Exception as e:
        logger.error("CRM webhook processing failed: %s", e)
        return {"status": "error", "message": str(e)}
 
    return {"status": "ok", "event_type": event_type}
