import os
import logging
import json
import re
import hmac
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from supabase import create_client, Client

from transcript_display import build_display

try:
    from retell import Retell
except Exception:  # SDK absent -> verification fails closed when a secret is configured
    Retell = None
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = FastAPI(title="Retell Voice Agent Webhook Server")

# Rate limiting. Railway fronts the app with a proxy, so the real caller is the
# first hop of X-Forwarded-For; fall back to the socket peer.
def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

limiter = Limiter(key_func=_client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Only allow specific origins. Empty entries dropped; Vercel previews matched by regex.
_frontend = os.environ.get("FRONTEND_URL", "").strip()
ALLOWED_ORIGINS = [o for o in ["http://localhost:5173", _frontend] if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
 
ZAPIER_WEBHOOK_URL = os.environ.get("ZAPIER_WEBHOOK_URL")
# The Zapier feed is per-client plumbing that belongs to ONE tenant (McHugh —
# it drives their welcome-SMS/EFC automations). Calls from any other tenant's
# agent must NEVER be forwarded there, or their callers' data lands in
# McHugh's Zap history and downstream systems (2026-07-30 cross-tenant fix).
ZAPIER_CLIENT_ID = os.environ.get("ZAPIER_CLIENT_ID", "6d047c8a-bedf-4feb-9223-803c57a8ce1a")

# Leads rows trigger the external welcome-SMS automation (Supabase->Zapier),
# which is NOT client-scoped yet — a lead for any client would text the caller
# McHugh-branded SMS. Until that Zapier gets a client_id filter, only write
# leads for allowlisted clients (comma-separated env; default: McHugh).
LEADS_CLIENT_ALLOWLIST = set(
    s.strip()
    for s in os.environ.get(
        "LEADS_CLIENT_ALLOWLIST", "6d047c8a-bedf-4feb-9223-803c57a8ce1a"
    ).split(",")
    if s.strip()
)
RETELL_API_KEY = os.environ.get("RETELL_API_KEY")
# Webhook signing key. In this setup it's the same key as RETELL_API_KEY, so we fall
# back to RETELL_API_KEY when a dedicated secret isn't set.
RETELL_WEBHOOK_SECRET = os.environ.get("RETELL_WEBHOOK_SECRET")
# Shared secret required on /webhook/crm callers (set the same value in your CRM/Zapier).
CRM_WEBHOOK_SECRET = os.environ.get("CRM_WEBHOOK_SECRET")
# Reject oversized request bodies (DoS / storage abuse). Default 1 MB.
MAX_BODY_BYTES = int(os.environ.get("MAX_WEBHOOK_BODY_BYTES", "1000000"))
# Reject on failed/unverifiable signatures. Currently defaults to monitor mode
# (verify + log, never drop) until one real Retell call is confirmed verifying
# in Railway logs — then flip this default to "true" (fail closed).
# TODO(enforce): flip default to "true" after log confirmation.
WEBHOOK_VERIFY_ENFORCE = os.environ.get("WEBHOOK_VERIFY_ENFORCE", "false").lower() == "true"

# One Retell client for signature verification (verify() takes the key per-call, so the
# init key doesn't matter functionally). Built once; failure here never blocks traffic.
_retell_client = None
if Retell is not None:
    try:
        _retell_client = Retell(api_key=(RETELL_WEBHOOK_SECRET or RETELL_API_KEY or "unused"))
    except Exception as e:
        logging.getLogger(__name__).error("could not init Retell verify client: %s", e)
 
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
 
supabase_client = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    logger.info("Supabase client initialized")
else:
    logger.warning("Supabase credentials not set")
 
# Agent → client routing lives in the clients table (clients.agent_id), so
# onboarding a client is a data change, not a deploy. This env var (JSON object
# of {"agent_...": "client-uuid"}) is an emergency override checked first.
AGENT_CLIENT_MAP = {}
try:
    AGENT_CLIENT_MAP = json.loads(os.environ.get("AGENT_CLIENT_MAP", "{}"))
except Exception as e:
    logger.error("AGENT_CLIENT_MAP env is not valid JSON, ignoring: %s", e)

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
LOCAL_TIMEZONE = ZoneInfo("America/New_York")
 
 
# agent_id → (client_id | None, expiry). Hits cache for 5 min; unknown agents
# cache for 1 min so a typo'd agent_id can't hammer the DB on every event.
_agent_cache = {}
_AGENT_CACHE_TTL = 300
_AGENT_CACHE_NEG_TTL = 60


def get_client_id(agent_id):
    if not agent_id:
        return None
    if agent_id in AGENT_CLIENT_MAP:
        return AGENT_CLIENT_MAP[agent_id]

    cached = _agent_cache.get(agent_id)
    if cached and cached[1] > time.monotonic():
        return cached[0]

    if not supabase_client:
        return cached[0] if cached else None
    try:
        result = (
            supabase_client.table("clients")
            .select("id")
            .eq("agent_id", agent_id)
            .limit(1)
            .execute()
        )
        client_id = result.data[0]["id"] if result.data else None
        ttl = _AGENT_CACHE_TTL if client_id else _AGENT_CACHE_NEG_TTL
        _agent_cache[agent_id] = (client_id, time.monotonic() + ttl)
        return client_id
    except Exception as e:
        # DB hiccup: serve the last known mapping (even if expired) over dropping the call.
        logger.error("clients.agent_id lookup failed for %s: %s", agent_id, e)
        return cached[0] if cached else None


def mask_phone(raw):
    """Mask a phone number for logs: keep only the last 4 digits."""
    d = re.sub(r"\D", "", raw or "")
    if not d:
        return "withheld"
    return f"***{d[-4:]}" if len(d) >= 4 else "***"


def body_too_large(request: Request) -> bool:
    cl = request.headers.get("content-length")
    return cl is not None and cl.isdigit() and int(cl) > MAX_BODY_BYTES


def verify_retell_signature(raw_body: str, signature: str) -> bool:
    """Return True only when the webhook verifiably came from Retell. When
    verification CANNOT run (no key, SDK/client unavailable, unexpected error),
    the result follows the mode: enforce mode fails CLOSED (a config/SDK problem
    must not silently disable authentication); monitor mode fails open with a
    loud log. The caller drops False requests only when WEBHOOK_VERIFY_ENFORCE."""
    verify_key = RETELL_WEBHOOK_SECRET or RETELL_API_KEY
    if not verify_key:
        logger.error("No webhook verify key set (INSECURE)%s",
                     "" if WEBHOOK_VERIFY_ENFORCE else " - accepting unverified")
        return not WEBHOOK_VERIFY_ENFORCE
    if _retell_client is None:
        logger.error("Retell verify client unavailable%s",
                     "" if WEBHOOK_VERIFY_ENFORCE else " - accepting unverified")
        return not WEBHOOK_VERIFY_ENFORCE
    if not signature:
        logger.warning("Webhook has no X-Retell-Signature header")
        return False
    try:
        return bool(_retell_client.verify(raw_body, api_key=verify_key, signature=signature))
    except Exception as e:
        logger.error("verification errored%s: %s",
                     "" if WEBHOOK_VERIFY_ENFORCE else " - accepting unverified", e)
        return not WEBHOOK_VERIFY_ENFORCE


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


def parse_retell_start(call):
    start_timestamp = call.get("start_timestamp")
    if not start_timestamp:
        return None
    try:
        return datetime.fromtimestamp(int(start_timestamp) / 1000, tz=LOCAL_TIMEZONE)
    except (TypeError, ValueError):
        return None
 
 
def extract_call_data(body):
    call = body.get("call", {})
    analysis = call.get("call_analysis", {})
    custom = analysis.get("custom_analysis_data", {})
    started_at = parse_retell_start(call)
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
        "duration_ms": call.get("duration_ms") or call.get("call_duration_ms"),
        "call_date": started_at.strftime("%Y-%m-%d") if started_at else clean_text(custom.get("call_date")),
        "call_time": started_at.strftime("%H:%M:%S") if started_at else "",
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
        "transcript_display": build_display(call),
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
 
    call_record_base = {
        "client_id": client_id,
        "call_date": data.get("call_date") or datetime.now(LOCAL_TIMEZONE).strftime("%Y-%m-%d"),
        "call_time": data.get("call_time") or None,
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

    call_record_extra = {
        "transcript_display": data.get("transcript_display"),
        "final_outcome": data.get("final_outcome", ""),
        "trial_booked": data.get("trial_booked", False),
        "trial_cancelled": data.get("trial_cancelled", False),
        "needs_follow_up": data.get("needs_follow_up", False),
        "follow_up_reason": data.get("follow_up_reason", "N/A"),
        "bookings": data.get("bookings", []),
    }
    call_record = {**call_record_base, **call_record_extra}
 
    try:
        try:
            call_result = supabase_client.table("calls").insert(call_record).execute()
        except Exception as e:
            error_text = str(e)
            if "final_outcome" not in error_text and "schema cache" not in error_text and "PGRST204" not in error_text:
                raise
            logger.warning("Supabase schema missing new call fields; retrying without outcome metadata")
            call_record = call_record_base
            call_result = supabase_client.table("calls").insert(call_record).execute()
        inserted_call_id = call_result.data[0]["id"] if call_result.data else None
        logger.info("Call written to Supabase: %s", inserted_call_id)
 
        if is_lead and inserted_call_id and client_id not in LEADS_CLIENT_ALLOWLIST:
            logger.info(
                "Skipping lead write for client %s (not in LEADS_CLIENT_ALLOWLIST; welcome-SMS Zapier is unscoped)",
                client_id,
            )
        elif is_lead and inserted_call_id:
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
@limiter.limit("120/minute")
async def retell_webhook(request: Request):
    if body_too_large(request):
        return Response(status_code=413)
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        return Response(status_code=413)
    raw_body = raw.decode("utf-8", "replace")

    # Verify the request genuinely came from Retell (raw body — re-serializing JSON
    # would change bytes and break the signature).
    sig_ok = verify_retell_signature(raw_body, request.headers.get("X-Retell-Signature", ""))
    if not sig_ok and WEBHOOK_VERIFY_ENFORCE:
        logger.warning("Webhook rejected: invalid signature (enforce on)")
        return Response(status_code=401)
    if not sig_ok:
        logger.warning("Webhook signature not verified; accepting (monitor mode - set WEBHOOK_VERIFY_ENFORCE=true to reject)")

    try:
        body = json.loads(raw_body)
    except Exception:
        return Response(status_code=400)
    event = body.get("event", "")
 
    if event != "call_analyzed":
        return {"status": "ignored", "event": event}
 
    data = extract_call_data(body)
    logger.info("Processing call from %s agent %s", mask_phone(data["from_number"]), data["agent_id"])
 
    client_id = get_client_id(data["agent_id"])

    if client_id == ZAPIER_CLIENT_ID:
        zapier_payload = {
            "caller_name": data["caller_name"],
            "caller_phone": data["caller_phone"] or data["from_number"],
            "trial_day": data["trial_day"],
            "trial_time": data["trial_time"],
            "program": data["program"],
            "call_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
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
    else:
        logger.info("Zapier forward skipped for client %s (feed owner only)", client_id)

    if client_id:
        write_to_supabase(data, client_id)
    else:
        logger.warning("No client_id mapping for agent %s", data["agent_id"])
 
    return {"status": "ok", "event": event}
 
 
@app.post("/webhook/crm")
@limiter.limit("30/minute")
async def crm_webhook(request: Request):
    if not supabase_client:
        return Response(status_code=503)

    # Require a shared secret so external callers can't forge CRM events (which can
    # flip leads to "Converted"). Fails open with a warning only while unconfigured.
    if CRM_WEBHOOK_SECRET:
        provided = request.headers.get("X-Webhook-Secret", "")
        if not hmac.compare_digest(provided, CRM_WEBHOOK_SECRET):
            logger.warning("CRM webhook rejected: bad/missing X-Webhook-Secret")
            return Response(status_code=401)
    else:
        logger.warning("CRM_WEBHOOK_SECRET not set - /webhook/crm is UNAUTHENTICATED")

    if body_too_large(request):
        return Response(status_code=413)
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        return Response(status_code=413)
    try:
        body = json.loads(raw)
    except Exception:
        return Response(status_code=400)
    logger.info("CRM webhook received: %s", body.get("event_type", "unknown"))
 
    client_id = body.get("client_id")
    event_type = body.get("event_type", "")
    contact_name = body.get("contact_name", "")
    contact_phone = body.get("contact_phone", "")
    event_data = body.get("event_data", {})
 
    if not client_id:
        return Response(status_code=400)
 
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
                logger.info("Lead auto-converted for %s", mask_phone(contact_phone))
                supabase_client.table("calls").update({"lead_converted": True}).eq("client_id", client_id).eq("caller_phone", contact_phone).eq("is_lead", True).execute()
 
    except Exception as e:
        # Log details server-side; never leak exception text to the caller.
        logger.error("CRM webhook processing failed: %s", e)
        return Response(status_code=500)
 
    return {"status": "ok", "event_type": event_type}
