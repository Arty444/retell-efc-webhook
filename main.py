import os
import logging
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
 
 
def get_client_id(agent_id):
    return AGENT_CLIENT_MAP.get(agent_id)
 
 
def extract_call_data(body):
    call = body.get("call", {})
    analysis = call.get("call_analysis", {})
    custom = analysis.get("custom_analysis_data", {})
    return {
        "agent_id": call.get("agent_id", ""),
        "call_id": call.get("call_id", ""),
        "from_number": call.get("from_number", ""),
        "to_number": call.get("to_number", ""),
        "direction": call.get("direction", ""),
        "duration_ms": call.get("call_duration_ms"),
        "caller_name": custom.get("caller_name", ""),
        "caller_phone": custom.get("caller_phone", ""),
        "program": custom.get("program", ""),
        "trial_day": custom.get("trial_day", ""),
        "trial_time": custom.get("trial_time", ""),
        "call_type": custom.get("Call_Type", ""),
        "call_successful": custom.get("Call_Successful", False),
        "is_spam": str(custom.get("Spam", "No")).lower() == "yes",
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
 
    is_lead = bool(data.get("trial_day"))
 
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
            lead_record = {
                "client_id": client_id,
                "call_id": inserted_call_id,
                "caller_name": call_record["caller_name"],
                "caller_phone": call_record["caller_phone"],
                "program": call_record["program"],
                "trial_day": call_record["trial_day"],
                "trial_time": call_record["trial_time"],
                "status": "Booked",
            }
            supabase_client.table("leads").insert(lead_record).execute()
            logger.info("Lead created for call %s", inserted_call_id)
 
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
