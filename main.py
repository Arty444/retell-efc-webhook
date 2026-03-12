import os
import logging
from datetime import datetime

import httpx
from fastapi import FastAPI, Request, HTTPException

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RETELL_API_KEY = os.environ.get("RETELL_API_KEY", "")
ZAPIER_WEBHOOK_URL = os.environ.get(
    "ZAPIER_WEBHOOK_URL",
    "https://hooks.zapier.com/hooks/catch/20502458/uxetn1u/",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("retell-efc")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Retell → EFC Aquila Webhook Bridge")


@app.get("/")
async def health():
    return {"status": "ok", "service": "retell-efc-webhook"}


@app.post("/webhook/retell")
async def retell_post_call_webhook(request: Request):
    """
    Receives the Retell AI post-call webhook, extracts lead data,
    and forwards it to Zapier which creates a lead in EFC Aquila.
    """
    body = await request.json()
    log.info("Received Retell webhook: event=%s", body.get("event"))

    # ------------------------------------------------------------------
    # 1. Only process "call_ended" events
    # ------------------------------------------------------------------
    event = body.get("event")
    if event != "call_ended":
        log.info("Ignoring non-call_ended event: %s", event)
        return {"status": "ignored", "reason": f"event type '{event}' not handled"}

    call_data = body.get("call", {})
    call_id = call_data.get("call_id", "unknown")
    log.info("Processing call_ended for call_id=%s", call_id)

    # ------------------------------------------------------------------
    # 2. Extract lead data from call analysis / custom data
    # ------------------------------------------------------------------
    # Retell puts structured data in call_analysis.custom_analysis_data
    # or in call.metadata depending on your agent configuration.
    call_analysis = call_data.get("call_analysis", {})
    custom_data = call_analysis.get("custom_analysis_data", {})

    # Also check call metadata and agent-level custom data
    metadata = call_data.get("metadata", {})

    # Try multiple locations where Retell may store extracted info
    first_name = (
        custom_data.get("first_name")
        or custom_data.get("firstName")
        or metadata.get("first_name")
        or ""
    )
    last_name = (
        custom_data.get("last_name")
        or custom_data.get("lastName")
        or metadata.get("last_name")
        or ""
    )
    phone = (
        custom_data.get("phone")
        or custom_data.get("phone_number")
        or metadata.get("phone")
        or call_data.get("from_number")
        or ""
    )
    trial_class_day = (
        custom_data.get("trial_class_day")
        or custom_data.get("trialClassDay")
        or metadata.get("trial_class_day")
        or ""
    )
    trial_class_time = (
        custom_data.get("trial_class_time")
        or custom_data.get("trialClassTime")
        or metadata.get("trial_class_time")
        or ""
    )

    # Check if a trial was actually booked
    trial_booked = (
        custom_data.get("trial_booked")
        or custom_data.get("trialBooked")
        or metadata.get("trial_booked")
        or False
    )

    # Accept truthy string values like "true", "yes", "1"
    if isinstance(trial_booked, str):
        trial_booked = trial_booked.lower() in ("true", "yes", "1")

    if not trial_booked:
        log.info(
            "Call %s ended but no trial class was booked. Skipping lead creation.",
            call_id,
        )
        return {
            "status": "skipped",
            "reason": "trial not booked",
            "call_id": call_id,
        }

    # Validate minimum required data
    if not first_name or not last_name:
        log.warning(
            "Call %s: trial booked but missing name (first=%r, last=%r). Skipping.",
            call_id,
            first_name,
            last_name,
        )
        return {
            "status": "skipped",
            "reason": "missing name data",
            "call_id": call_id,
        }

    # ------------------------------------------------------------------
    # 3. Build payload for Zapier → EFC Aquila "Add Lead"
    # ------------------------------------------------------------------
    lead_payload = {
        "first_name": first_name,
        "last_name": last_name,
        "phone": phone,
        "trial_class_day": trial_class_day,
        "trial_class_time": trial_class_time,
        "source": "Retell AI Phone Call",
        "call_id": call_id,
        "created_at": datetime.utcnow().isoformat(),
    }

    log.info("Sending lead to Zapier: %s %s, phone=%s", first_name, last_name, phone)

    # ------------------------------------------------------------------
    # 4. POST to Zapier webhook
    # ------------------------------------------------------------------
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(ZAPIER_WEBHOOK_URL, json=lead_payload)
            resp.raise_for_status()

        log.info(
            "SUCCESS: Lead sent to Zapier for %s %s (call %s). Zapier status=%s",
            first_name,
            last_name,
            call_id,
            resp.status_code,
        )
        return {
            "status": "success",
            "lead": f"{first_name} {last_name}",
            "call_id": call_id,
            "zapier_status": resp.status_code,
        }

    except httpx.HTTPStatusError as e:
        log.error(
            "FAILED: Zapier returned %s for call %s: %s",
            e.response.status_code,
            call_id,
            e.response.text,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Zapier returned {e.response.status_code}",
        )
    except httpx.RequestError as e:
        log.error("FAILED: Could not reach Zapier for call %s: %s", call_id, str(e))
        raise HTTPException(status_code=502, detail="Could not reach Zapier")
