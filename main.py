import os
import logging
from datetime import datetime
import httpx
from fastapi import FastAPI, Request

ZAPIER_WEBHOOK_URL = os.environ.get("ZAPIER_WEBHOOK_URL", "https://hooks.zapier.com/hooks/catch/20502458/uxcnw9q/")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("retell-efc")

app = FastAPI(title="Retell Webhook Bridge")

@app.get("/")
async def health():
    return {"status": "ok", "service": "retell-efc-webhook"}

@app.post("/webhook/retell")
async def retell_webhook(request: Request):
    body = await request.json()
    log.info("Received: %s", body)
    
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
    }
    
    log.info("Sending to Zapier: %s", payload)
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(ZAPIER_WEBHOOK_URL, json=payload)
        log.info("Zapier response: %s", resp.status_code)
    
    return {"status": "success", "payload": payload}
