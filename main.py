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

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    logger.info("Supabase client initialized")
else:
    logger.warning("Supabase credentials not set")

AGENT_CLIENT_MAP: dict[str, str] = {
    "agent_27efcd8d33e3d52313d74a74a2": "6d047c8a-bedf-4feb-9223-803c57a8ce1a",
}


def get_client_id(agent_id: str) -> str | None:
    return AGENT_CLIENT_MAP.get(agent_id)


def extract_call_data(body: dict) -> dict:
    call = body.get("call", {})
    analysis = call.get("call_analysis", {})
    custom = analysis.get("custom_analysis_data", {})
    return {
        "agent_id": call.get("agent_id", ""),
        "call
