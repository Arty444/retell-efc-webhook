import os
import base64
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from supabase import create_client, Client

from bird_analyzer import analyze_audio
from audio_direction import estimate_direction, estimate_direction_stereo, estimate_directions_multi_source
from sound_separation import separate_sources
from multi_device_locator import DeviceRoom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = FastAPI(title="Retell Voice Agent Webhook Server")

# Mount static files for Bird Sound Locator PWA
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
 
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


# ─── Bird Sound Locator ────────────────────────────────────────────────────────

# Multi-device rooms for TDOA triangulation
device_rooms: dict[str, DeviceRoom] = {}


@app.get("/app", response_class=HTMLResponse)
async def bird_locator_app():
    """Serve the Bird Sound Locator PWA."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(), status_code=200)


@app.post("/api/room/create")
async def create_room():
    """Create a new multi-device room for triangulation."""
    room = DeviceRoom()
    device_rooms[room.room_id] = room
    return {"room_id": room.room_id}


@app.get("/api/room/{room_id}")
async def get_room_info(room_id: str):
    """Get info about a multi-device room."""
    room = device_rooms.get(room_id)
    if not room:
        return {"error": "Room not found"}
    return {
        "room_id": room.room_id,
        "device_count": room.get_device_count(),
        "devices": room.get_device_positions(),
        "can_localize": room.can_localize(),
    }


@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bird sound analysis.

    Protocol:
    Client sends:
      {"type": "config", "latitude": float, "longitude": float, "enable_separation": bool}
      {"type": "audio_chunk", "data": "<base64 Int16 PCM>", "sample_rate": int, "heading": float}
      {"type": "audio_stereo", "ch1": "<base64>", "ch2": "<base64>", "sample_rate": int, "heading": float}
      {"type": "join_room", "room_id": str, "device_id": str}

    Server sends:
      {"type": "detection", "detections": [...], "direction": {...}}
      {"type": "multi_detection", "birds": [...]}  -- separated sources with individual directions
      {"type": "triangulation", "result": {...}}    -- multi-device TDOA result
      {"type": "status", "message": str}
    """
    await websocket.accept()
    logger.info("Bird locator WebSocket connected")

    # Per-connection state
    config = {"latitude": 0.0, "longitude": 0.0, "enable_separation": False}
    audio_buffer = []
    ANALYSIS_WINDOW_SECONDS = 3.0
    ANALYSIS_INTERVAL_CHUNKS = 4
    chunk_count = 0
    room_id = None
    device_id = None

    try:
        await websocket.send_json({"type": "status", "message": "Connected. Start listening to detect birds!"})

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "config":
                config["latitude"] = float(data.get("latitude", 0))
                config["longitude"] = float(data.get("longitude", 0))
                config["enable_separation"] = bool(data.get("enable_separation", False))
                mode = "multi-bird" if config["enable_separation"] else "single"
                await websocket.send_json({
                    "type": "status",
                    "message": f"Location set ({config['latitude']:.1f}, {config['longitude']:.1f}) | Mode: {mode}"
                })
                continue

            if msg_type == "join_room":
                room_id = data.get("room_id")
                device_id = data.get("device_id", f"dev-{id(websocket)}")
                if room_id and room_id not in device_rooms:
                    device_rooms[room_id] = DeviceRoom(room_id)
                if room_id:
                    room = device_rooms[room_id]
                    room.add_device(device_id, config["latitude"], config["longitude"])
                    await websocket.send_json({
                        "type": "room_joined",
                        "room_id": room_id,
                        "device_id": device_id,
                        "device_count": room.get_device_count(),
                        "devices": room.get_device_positions(),
                    })
                continue

            if msg_type == "audio_stereo":
                # Stereo mic mode: two channels for ITD-based direction
                try:
                    ch1_bytes = base64.b64decode(data["ch1"])
                    ch2_bytes = base64.b64decode(data["ch2"])
                    ch1 = np.frombuffer(ch1_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    ch2 = np.frombuffer(ch2_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    sr = int(data.get("sample_rate", 44100))
                    heading = float(data.get("heading", 0))
                except Exception as e:
                    logger.warning("Failed to decode stereo audio: %s", e)
                    continue

                stereo_dir = estimate_direction_stereo(ch1, ch2, sr, heading)
                if stereo_dir:
                    await websocket.send_json({
                        "type": "stereo_direction",
                        "direction": stereo_dir,
                    })
                continue

            if msg_type == "audio_chunk":
                # Decode audio
                try:
                    raw_bytes = base64.b64decode(data["data"])
                    int16_data = np.frombuffer(raw_bytes, dtype=np.int16)
                    float32_data = int16_data.astype(np.float32) / 32768.0
                    sample_rate = int(data.get("sample_rate", 44100))
                    heading = float(data.get("heading", 0))
                except Exception as e:
                    logger.warning("Failed to decode audio chunk: %s", e)
                    continue

                audio_buffer.append({
                    "pcm": float32_data,
                    "heading": heading,
                    "timestamp": datetime.utcnow().timestamp(),
                })

                # Submit to multi-device room if joined
                if room_id and room_id in device_rooms and device_id:
                    room = device_rooms[room_id]
                    room.submit_audio(device_id, float32_data, datetime.utcnow().timestamp(), sample_rate)

                    if room.can_localize():
                        tri_result = room.localize()
                        if tri_result:
                            await websocket.send_json({
                                "type": "triangulation",
                                "result": {
                                    "bearing": tri_result.bearing,
                                    "distance": tri_result.distance,
                                    "latitude": tri_result.latitude,
                                    "longitude": tri_result.longitude,
                                    "confidence": tri_result.confidence,
                                    "x": tri_result.x,
                                    "y": tri_result.y,
                                },
                                "devices": room.get_device_positions(),
                            })

                # Trim buffer
                now = datetime.utcnow().timestamp()
                audio_buffer = [
                    c for c in audio_buffer
                    if now - c["timestamp"] < ANALYSIS_WINDOW_SECONDS * 2
                ]

                chunk_count += 1

                # Run analysis periodically
                if chunk_count % ANALYSIS_INTERVAL_CHUNKS == 0 and len(audio_buffer) >= 2:
                    recent = [c for c in audio_buffer if now - c["timestamp"] < ANALYSIS_WINDOW_SECONDS]
                    if not recent:
                        continue

                    merged_pcm = np.concatenate([c["pcm"] for c in recent])

                    if config["enable_separation"]:
                        # Multi-bird mode: separate sources, classify each, estimate individual directions
                        await _handle_multi_bird(
                            websocket, merged_pcm, recent, sample_rate, config
                        )
                    else:
                        # Single-bird mode (original behavior)
                        await _handle_single_bird(
                            websocket, merged_pcm, recent, sample_rate, config
                        )

    except WebSocketDisconnect:
        logger.info("Bird locator WebSocket disconnected")
        if room_id and room_id in device_rooms and device_id:
            device_rooms[room_id].remove_device(device_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e)


async def _handle_single_bird(websocket, merged_pcm, recent, sample_rate, config):
    """Original single-bird detection and direction estimation."""
    try:
        detections = analyze_audio(
            pcm_data=merged_pcm,
            sample_rate=sample_rate,
            latitude=config["latitude"],
            longitude=config["longitude"],
            min_confidence=0.25,
        )
    except Exception as e:
        logger.error("BirdNET analysis failed: %s", e)
        detections = []

    direction = None
    try:
        direction_chunks = [{"pcm_data": c["pcm"], "heading": c["heading"]} for c in recent]
        direction = estimate_direction(direction_chunks, sample_rate)
    except Exception as e:
        logger.warning("Direction estimation failed: %s", e)

    if detections:
        await websocket.send_json({
            "type": "detection",
            "detections": detections,
            "direction": direction,
        })
        logger.info(
            "Detected: %s (%.0f%%) heading=%s",
            detections[0]["species"],
            detections[0]["confidence"] * 100,
            direction.get("heading") if direction else "unknown",
        )


async def _handle_multi_bird(websocket, merged_pcm, recent, sample_rate, config):
    """Separate overlapping bird calls, classify each, estimate individual directions."""
    # Step 1: Separate sources
    try:
        sources = separate_sources(merged_pcm, sample_rate, n_sources=0)
    except Exception as e:
        logger.error("Source separation failed: %s", e)
        # Fall back to single-bird mode
        await _handle_single_bird(websocket, merged_pcm, recent, sample_rate, config)
        return

    if not sources:
        return

    # Step 2: Classify each separated source with BirdNET
    birds = []
    for idx, source in enumerate(sources):
        try:
            detections = analyze_audio(
                pcm_data=source["audio"],
                sample_rate=sample_rate,
                latitude=config["latitude"],
                longitude=config["longitude"],
                min_confidence=0.20,
            )
        except Exception:
            detections = []

        species = detections[0]["species"] if detections else "Unknown bird"
        confidence = detections[0]["confidence"] if detections else 0.0

        birds.append({
            "source_idx": idx,
            "species": species,
            "confidence": round(confidence, 3),
            "dominant_freq": source["dominant_freq"],
            "freq_range": source["freq_range"],
            "energy": source["energy"],
        })

    # Step 3: Estimate direction for each source independently
    try:
        direction_chunks = [{"pcm_data": c["pcm"], "heading": c["heading"]} for c in recent]
        source_directions = estimate_directions_multi_source(sources, direction_chunks, sample_rate)

        # Merge direction info into birds
        for sd in source_directions:
            idx = sd["source_idx"]
            if idx < len(birds):
                birds[idx]["heading"] = sd["heading"]
                birds[idx]["direction_confidence"] = sd["confidence"]
    except Exception as e:
        logger.warning("Multi-source direction estimation failed: %s", e)

    if birds:
        await websocket.send_json({
            "type": "multi_detection",
            "birds": birds,
            "source_count": len(sources),
        })
        logger.info("Multi-bird: %d sources, top: %s", len(birds), birds[0]["species"])
