import os
import base64
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from bird_analyzer import analyze_audio
from audio_direction import (
    estimate_direction,
    estimate_direction_stereo,
    estimate_direction_stereo_continuous,
    estimate_direction_4mic,
    estimate_direction_4mic_continuous,
    estimate_directions_multi_source,
)
from sound_separation import separate_sources
from multi_device_locator import DeviceRoom
from distance_estimator import BirdDistanceEstimator
from acoustic_slam import IMUReading
from visual_detector import detect_bird_in_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sanitize_for_json(obj):
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        sanitized = [_sanitize_for_json(v) for v in obj]
        return tuple(sanitized) if isinstance(obj, tuple) else sanitized
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


async def _send_json(ws: WebSocket, data: dict):
    """Send JSON via WebSocket, sanitizing numpy types first."""
    await ws.send_json(_sanitize_for_json(data))


app = FastAPI(title="Bird Sound Locator")

# Mount static files for PWA
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ──────────────────────────────────────────────────────────────────

# Multi-device rooms for TDOA triangulation
device_rooms: dict[str, DeviceRoom] = {}


@app.get("/")
async def health():
    return {"status": "ok", "service": "bird-sound-locator"}


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
      {"type": "audio_4mic", "channels": ["<base64>",...], "sample_rate": int, "heading": float, "pitch": float, "roll": float}
      {"type": "join_room", "room_id": str, "device_id": str}

    Server sends:
      {"type": "detection", "detections": [...], "direction": {...}}
      {"type": "multi_detection", "birds": [...]}
      {"type": "triangulation", "result": {...}}
      {"type": "status", "message": str}
    """
    await websocket.accept()
    logger.info("Bird locator WebSocket connected")

    # Per-connection state
    config = {"latitude": 0.0, "longitude": 0.0, "enable_separation": False}
    audio_buffer = []
    stereo_buffer = []
    has_stereo = False
    mic_count = 1
    ANALYSIS_WINDOW_SECONDS = 3.0
    ANALYSIS_INTERVAL_CHUNKS = 4
    chunk_count = 0
    room_id = None
    device_id = None
    distance_estimator = BirdDistanceEstimator()

    try:
        await _send_json(websocket, {"type": "status", "message": "Connected. Start listening to detect birds!"})

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "config":
                config["latitude"] = float(data.get("latitude", 0))
                config["longitude"] = float(data.get("longitude", 0))
                config["enable_separation"] = bool(data.get("enable_separation", False))
                mode = "multi-bird" if config["enable_separation"] else "single"
                await _send_json(websocket, {
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
                    await _send_json(websocket, {
                        "type": "room_joined",
                        "room_id": room_id,
                        "device_id": device_id,
                        "device_count": room.get_device_count(),
                        "devices": room.get_device_positions(),
                    })
                continue

            if msg_type == "imu_data":
                try:
                    imu = IMUReading(
                        timestamp=float(data.get("timestamp", datetime.utcnow().timestamp())),
                        accel_x=float(data.get("accel_x", 0)),
                        accel_y=float(data.get("accel_y", 0)),
                        accel_z=float(data.get("accel_z", 0)),
                        gyro_x=float(data.get("gyro_x", 0)),
                        gyro_y=float(data.get("gyro_y", 0)),
                        gyro_z=float(data.get("gyro_z", 0)),
                        heading=float(data.get("heading", 0)),
                        pitch=float(data.get("pitch", 90)),
                        roll=float(data.get("roll", 0)),
                    )
                    distance_estimator.update_imu(imu)
                except Exception as e:
                    logger.debug("IMU data error: %s", e)
                continue

            if msg_type == "camera_frame":
                try:
                    frame_b64 = data.get("data", "")
                    frame_bytes = base64.b64decode(frame_b64) if frame_b64 else b""
                    frame_w = int(data.get("width", 0))
                    frame_h = int(data.get("height", 0))
                    cam_heading = float(data.get("heading", 0))
                    zoom = float(data.get("zoom", 1.0))

                    if frame_bytes and frame_w > 0 and frame_h > 0:
                        detection_result = detect_bird_in_frame(
                            frame_bytes, frame_w, frame_h,
                            target_heading=getattr(distance_estimator, '_last_azimuth', None),
                            camera_heading=cam_heading,
                        )
                        if detection_result and detection_result.get("bbox"):
                            distance_estimator.update_visual(
                                detection_result["bbox"],
                                frame_w, frame_h,
                                zoom=zoom,
                            )
                            await _send_json(websocket, {
                                "type": "visual_detection",
                                "bbox": detection_result["bbox"],
                                "confidence": detection_result["confidence"],
                                "in_frame": detection_result["in_frame"],
                                "distance": distance_estimator.get_distance(),
                            })
                except Exception as e:
                    logger.debug("Camera frame error: %s", e)
                continue

            if msg_type == "audio_4mic":
                has_stereo = True
                try:
                    ch_data = data["channels"]
                    channels_np = []
                    for ch_b64 in ch_data:
                        ch_bytes = base64.b64decode(ch_b64)
                        ch_int16 = np.frombuffer(ch_bytes, dtype=np.int16)
                        channels_np.append(ch_int16.astype(np.float32) / 32768.0)
                    sr = int(data.get("sample_rate", 44100))
                    heading = float(data.get("heading", 0))
                    pitch = float(data.get("pitch", 90))
                    roll = float(data.get("roll", 0))
                    mic_count = len(channels_np)
                except Exception as e:
                    logger.warning("Failed to decode 4-mic audio: %s", e)
                    continue

                audio_buffer.append({
                    "pcm": channels_np[0],
                    "heading": heading,
                    "timestamp": datetime.utcnow().timestamp(),
                })

                stereo_buffer.append({
                    "channels": channels_np,
                    "heading": heading, "pitch": pitch, "roll": roll,
                    "timestamp": datetime.utcnow().timestamp(),
                })

                now = datetime.utcnow().timestamp()
                stereo_buffer = [c for c in stereo_buffer if now - c["timestamp"] < ANALYSIS_WINDOW_SECONDS]
                audio_buffer = [c for c in audio_buffer if now - c["timestamp"] < ANALYSIS_WINDOW_SECONDS * 2]

                chunk_count += 1

                rms = float(np.sqrt(np.mean(channels_np[0]**2)))
                distance_estimator.update_audio(channels_np[0], sr, rms)

                if chunk_count % 2 == 0 and len(stereo_buffer) >= 2:
                    dir_4mic = estimate_direction_4mic_continuous(stereo_buffer, sr)
                    if dir_4mic:
                        distance_estimator.update_direction(
                            dir_4mic.get("heading", 0),
                            dir_4mic.get("elevation", 0),
                            dir_4mic.get("confidence", 0),
                        )
                        distance_estimator._last_azimuth = dir_4mic.get("heading")
                        dist_info = distance_estimator.get_distance()
                        await _send_json(websocket, {
                            "type": "direction_4mic",
                            "direction": dir_4mic,
                            "mic_count": mic_count,
                            "distance": dist_info,
                        })

                if chunk_count % ANALYSIS_INTERVAL_CHUNKS == 0 and len(audio_buffer) >= 2:
                    recent = [c for c in audio_buffer if now - c["timestamp"] < ANALYSIS_WINDOW_SECONDS]
                    if recent:
                        merged_pcm = np.concatenate([c["pcm"] for c in recent])
                        if config["enable_separation"]:
                            await _handle_multi_bird(websocket, merged_pcm, recent, sr, config, distance_estimator)
                        else:
                            try:
                                detections = analyze_audio(
                                    pcm_data=merged_pcm, sample_rate=sr,
                                    latitude=config["latitude"], longitude=config["longitude"],
                                    min_confidence=0.25,
                                )
                            except Exception as e:
                                logger.error("BirdNET analysis failed: %s", e)
                                detections = []

                            dir_4mic = estimate_direction_4mic_continuous(stereo_buffer, sr)
                            if detections:
                                distance_estimator.update_species(detections[0]["species"])
                                dist_info = distance_estimator.get_distance()
                                await _send_json(websocket, {
                                    "type": "detection",
                                    "detections": detections,
                                    "direction": dir_4mic,
                                    "distance": dist_info,
                                })
                                logger.info(
                                    "4-mic detected: %s (%.0f%%) heading=%s elev=%s dist=%.1fm",
                                    detections[0]["species"],
                                    detections[0]["confidence"] * 100,
                                    dir_4mic.get("heading") if dir_4mic else "?",
                                    dir_4mic.get("elevation") if dir_4mic else "?",
                                    dist_info.get("distance", 0),
                                )
                continue

            if msg_type == "audio_stereo":
                has_stereo = True
                try:
                    ch1_bytes = base64.b64decode(data["ch1"])
                    ch2_bytes = base64.b64decode(data["ch2"])
                    ch1 = np.frombuffer(ch1_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    ch2 = np.frombuffer(ch2_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    sr = int(data.get("sample_rate", 44100))
                    heading = float(data.get("heading", 0))
                    pitch = float(data.get("pitch", 90))
                    roll = float(data.get("roll", 0))
                except Exception as e:
                    logger.warning("Failed to decode stereo audio: %s", e)
                    continue

                audio_buffer.append({
                    "pcm": ch1,
                    "heading": heading,
                    "timestamp": datetime.utcnow().timestamp(),
                })

                stereo_buffer.append({
                    "ch1": ch1, "ch2": ch2,
                    "heading": heading, "pitch": pitch, "roll": roll,
                    "timestamp": datetime.utcnow().timestamp(),
                })

                now = datetime.utcnow().timestamp()
                stereo_buffer = [c for c in stereo_buffer if now - c["timestamp"] < ANALYSIS_WINDOW_SECONDS]
                audio_buffer = [c for c in audio_buffer if now - c["timestamp"] < ANALYSIS_WINDOW_SECONDS * 2]

                rms = float(np.sqrt(np.mean(ch1**2)))
                distance_estimator.update_audio(ch1, sr, rms)

                chunk_count += 1

                if chunk_count % 2 == 0 and len(stereo_buffer) >= 2:
                    stereo_dir = estimate_direction_stereo_continuous(stereo_buffer, sr)
                    if stereo_dir:
                        distance_estimator.update_direction(
                            stereo_dir.get("heading", 0),
                            stereo_dir.get("elevation", 0),
                            stereo_dir.get("confidence", 0),
                        )
                        distance_estimator._last_azimuth = stereo_dir.get("heading")
                        dist_info = distance_estimator.get_distance()
                        await _send_json(websocket, {
                            "type": "stereo_direction",
                            "direction": stereo_dir,
                            "distance": dist_info,
                        })

                if chunk_count % ANALYSIS_INTERVAL_CHUNKS == 0 and len(audio_buffer) >= 2:
                    recent = [c for c in audio_buffer if now - c["timestamp"] < ANALYSIS_WINDOW_SECONDS]
                    if recent:
                        merged_pcm = np.concatenate([c["pcm"] for c in recent])
                        if config["enable_separation"]:
                            await _handle_multi_bird(websocket, merged_pcm, recent, sr, config, distance_estimator)
                        else:
                            try:
                                detections = analyze_audio(
                                    pcm_data=merged_pcm, sample_rate=sr,
                                    latitude=config["latitude"], longitude=config["longitude"],
                                    min_confidence=0.25,
                                )
                            except Exception as e:
                                logger.error("BirdNET analysis failed: %s", e)
                                detections = []

                            stereo_dir = estimate_direction_stereo_continuous(stereo_buffer, sr)
                            if detections:
                                distance_estimator.update_species(detections[0]["species"])
                                dist_info = distance_estimator.get_distance()
                                await _send_json(websocket, {
                                    "type": "detection",
                                    "detections": detections,
                                    "direction": stereo_dir,
                                    "distance": dist_info,
                                })
                continue

            if msg_type == "audio_chunk":
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

                if room_id and room_id in device_rooms and device_id:
                    room = device_rooms[room_id]
                    room.submit_audio(device_id, float32_data, datetime.utcnow().timestamp(), sample_rate)

                    if room.can_localize():
                        tri_result = room.localize()
                        if tri_result:
                            await _send_json(websocket, {
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

                now = datetime.utcnow().timestamp()
                audio_buffer = [
                    c for c in audio_buffer
                    if now - c["timestamp"] < ANALYSIS_WINDOW_SECONDS * 2
                ]

                chunk_count += 1

                if chunk_count % ANALYSIS_INTERVAL_CHUNKS == 0 and len(audio_buffer) >= 2:
                    recent = [c for c in audio_buffer if now - c["timestamp"] < ANALYSIS_WINDOW_SECONDS]
                    if not recent:
                        continue

                    merged_pcm = np.concatenate([c["pcm"] for c in recent])

                    rms = float(np.sqrt(np.mean(merged_pcm**2)))
                    distance_estimator.update_audio(merged_pcm, sample_rate, rms)

                    if config["enable_separation"]:
                        await _handle_multi_bird(
                            websocket, merged_pcm, recent, sample_rate, config, distance_estimator
                        )
                    else:
                        await _handle_single_bird(
                            websocket, merged_pcm, recent, sample_rate, config, distance_estimator
                        )

    except WebSocketDisconnect:
        logger.info("Bird locator WebSocket disconnected")
        if room_id and room_id in device_rooms and device_id:
            device_rooms[room_id].remove_device(device_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e)


async def _handle_single_bird(websocket, merged_pcm, recent, sample_rate, config, dist_est=None):
    """Single-bird detection, direction estimation, and distance fusion."""
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
        dist_info = None
        if dist_est:
            dist_est.update_species(detections[0]["species"])
            if direction and direction.get("heading") is not None:
                dist_est.update_direction(
                    direction.get("heading", 0),
                    direction.get("elevation", 0),
                    direction.get("confidence", 0),
                )
            dist_info = dist_est.get_distance()

        await _send_json(websocket, {
            "type": "detection",
            "detections": detections,
            "direction": direction,
            "distance": dist_info,
        })
        logger.info(
            "Detected: %s (%.0f%%) heading=%s dist=%s",
            detections[0]["species"],
            detections[0]["confidence"] * 100,
            direction.get("heading") if direction else "unknown",
            f"{dist_info['distance']:.1f}m" if dist_info else "?",
        )


async def _handle_multi_bird(websocket, merged_pcm, recent, sample_rate, config, dist_est=None):
    """Separate overlapping bird calls, classify each, estimate individual directions."""
    try:
        sources = separate_sources(merged_pcm, sample_rate, n_sources=0)
    except Exception as e:
        logger.error("Source separation failed: %s", e)
        await _handle_single_bird(websocket, merged_pcm, recent, sample_rate, config, dist_est)
        return

    if not sources:
        return

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

    try:
        direction_chunks = [{"pcm_data": c["pcm"], "heading": c["heading"]} for c in recent]
        source_directions = estimate_directions_multi_source(sources, direction_chunks, sample_rate)

        for sd in source_directions:
            idx = sd["source_idx"]
            if idx < len(birds):
                birds[idx]["heading"] = sd["heading"]
                birds[idx]["direction_confidence"] = sd["confidence"]
    except Exception as e:
        logger.warning("Multi-source direction estimation failed: %s", e)

    if birds:
        dist_info = None
        if dist_est and birds[0].get("species") != "Unknown bird":
            dist_est.update_species(birds[0]["species"])
            dist_info = dist_est.get_distance()

        await _send_json(websocket, {
            "type": "multi_detection",
            "birds": birds,
            "source_count": len(sources),
            "distance": dist_info,
        })
        logger.info("Multi-bird: %d sources, top: %s", len(birds), birds[0]["species"])
