"""Bird sound classification using BirdNET via birdnetlib."""

import io
import logging
import tempfile
from datetime import datetime

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_analyzer = None


def get_analyzer():
    """Lazy-load the BirdNET analyzer (downloads model on first use)."""
    global _analyzer
    if _analyzer is None:
        from birdnetlib.analyzer import Analyzer
        _analyzer = Analyzer()
        logger.info("BirdNET analyzer loaded")
    return _analyzer


def analyze_audio(
    pcm_data: np.ndarray,
    sample_rate: int = 44100,
    latitude: float = 0.0,
    longitude: float = 0.0,
    min_confidence: float = 0.25,
) -> list[dict]:
    """
    Classify bird sounds in a PCM audio buffer.

    Args:
        pcm_data: Float32 numpy array of audio samples
        sample_rate: Sample rate of the audio
        latitude: GPS latitude for seasonal/regional filtering
        longitude: GPS longitude for seasonal/regional filtering
        min_confidence: Minimum confidence threshold for detections

    Returns:
        List of detections: [{"species": str, "confidence": float, "start_time": float, "end_time": float}]
    """
    from birdnetlib import Recording

    analyzer = get_analyzer()

    # Write PCM to a temporary WAV file (birdnetlib requires a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, pcm_data, sample_rate)

        recording = Recording(
            analyzer,
            tmp.name,
            lat=latitude,
            lon=longitude,
            date=datetime.now(),
            min_conf=min_confidence,
        )
        recording.analyze()

    detections = []
    for det in recording.detections:
        detections.append({
            "species": det.get("common_name", det.get("scientific_name", "Unknown")),
            "scientific_name": det.get("scientific_name", ""),
            "confidence": round(det.get("confidence", 0.0), 3),
            "start_time": det.get("start_time", 0.0),
            "end_time": det.get("end_time", 0.0),
        })

    return detections
