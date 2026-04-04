"""Amplitude-based direction estimation for bird sounds.

Uses bandpass filtering in the bird vocalization range (1-8 kHz)
and correlates RMS amplitude with compass headings to estimate
the direction of the loudest sound source.
"""

import numpy as np
from scipy.signal import butter, sosfilt

# Bird vocalization frequency range
BIRD_FREQ_LOW = 1000   # Hz
BIRD_FREQ_HIGH = 8000  # Hz
NOISE_THRESHOLD = 0.005  # RMS below this is considered silence


def _bandpass_filter(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply a bandpass filter for bird vocalization frequencies."""
    nyquist = sample_rate / 2
    low = BIRD_FREQ_LOW / nyquist
    high = min(BIRD_FREQ_HIGH / nyquist, 0.99)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, data)


def compute_rms(data: np.ndarray) -> float:
    """Compute RMS amplitude of audio data."""
    return float(np.sqrt(np.mean(data ** 2)))


def estimate_direction(chunks: list[dict], sample_rate: int = 44100) -> dict | None:
    """
    Estimate the direction of a bird sound from audio chunks with headings.

    Args:
        chunks: List of {"pcm_data": np.ndarray, "heading": float (degrees 0-360)}
        sample_rate: Audio sample rate

    Returns:
        {"heading": float, "confidence": float, "amplitudes": list} or None if too quiet
    """
    if not chunks:
        return None

    amplitudes = []
    for chunk in chunks:
        pcm = chunk["pcm_data"]
        heading = chunk["heading"]

        # Apply bandpass filter for bird frequencies
        filtered = _bandpass_filter(pcm, sample_rate)
        rms = compute_rms(filtered)
        amplitudes.append({"heading": heading, "rms": rms})

    # Find peak amplitude
    max_entry = max(amplitudes, key=lambda x: x["rms"])

    if max_entry["rms"] < NOISE_THRESHOLD:
        return None

    # Compute a confidence score based on how much the peak stands out
    rms_values = [a["rms"] for a in amplitudes]
    mean_rms = np.mean(rms_values)
    if mean_rms > 0:
        peak_ratio = max_entry["rms"] / mean_rms
        # Normalize: ratio of 1 = no directionality, ratio of 3+ = strong
        confidence = min(1.0, (peak_ratio - 1.0) / 2.0)
    else:
        confidence = 0.0

    return {
        "heading": max_entry["heading"],
        "confidence": round(confidence, 3),
        "amplitudes": [{"heading": a["heading"], "rms": round(a["rms"], 5)} for a in amplitudes],
    }
