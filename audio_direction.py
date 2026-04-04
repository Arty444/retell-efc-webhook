"""Direction estimation for bird sounds.

Supports three modes:
1. Single-mic amplitude scanning: correlate RMS with compass headings as user rotates
2. Stereo mic ITD: use phone's multiple microphones for inter-aural time difference
3. Per-source direction: estimate direction for each separated bird call independently

Uses bandpass filtering in the bird vocalization range (1-8 kHz).
"""

import numpy as np
from scipy.signal import butter, sosfilt, correlate

# Bird vocalization frequency range
BIRD_FREQ_LOW = 1000   # Hz
BIRD_FREQ_HIGH = 8000  # Hz
NOISE_THRESHOLD = 0.005  # RMS below this is considered silence

# Typical phone mic spacing (top to bottom mic): ~12-15 cm
DEFAULT_MIC_SPACING_M = 0.14  # meters
SPEED_OF_SOUND = 343.0  # m/s


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
    (Single-mic amplitude scanning mode)

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


def estimate_direction_stereo(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    sample_rate: int = 44100,
    device_heading: float = 0.0,
    mic_spacing: float = DEFAULT_MIC_SPACING_M,
) -> dict | None:
    """
    Estimate sound direction using two microphone channels (stereo).

    Uses Interaural Time Difference (ITD) - the same principle that humans
    and owls use for horizontal sound localization. The time delay between
    when a sound reaches each microphone reveals the angle of arrival.

    Most phones have at least 2 mics (top and bottom). When held vertically,
    this gives left-right differentiation. With the phone horizontal,
    it gives front-back differentiation.

    Args:
        channel_1: Audio from microphone 1 (e.g., bottom mic)
        channel_2: Audio from microphone 2 (e.g., top mic)
        sample_rate: Sample rate in Hz
        device_heading: Current compass heading of the device in degrees
        mic_spacing: Distance between microphones in meters

    Returns:
        {"heading": float, "confidence": float, "itd_ms": float, "angle_offset": float}
    """
    # Bandpass filter both channels
    ch1 = _bandpass_filter(channel_1, sample_rate)
    ch2 = _bandpass_filter(channel_2, sample_rate)

    # Check if there's actually bird sound present
    rms1 = compute_rms(ch1)
    rms2 = compute_rms(ch2)
    if max(rms1, rms2) < NOISE_THRESHOLD:
        return None

    # Ensure same length
    min_len = min(len(ch1), len(ch2))
    ch1 = ch1[:min_len]
    ch2 = ch2[:min_len]

    # Cross-correlation to find time delay (GCC-PHAT for robustness)
    n_fft = 2 ** int(np.ceil(np.log2(2 * min_len - 1)))
    CH1 = np.fft.rfft(ch1, n=n_fft)
    CH2 = np.fft.rfft(ch2, n=n_fft)

    cross = CH1 * np.conj(CH2)
    magnitude = np.abs(cross)
    magnitude[magnitude < 1e-10] = 1e-10
    gcc = np.fft.irfft(cross / magnitude, n=n_fft)

    # Maximum possible delay based on mic spacing
    max_delay_samples = int(mic_spacing / SPEED_OF_SOUND * sample_rate) + 5
    max_delay_samples = min(max_delay_samples, n_fft // 2)

    # Extract valid correlation range
    gcc_valid = np.concatenate([
        gcc[-max_delay_samples:],
        gcc[:max_delay_samples + 1],
    ])

    peak_idx = np.argmax(np.abs(gcc_valid))
    delay_samples = peak_idx - max_delay_samples
    delay_seconds = delay_samples / sample_rate

    # Convert time delay to angle using: sin(theta) = delay * speed_of_sound / mic_spacing
    sin_theta = np.clip(delay_seconds * SPEED_OF_SOUND / mic_spacing, -1.0, 1.0)
    angle_offset = float(np.degrees(np.arcsin(sin_theta)))

    # Confidence from cross-correlation peak sharpness
    peak_val = float(np.abs(gcc_valid[peak_idx]))
    avg_val = float(np.mean(np.abs(gcc_valid)))
    confidence = min(1.0, max(0.0, (peak_val / (avg_val + 1e-10) - 1) / 5))

    # Combine with device heading
    estimated_heading = (device_heading + angle_offset) % 360

    return {
        "heading": round(estimated_heading, 1),
        "confidence": round(confidence, 3),
        "itd_ms": round(delay_seconds * 1000, 3),
        "angle_offset": round(angle_offset, 1),
    }


def estimate_directions_multi_source(
    sources: list[dict],
    chunks: list[dict],
    sample_rate: int = 44100,
) -> list[dict]:
    """
    Estimate direction for each separated bird sound source independently.

    For each source from sound_separation.separate_sources(), applies
    the source's frequency profile as a filter and then estimates direction
    from the amplitude-heading correlation in that specific frequency band.

    Args:
        sources: Output from sound_separation.separate_sources()
                 Each has "freq_range": (low_hz, high_hz) and "dominant_freq"
        chunks: List of {"pcm_data": np.ndarray, "heading": float}
                Raw audio chunks at different compass headings

    Returns:
        List of {"source_idx": int, "species": str, "heading": float,
                 "confidence": float, "dominant_freq": float}
    """
    if not sources or not chunks:
        return []

    results = []
    for idx, source in enumerate(sources):
        freq_low, freq_high = source.get("freq_range", (BIRD_FREQ_LOW, BIRD_FREQ_HIGH))

        # Skip sources with invalid frequency ranges
        if freq_high <= freq_low or freq_low <= 0:
            continue

        # Create a custom bandpass filter for this source's frequency range
        amplitudes = []
        for chunk in chunks:
            pcm = chunk["pcm_data"]
            heading = chunk["heading"]

            nyquist = sample_rate / 2
            low = max(freq_low / nyquist, 0.01)
            high = min(freq_high / nyquist, 0.99)

            if low >= high:
                continue

            try:
                sos = butter(4, [low, high], btype="band", output="sos")
                filtered = sosfilt(sos, pcm)
                rms = compute_rms(filtered)
                amplitudes.append({"heading": heading, "rms": rms})
            except Exception:
                continue

        if not amplitudes:
            continue

        max_entry = max(amplitudes, key=lambda x: x["rms"])
        if max_entry["rms"] < NOISE_THRESHOLD:
            continue

        rms_values = [a["rms"] for a in amplitudes]
        mean_rms = np.mean(rms_values)
        if mean_rms > 0:
            peak_ratio = max_entry["rms"] / mean_rms
            confidence = min(1.0, (peak_ratio - 1.0) / 2.0)
        else:
            confidence = 0.0

        results.append({
            "source_idx": idx,
            "heading": max_entry["heading"],
            "confidence": round(confidence, 3),
            "dominant_freq": source.get("dominant_freq", 0),
            "freq_range": source.get("freq_range", (0, 0)),
            "energy": source.get("energy", 0),
        })

    return results
