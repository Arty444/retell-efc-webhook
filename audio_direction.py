"""Direction estimation for bird sounds using phone's built-in microphones.

Primary method: Stereo/multi-mic ITD (Interaural Time Difference)
- Most phones have 2-3 mics (bottom, top, back) spaced ~12-15cm apart
- By computing the time delay between mics via GCC-PHAT cross-correlation,
  we get the angle of arrival -- no rotation required
- Phone orientation (from DeviceOrientation API) maps the mic axis to world coordinates

Fallback: Single-mic amplitude scanning (requires user rotation)

The science: This is exactly how human hearing works. Our two ears are ~17cm
apart. A sound from the left reaches the left ear ~0.5ms before the right ear.
The brain uses this Interaural Time Difference (ITD) to localize sounds.
Owls take this further -- their asymmetric ear placement gives them vertical
localization too, exactly like a phone held at an angle.
"""

import math
import numpy as np
from scipy.signal import butter, sosfilt

# Bird vocalization frequency range
BIRD_FREQ_LOW = 1000   # Hz
BIRD_FREQ_HIGH = 8000  # Hz
NOISE_THRESHOLD = 0.005  # RMS below this is considered silence

SPEED_OF_SOUND = 343.0  # m/s at ~20C

# Common phone mic configurations (spacing in meters)
PHONE_MIC_CONFIGS = {
    "default": {"spacing": 0.14, "axis": "vertical"},
    "iphone": {"spacing": 0.138, "axis": "vertical"},
    "samsung_galaxy": {"spacing": 0.145, "axis": "vertical"},
    "pixel": {"spacing": 0.140, "axis": "vertical"},
}


def _bandpass_filter(data: np.ndarray, sample_rate: int,
                     low_hz: int = BIRD_FREQ_LOW, high_hz: int = BIRD_FREQ_HIGH) -> np.ndarray:
    """Apply a bandpass filter for given frequency range."""
    nyquist = sample_rate / 2
    low = max(low_hz / nyquist, 0.01)
    high = min(high_hz / nyquist, 0.99)
    if low >= high:
        return data
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, data)


def compute_rms(data: np.ndarray) -> float:
    """Compute RMS amplitude of audio data."""
    return float(np.sqrt(np.mean(data ** 2)))


def _gcc_phat_delay(sig1: np.ndarray, sig2: np.ndarray, sample_rate: int,
                    max_delay_seconds: float = 0.001) -> dict | None:
    """
    Compute time delay between two signals using GCC-PHAT.

    GCC-PHAT (Generalized Cross-Correlation with Phase Transform) whitens
    the cross-power spectrum, producing a sharper delay peak than plain
    cross-correlation. More robust to noise and reflections.
    """
    min_len = min(len(sig1), len(sig2))
    if min_len < 128:
        return None

    sig1 = sig1[:min_len]
    sig2 = sig2[:min_len]

    n_fft = 2 ** int(np.ceil(np.log2(2 * min_len - 1)))
    S1 = np.fft.rfft(sig1, n=n_fft)
    S2 = np.fft.rfft(sig2, n=n_fft)

    cross = S1 * np.conj(S2)
    magnitude = np.abs(cross)
    magnitude[magnitude < 1e-10] = 1e-10

    gcc = np.fft.irfft(cross / magnitude, n=n_fft)

    max_delay_samples = int(max_delay_seconds * sample_rate) + 2
    max_delay_samples = min(max_delay_samples, n_fft // 2)

    gcc_valid = np.concatenate([
        gcc[-max_delay_samples:],
        gcc[:max_delay_samples + 1],
    ])

    peak_idx = np.argmax(np.abs(gcc_valid))
    peak_value = float(np.abs(gcc_valid[peak_idx]))

    delay_samples = peak_idx - max_delay_samples
    delay_seconds = delay_samples / sample_rate

    # Sub-sample refinement via parabolic interpolation
    if 0 < peak_idx < len(gcc_valid) - 1:
        alpha = float(np.abs(gcc_valid[peak_idx - 1]))
        beta = peak_value
        gamma = float(np.abs(gcc_valid[peak_idx + 1]))
        denom = 2 * beta - alpha - gamma
        if denom != 0:
            p = 0.5 * (alpha - gamma) / denom
            delay_seconds = (delay_samples + p) / sample_rate

    # Confidence: peak sharpness
    avg_value = float(np.mean(np.abs(gcc_valid)))
    strength = min(1.0, max(0.0, (peak_value / (avg_value + 1e-10) - 1) / 5))

    return {
        "delay_seconds": delay_seconds,
        "delay_samples": delay_samples,
        "strength": strength,
    }


def _map_to_world_coordinates(
    angle_on_axis: float,
    heading: float,
    pitch: float,
    roll: float,
) -> tuple[float, float, str]:
    """
    Map an angle measured along the phone's mic axis to world heading/elevation.

    Phone orientations and what the mic axis measures:
    - Portrait upright (pitch ~90): mic axis is vertical -> measures elevation
    - Landscape (roll ~90): mic axis is horizontal -> measures azimuth directly
    - Angled: decompose using orientation angles
    """
    roll_rad = math.radians(roll)

    horizontal_component = abs(math.sin(roll_rad))
    vertical_component = abs(math.cos(roll_rad))

    if horizontal_component > vertical_component:
        # Mics are more horizontal -> angle maps to azimuth
        sign = 1.0 if roll > 0 else -1.0
        world_heading = (heading + sign * angle_on_axis) % 360
        elevation = 0.0
        mic_channel = "azimuth"
    else:
        # Mics are more vertical -> angle maps to elevation
        elevation = angle_on_axis
        world_heading = heading
        mic_channel = "elevation"

    return (world_heading, elevation, mic_channel)


def estimate_direction_stereo(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    sample_rate: int = 44100,
    device_heading: float = 0.0,
    device_pitch: float = 90.0,
    device_roll: float = 0.0,
    mic_spacing: float = None,
    phone_model: str = "default",
) -> dict | None:
    """
    PRIMARY direction method: Use phone's built-in stereo mics. No rotation needed.

    How it works:
    1. Bandpass filter both mic channels for bird frequencies (1-8 kHz)
    2. GCC-PHAT cross-correlation to find the time delay between mics
    3. Convert time delay to angle: sin(theta) = delay * c / mic_spacing
    4. Map the angle from mic axis to world coordinates using device orientation

    Phone mic geometry:
    - Bottom mic (ch1) and top mic (ch2) are ~14cm apart
    - When phone is upright (portrait), mics are vertically aligned
      -> ITD gives elevation angle
    - When phone is sideways (landscape), mics are horizontal
      -> ITD gives azimuth (left/right) directly
    - Device orientation sensors tell us which case we're in
    """
    config = PHONE_MIC_CONFIGS.get(phone_model, PHONE_MIC_CONFIGS["default"])
    if mic_spacing is None:
        mic_spacing = config["spacing"]

    ch1 = _bandpass_filter(channel_1, sample_rate)
    ch2 = _bandpass_filter(channel_2, sample_rate)

    rms1 = compute_rms(ch1)
    rms2 = compute_rms(ch2)
    if max(rms1, rms2) < NOISE_THRESHOLD:
        return None

    max_delay = mic_spacing / SPEED_OF_SOUND
    result = _gcc_phat_delay(ch1, ch2, sample_rate, max_delay * 1.5)
    if result is None or result["strength"] < 0.03:
        return None

    delay = result["delay_seconds"]
    itd_microseconds = delay * 1_000_000

    sin_theta = np.clip(delay * SPEED_OF_SOUND / mic_spacing, -1.0, 1.0)
    angle_on_axis = float(np.degrees(np.arcsin(sin_theta)))

    heading, elevation, mic_channel = _map_to_world_coordinates(
        angle_on_axis, device_heading, device_pitch, device_roll
    )

    return {
        "heading": round(heading, 1),
        "elevation": round(elevation, 1),
        "confidence": round(result["strength"], 3),
        "itd_us": round(itd_microseconds, 1),
        "angle_on_mic_axis": round(angle_on_axis, 1),
        "mic_channel": mic_channel,
    }


def estimate_direction_stereo_continuous(
    chunks: list[dict],
    sample_rate: int = 44100,
    phone_model: str = "default",
) -> dict | None:
    """
    Continuous ITD direction estimation from multiple stereo audio chunks.
    Averages ITD measurements across time for more stable direction estimates.

    Args:
        chunks: List of {
            "ch1": np.ndarray, "ch2": np.ndarray,
            "heading": float, "pitch": float, "roll": float
        }
    """
    if not chunks:
        return None

    estimates = []
    for chunk in chunks:
        est = estimate_direction_stereo(
            channel_1=chunk["ch1"],
            channel_2=chunk["ch2"],
            sample_rate=sample_rate,
            device_heading=chunk.get("heading", 0),
            device_pitch=chunk.get("pitch", 90),
            device_roll=chunk.get("roll", 0),
            phone_model=phone_model,
        )
        if est and est["confidence"] > 0.05:
            estimates.append(est)

    if not estimates:
        return None

    # Circular mean for headings
    sin_sum = sum(math.sin(math.radians(e["heading"])) for e in estimates)
    cos_sum = sum(math.cos(math.radians(e["heading"])) for e in estimates)
    avg_heading = math.degrees(math.atan2(sin_sum, cos_sum)) % 360

    avg_elevation = sum(e["elevation"] for e in estimates) / len(estimates)
    avg_confidence = sum(e["confidence"] for e in estimates) / len(estimates)
    avg_itd = sum(e["itd_us"] for e in estimates) / len(estimates)

    # Boost confidence if measurements are consistent
    heading_variance = np.var([e["heading"] for e in estimates])
    consistency_bonus = max(0, 0.2 - heading_variance / 1000)
    avg_confidence = min(1.0, avg_confidence + consistency_bonus)

    return {
        "heading": round(avg_heading, 1),
        "elevation": round(avg_elevation, 1),
        "confidence": round(avg_confidence, 3),
        "itd_us": round(avg_itd, 1),
        "n_measurements": len(estimates),
        "mic_channel": estimates[0]["mic_channel"],
    }


# --- Fallback: Single-mic amplitude scanning ---

def estimate_direction(chunks: list[dict], sample_rate: int = 44100) -> dict | None:
    """
    Fallback: Estimate direction from audio chunks with headings (requires rotation).
    """
    if not chunks:
        return None

    amplitudes = []
    for chunk in chunks:
        filtered = _bandpass_filter(chunk["pcm_data"], sample_rate)
        rms = compute_rms(filtered)
        amplitudes.append({"heading": chunk["heading"], "rms": rms})

    max_entry = max(amplitudes, key=lambda x: x["rms"])
    if max_entry["rms"] < NOISE_THRESHOLD:
        return None

    rms_values = [a["rms"] for a in amplitudes]
    mean_rms = np.mean(rms_values)
    confidence = min(1.0, (max_entry["rms"] / (mean_rms + 1e-10) - 1.0) / 2.0) if mean_rms > 0 else 0.0

    return {
        "heading": max_entry["heading"],
        "confidence": round(confidence, 3),
        "amplitudes": [{"heading": a["heading"], "rms": round(a["rms"], 5)} for a in amplitudes],
    }


def estimate_directions_multi_source(
    sources: list[dict],
    chunks: list[dict],
    sample_rate: int = 44100,
) -> list[dict]:
    """Estimate direction for each separated bird sound source independently."""
    if not sources or not chunks:
        return []

    results = []
    for idx, source in enumerate(sources):
        freq_low, freq_high = source.get("freq_range", (BIRD_FREQ_LOW, BIRD_FREQ_HIGH))
        if freq_high <= freq_low or freq_low <= 0:
            continue

        amplitudes = []
        for chunk in chunks:
            try:
                filtered = _bandpass_filter(chunk["pcm_data"], sample_rate, int(freq_low), int(freq_high))
                rms = compute_rms(filtered)
                amplitudes.append({"heading": chunk["heading"], "rms": rms})
            except Exception:
                continue

        if not amplitudes:
            continue

        max_entry = max(amplitudes, key=lambda x: x["rms"])
        if max_entry["rms"] < NOISE_THRESHOLD:
            continue

        rms_values = [a["rms"] for a in amplitudes]
        mean_rms = np.mean(rms_values)
        confidence = min(1.0, (max_entry["rms"] / (mean_rms + 1e-10) - 1.0) / 2.0) if mean_rms > 0 else 0.0

        results.append({
            "source_idx": idx,
            "heading": max_entry["heading"],
            "confidence": round(confidence, 3),
            "dominant_freq": source.get("dominant_freq", 0),
            "freq_range": source.get("freq_range", (0, 0)),
            "energy": source.get("energy", 0),
        })

    return results
