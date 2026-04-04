"""3D sound localization using iPhone 16 Pro Max's 4-microphone array.

iPhone 16 Pro Max Microphone Geometry (measured from teardowns):
=============================================================

Device dimensions: 163.0 x 77.6 x 8.25 mm

    MIC 3 (front top, earpiece grille)
    Position: (38.8, 159.0, 0.0) mm  -- center-top of front face
    |
    |   MIC 4 (rear, near camera module)
    |   Position: (20.0, 150.0, 8.25) mm  -- upper-left rear, by camera
    |
    |         FRONT FACE
    |         163mm tall
    |         77.6mm wide
    |
    |
    MIC 1 (bottom-left of USB-C)    MIC 2 (bottom-right of USB-C)
    Position: (30.0, 0.0, 0.0)      Position: (47.6, 0.0, 0.0)
    ~~~~~~~USB-C PORT~~~~~~~
          (centered at 38.8mm)

Mic pair distances:
  Mic 1-2 (bottom pair):    17.6 mm  -- horizontal, gives left/right
  Mic 1-3 (bottom-L to top): 161.9 mm -- vertical, gives up/down  
  Mic 2-3 (bottom-R to top): 160.3 mm -- vertical, confirms up/down
  Mic 1-4 (BL to rear):     152.6 mm -- diagonal+depth, gives front/back
  Mic 2-4 (BR to rear):     153.4 mm -- diagonal+depth, confirms front/back
  Mic 3-4 (top to rear):    20.8 mm  -- small, front/back discrimination

With 4 mics in 3D, we get 6 mic pairs = 6 TDOA measurements.
This is an overdetermined system for 2 unknowns (azimuth, elevation),
solved via least-squares for robust 3D direction estimation.

The math:
  For a sound arriving from direction (azimuth, elevation):
    - Unit vector: d = (cos(el)*sin(az), cos(el)*cos(az), sin(el))  [ENU frame]
    - Expected TDOA between mic i and j: tau_ij = (r_i - r_j) . d / c
    - Measured TDOA: cross-correlation peak via GCC-PHAT
    - Minimize: sum( (measured_tau_ij - expected_tau_ij)^2 ) over (az, el)
"""

import math
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.optimize import minimize

# ─── Constants ──────────────────────────────────────────────────────────────────

SPEED_OF_SOUND = 343.0  # m/s at 20°C
BIRD_FREQ_LOW = 1000    # Hz
BIRD_FREQ_HIGH = 8000   # Hz
NOISE_THRESHOLD = 0.005

# ─── iPhone 16 Pro Max Mic Array Geometry ───────────────────────────────────────
# Positions in millimeters, phone coordinate system:
#   X = across width (left=0, right=77.6)
#   Y = along height (bottom=0, top=163.0)
#   Z = depth (front face=0, back=8.25)
# Converted to meters for acoustic calculations.

IPHONE_16_PM_MICS = {
    "name": "iPhone 16 Pro Max",
    "dimensions_mm": (77.6, 163.0, 8.25),
    "mics": [
        # Mic 0: Bottom-left (left of USB-C port)
        {"label": "bottom_left",  "pos_mm": (30.0,   0.0,  0.0), "face": "bottom"},
        # Mic 1: Bottom-right (right of USB-C port)
        {"label": "bottom_right", "pos_mm": (47.6,   0.0,  0.0), "face": "bottom"},
        # Mic 2: Front-top (earpiece/speaker grille)
        {"label": "front_top",    "pos_mm": (38.8, 159.0,  0.0), "face": "front"},
        # Mic 3: Rear-top (near camera module, upper-left of back)
        {"label": "rear_camera",  "pos_mm": (20.0, 150.0,  8.25), "face": "rear"},
    ],
    # Pre-computed mic positions in meters (converted at module load)
    "mic_positions_m": None,
    # Pre-computed pair distances in meters
    "pair_distances_m": None,
}

# Convert mm to meters and pre-compute pair info
_mics_m = []
for mic in IPHONE_16_PM_MICS["mics"]:
    _mics_m.append(np.array(mic["pos_mm"]) / 1000.0)
IPHONE_16_PM_MICS["mic_positions_m"] = _mics_m

_pairs = []
_pair_distances = []
for i in range(len(_mics_m)):
    for j in range(i + 1, len(_mics_m)):
        dist = float(np.linalg.norm(_mics_m[i] - _mics_m[j]))
        _pairs.append((i, j))
        _pair_distances.append(dist)
IPHONE_16_PM_MICS["pair_distances_m"] = _pair_distances
IPHONE_16_PM_MICS["pairs"] = _pairs

# Maximum TDOA for any pair (largest pair distance / speed of sound)
MAX_PAIR_DISTANCE = max(_pair_distances)
MAX_TDOA = MAX_PAIR_DISTANCE / SPEED_OF_SOUND  # ~0.47ms for 161.9mm


# ─── Signal Processing ─────────────────────────────────────────────────────────

def _bandpass_filter(data: np.ndarray, sample_rate: int,
                     low_hz: int = BIRD_FREQ_LOW, high_hz: int = BIRD_FREQ_HIGH) -> np.ndarray:
    """4th-order Butterworth bandpass for bird vocalization frequencies."""
    nyquist = sample_rate / 2
    low = max(low_hz / nyquist, 0.01)
    high = min(high_hz / nyquist, 0.99)
    if low >= high:
        return data
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, data)


def compute_rms(data: np.ndarray) -> float:
    """Compute RMS amplitude."""
    return float(np.sqrt(np.mean(data ** 2)))


def _gcc_phat_delay(sig1: np.ndarray, sig2: np.ndarray, sample_rate: int,
                    max_delay_seconds: float) -> dict | None:
    """
    GCC-PHAT (Generalized Cross-Correlation with Phase Transform).

    Computes the time delay between two signals by:
    1. FFT both signals
    2. Compute cross-power spectrum: R = S1 * conj(S2)
    3. Whiten (PHAT): R_phat = R / |R|  (removes amplitude, keeps phase)
    4. IFFT to get sharpened cross-correlation
    5. Find peak = time delay

    Sub-sample precision via parabolic interpolation on the peak.

    Returns: {"delay_seconds": float, "strength": float} or None
    """
    min_len = min(len(sig1), len(sig2))
    if min_len < 128:
        return None

    s1 = sig1[:min_len]
    s2 = sig2[:min_len]

    n_fft = 2 ** int(np.ceil(np.log2(2 * min_len - 1)))
    S1 = np.fft.rfft(s1, n=n_fft)
    S2 = np.fft.rfft(s2, n=n_fft)

    cross = S1 * np.conj(S2)
    mag = np.abs(cross)
    mag[mag < 1e-10] = 1e-10
    gcc = np.fft.irfft(cross / mag, n=n_fft)

    max_samples = int(max_delay_seconds * sample_rate) + 3
    max_samples = min(max_samples, n_fft // 2)

    gcc_valid = np.concatenate([gcc[-max_samples:], gcc[:max_samples + 1]])
    peak_idx = np.argmax(np.abs(gcc_valid))
    peak_val = float(np.abs(gcc_valid[peak_idx]))

    delay_samples = peak_idx - max_samples

    # Parabolic sub-sample interpolation
    delay_frac = 0.0
    if 0 < peak_idx < len(gcc_valid) - 1:
        a = float(np.abs(gcc_valid[peak_idx - 1]))
        b = peak_val
        c = float(np.abs(gcc_valid[peak_idx + 1]))
        denom = 2 * b - a - c
        if abs(denom) > 1e-10:
            delay_frac = 0.5 * (a - c) / denom

    delay_seconds = (delay_samples + delay_frac) / sample_rate

    # Strength: peak-to-mean ratio (higher = more confident TDOA)
    avg_val = float(np.mean(np.abs(gcc_valid)))
    strength = min(1.0, max(0.0, (peak_val / (avg_val + 1e-10) - 1) / 5))

    return {"delay_seconds": delay_seconds, "strength": strength}


# ─── 4-Mic 3D Direction Estimation ─────────────────────────────────────────────

def _phone_to_world_direction(azimuth_phone: float, elevation_phone: float,
                               heading: float, pitch: float, roll: float) -> tuple[float, float]:
    """
    Transform a direction from phone coordinates to world coordinates.

    Phone frame (when held in portrait, screen facing user):
      X_phone = right (+) / left (-)
      Y_phone = up (+) / down (-)
      Z_phone = toward user (+) / away (-)

    World frame (ENU):
      X_world = East
      Y_world = North
      Z_world = Up

    Device orientation from DeviceOrientation API:
      heading (alpha): compass heading, 0-360, clockwise from North
      pitch (beta):    -180 to 180, 90 = upright portrait
      roll (gamma):    -90 to 90, 0 = portrait
    """
    # Convert phone-frame direction to unit vector
    az_rad = math.radians(azimuth_phone)
    el_rad = math.radians(elevation_phone)
    dx = math.cos(el_rad) * math.sin(az_rad)
    dy = math.cos(el_rad) * math.cos(az_rad)
    dz = math.sin(el_rad)
    d_phone = np.array([dx, dy, dz])

    # Build rotation matrix from phone to world
    # Roll (gamma) -> rotation about phone Y axis
    # Pitch (beta) -> rotation about phone X axis
    # Heading (alpha) -> rotation about world Z axis
    h = math.radians(heading)
    p = math.radians(pitch)
    r = math.radians(roll)

    # Rotation matrix: R = Rz(heading) * Rx(pitch) * Ry(roll)
    # Rz (heading around vertical)
    Rz = np.array([
        [math.cos(h), -math.sin(h), 0],
        [math.sin(h),  math.cos(h), 0],
        [0,            0,           1],
    ])
    # Rx (pitch around lateral axis)
    Rx = np.array([
        [1,  0,            0],
        [0,  math.cos(p), -math.sin(p)],
        [0,  math.sin(p),  math.cos(p)],
    ])
    # Ry (roll around longitudinal axis)
    Ry = np.array([
        [ math.cos(r), 0, math.sin(r)],
        [ 0,           1, 0],
        [-math.sin(r), 0, math.cos(r)],
    ])

    R = Rz @ Rx @ Ry
    d_world = R @ d_phone

    # Convert world vector to azimuth/elevation
    world_az = math.degrees(math.atan2(d_world[0], d_world[1])) % 360
    world_el = math.degrees(math.asin(np.clip(d_world[2], -1.0, 1.0)))

    return (world_az, world_el)


def estimate_direction_4mic(
    channels: list[np.ndarray],
    sample_rate: int = 44100,
    device_heading: float = 0.0,
    device_pitch: float = 90.0,
    device_roll: float = 0.0,
) -> dict | None:
    """
    3D sound source localization using all 4 iPhone 16 Pro Max microphones.

    Algorithm:
    1. Bandpass filter all 4 channels for bird frequencies
    2. Compute GCC-PHAT TDOA for all 6 mic pairs
    3. Solve for direction (azimuth, elevation) that best fits all 6 TDOAs
       via least-squares optimization
    4. Transform from phone frame to world frame using device orientation

    The system is overdetermined (6 measurements, 2 unknowns) which gives
    noise robustness. With only 2 mics we'd have 1 measurement and ambiguity.
    With 4 mics in 3D positions we can resolve both azimuth AND elevation
    unambiguously.

    Args:
        channels: List of 4 np.ndarray audio signals [bottom_left, bottom_right, front_top, rear_camera]
        sample_rate: Audio sample rate in Hz
        device_heading: Compass heading 0-360 (DeviceOrientation alpha)
        device_pitch: Pitch in degrees (DeviceOrientation beta, 90 = upright)
        device_roll: Roll in degrees (DeviceOrientation gamma, 0 = portrait)

    Returns:
        {
            "heading": float,         # World compass heading to bird (0-360)
            "elevation": float,       # World elevation angle to bird (-90 to 90)
            "confidence": float,      # 0-1 overall confidence
            "azimuth_phone": float,   # Direction in phone frame
            "elevation_phone": float, # Elevation in phone frame
            "tdoa_pairs": list,       # Raw TDOA measurements for debugging
            "residual": float,        # Optimization residual (lower = better fit)
        }
    """
    mic_config = IPHONE_16_PM_MICS
    mic_positions = mic_config["mic_positions_m"]
    pairs = mic_config["pairs"]
    pair_distances = mic_config["pair_distances_m"]

    n_mics = len(channels)
    if n_mics < 2:
        return None

    # Use only available mics (might be 2, 3, or 4)
    available_pairs = [(i, j) for i, j in pairs if i < n_mics and j < n_mics]
    if not available_pairs:
        return None

    # Step 1: Bandpass filter all channels
    filtered = []
    for ch in channels:
        f = _bandpass_filter(ch, sample_rate)
        filtered.append(f)

    # Check for bird sound presence
    rms_values = [compute_rms(f) for f in filtered]
    if max(rms_values) < NOISE_THRESHOLD:
        return None

    # Step 2: Compute GCC-PHAT TDOA for each mic pair
    tdoa_measurements = []
    for i, j in available_pairs:
        pair_dist = float(np.linalg.norm(mic_positions[i] - mic_positions[j]))
        max_delay = pair_dist / SPEED_OF_SOUND

        result = _gcc_phat_delay(filtered[i], filtered[j], sample_rate, max_delay * 1.5)
        if result is None:
            continue

        tdoa_measurements.append({
            "pair": (i, j),
            "tdoa": result["delay_seconds"],
            "strength": result["strength"],
            "pair_dist_mm": round(pair_dist * 1000, 1),
            "mic_i": mic_config["mics"][i]["label"] if i < len(mic_config["mics"]) else f"mic{i}",
            "mic_j": mic_config["mics"][j]["label"] if j < len(mic_config["mics"]) else f"mic{j}",
        })

    if len(tdoa_measurements) < 1:
        return None

    # Step 3: Solve for direction via least-squares
    # For a plane wave from direction d (unit vector), the expected TDOA
    # between mics i and j is: tau_ij = (pos_i - pos_j) . d / c
    #
    # We parameterize d as (azimuth_phone, elevation_phone) and minimize
    # the sum of squared residuals between measured and expected TDOAs.

    def _tdoa_residuals(params):
        az, el = params
        az_rad = math.radians(az)
        el_rad = math.radians(el)
        # Direction unit vector in phone coordinates
        # Phone frame: X=right, Y=up, Z=toward user (out of screen)
        d = np.array([
            math.cos(el_rad) * math.sin(az_rad),  # right/left
            math.cos(el_rad) * math.cos(az_rad),  # up/down (along phone length)
            math.sin(el_rad),                       # toward/away
        ])
        residuals = []
        for m in tdoa_measurements:
            i, j = m["pair"]
            delta_pos = mic_positions[i] - mic_positions[j]
            expected_tdoa = np.dot(delta_pos, d) / SPEED_OF_SOUND
            residuals.append((m["tdoa"] - expected_tdoa) * m["strength"])
        return residuals

    def _cost(params):
        r = _tdoa_residuals(params)
        return sum(x * x for x in r)

    # Grid search for initial estimate (avoid local minima)
    best_cost = float("inf")
    best_params = (0.0, 0.0)
    for az_init in range(-180, 181, 30):
        for el_init in range(-60, 61, 30):
            c = _cost((az_init, el_init))
            if c < best_cost:
                best_cost = c
                best_params = (az_init, el_init)

    # Refine with optimization
    result = minimize(
        _cost, best_params,
        method="Nelder-Mead",
        options={"xatol": 0.5, "fatol": 1e-10, "maxiter": 200},
    )
    az_phone, el_phone = result.x
    residual = float(result.fun)

    # Normalize azimuth to -180..180
    while az_phone > 180:
        az_phone -= 360
    while az_phone < -180:
        az_phone += 360
    el_phone = np.clip(el_phone, -90, 90)

    # Step 4: Transform to world coordinates
    world_heading, world_elevation = _phone_to_world_direction(
        az_phone, el_phone, device_heading, device_pitch, device_roll
    )

    # Confidence from average TDOA strength and residual quality
    avg_strength = np.mean([m["strength"] for m in tdoa_measurements])
    n_good = sum(1 for m in tdoa_measurements if m["strength"] > 0.1)
    pair_coverage = n_good / max(len(available_pairs), 1)
    residual_quality = max(0, 1.0 - residual * 1e6)  # lower residual = better
    confidence = (avg_strength * 0.4 + pair_coverage * 0.3 + residual_quality * 0.3)
    confidence = min(1.0, max(0.0, confidence))

    return {
        "heading": round(float(world_heading), 1),
        "elevation": round(float(world_elevation), 1),
        "confidence": round(confidence, 3),
        "azimuth_phone": round(float(az_phone), 1),
        "elevation_phone": round(float(el_phone), 1),
        "tdoa_pairs": tdoa_measurements,
        "residual": round(residual, 8),
        "n_pairs_used": len(tdoa_measurements),
    }


def estimate_direction_4mic_continuous(
    chunks: list[dict],
    sample_rate: int = 44100,
) -> dict | None:
    """
    Continuous 4-mic direction estimation averaged across multiple time chunks.

    Each chunk: {"channels": [ch0, ch1, ch2, ch3], "heading": float, "pitch": float, "roll": float}
    """
    if not chunks:
        return None

    estimates = []
    for chunk in chunks:
        chs = chunk.get("channels", [])
        if len(chs) < 2:
            continue
        est = estimate_direction_4mic(
            channels=chs,
            sample_rate=sample_rate,
            device_heading=chunk.get("heading", 0),
            device_pitch=chunk.get("pitch", 90),
            device_roll=chunk.get("roll", 0),
        )
        if est and est["confidence"] > 0.05:
            estimates.append(est)

    if not estimates:
        return None

    # Circular mean for heading
    sin_sum = sum(math.sin(math.radians(e["heading"])) for e in estimates)
    cos_sum = sum(math.cos(math.radians(e["heading"])) for e in estimates)
    avg_heading = math.degrees(math.atan2(sin_sum, cos_sum)) % 360

    avg_elevation = np.mean([e["elevation"] for e in estimates])
    avg_confidence = np.mean([e["confidence"] for e in estimates])

    # Consistency bonus
    headings = [e["heading"] for e in estimates]
    if len(headings) > 1:
        # Circular variance
        sin_var = np.var([math.sin(math.radians(h)) for h in headings])
        cos_var = np.var([math.cos(math.radians(h)) for h in headings])
        circ_var = sin_var + cos_var
        consistency = max(0, 0.15 - circ_var)
        avg_confidence = min(1.0, avg_confidence + consistency)

    return {
        "heading": round(float(avg_heading), 1),
        "elevation": round(float(avg_elevation), 1),
        "confidence": round(float(avg_confidence), 3),
        "n_measurements": len(estimates),
        "n_pairs_used": estimates[-1].get("n_pairs_used", 0),
    }


# ─── Legacy stereo fallback (2-channel) ────────────────────────────────────────

def estimate_direction_stereo(
    channel_1: np.ndarray, channel_2: np.ndarray,
    sample_rate: int = 44100,
    device_heading: float = 0.0,
    device_pitch: float = 90.0,
    device_roll: float = 0.0,
    mic_spacing: float = 0.152,
) -> dict | None:
    """Fallback 2-mic stereo direction (bottom-left to front-top)."""
    return estimate_direction_4mic(
        channels=[channel_1, channel_2],
        sample_rate=sample_rate,
        device_heading=device_heading,
        device_pitch=device_pitch,
        device_roll=device_roll,
    )


def estimate_direction_stereo_continuous(
    chunks: list[dict], sample_rate: int = 44100, phone_model: str = "default",
) -> dict | None:
    """Fallback: wrap 2-channel chunks into 4-mic continuous estimator."""
    converted = []
    for c in chunks:
        chs = []
        if "ch1" in c:
            chs.append(c["ch1"])
        if "ch2" in c:
            chs.append(c["ch2"])
        if len(chs) >= 2:
            converted.append({
                "channels": chs,
                "heading": c.get("heading", 0),
                "pitch": c.get("pitch", 90),
                "roll": c.get("roll", 0),
            })
    return estimate_direction_4mic_continuous(converted, sample_rate)


# ─── Fallback: Single-mic amplitude scanning ───────────────────────────────────

def estimate_direction(chunks: list[dict], sample_rate: int = 44100) -> dict | None:
    """Fallback: requires user rotation. Use 4-mic method instead when available."""
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
    sources: list[dict], chunks: list[dict], sample_rate: int = 44100,
) -> list[dict]:
    """Per-source direction estimation using frequency-band isolation."""
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
        mean_rms = np.mean([a["rms"] for a in amplitudes])
        confidence = min(1.0, (max_entry["rms"] / (mean_rms + 1e-10) - 1.0) / 2.0) if mean_rms > 0 else 0.0
        results.append({
            "source_idx": idx, "heading": max_entry["heading"],
            "confidence": round(confidence, 3),
            "dominant_freq": source.get("dominant_freq", 0),
            "freq_range": source.get("freq_range", (0, 0)),
            "energy": source.get("energy", 0),
        })
    return results
