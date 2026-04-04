"""Multi-device acoustic localization using Time Difference of Arrival (TDOA).

This module implements passive acoustic triangulation when multiple phones
are used as a distributed microphone array. Each phone captures audio and
reports its GPS position + timestamp. By computing the time delay between
when the same sound arrives at different phones, we can triangulate the
bird's position in 2D space.

Science: Passive Acoustic Localization / TDOA
- Same principle as GPS (but in reverse: known receiver positions, unknown source)
- Cross-correlation between audio from different devices gives time delay
- Each pair of devices defines a hyperbola of possible source locations
- 3+ devices = intersection of hyperbolas = source position
- Also known as multilateration

Inspired by how owl ears work: asymmetric ear placement creates
time-of-arrival differences that the brain uses to pinpoint prey in 3D.
"""

import math
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import correlate, butter, sosfilt

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = 343.0  # m/s at ~20°C
BIRD_FREQ_LOW = 800
BIRD_FREQ_HIGH = 10000


@dataclass
class DeviceInfo:
    """A device participating in the multi-device array."""
    device_id: str
    latitude: float = 0.0
    longitude: float = 0.0
    x: float = 0.0   # meters from reference point (computed)
    y: float = 0.0   # meters from reference point (computed)
    heading: float = 0.0
    last_audio: Optional[np.ndarray] = field(default=None, repr=False)
    last_timestamp: float = 0.0
    sample_rate: int = 44100


@dataclass
class LocalizationResult:
    """Result of acoustic triangulation."""
    x: float              # meters from reference device
    y: float              # meters from reference device
    latitude: float       # GPS latitude of estimated source
    longitude: float      # GPS longitude of estimated source
    bearing: float        # bearing from reference device in degrees
    distance: float       # distance from reference device in meters
    confidence: float     # 0-1 confidence score
    tdoa_pairs: list      # raw TDOA measurements for debugging


class DeviceRoom:
    """
    Manages a group of devices for multi-device acoustic triangulation.

    Devices join a "room" with a shared room_id. When all devices have
    submitted audio for the same time window, TDOA is computed.
    """

    def __init__(self, room_id: str = None):
        self.room_id = room_id or str(uuid.uuid4())[:8]
        self.devices: dict[str, DeviceInfo] = {}
        self.reference_device: Optional[str] = None

    def add_device(self, device_id: str, latitude: float, longitude: float) -> str:
        """Register a device. First device becomes the reference point."""
        self.devices[device_id] = DeviceInfo(
            device_id=device_id,
            latitude=latitude,
            longitude=longitude,
        )

        if self.reference_device is None:
            self.reference_device = device_id

        # Recompute all positions relative to reference
        self._update_positions()

        logger.info("Device %s joined room %s (%d devices)", device_id, self.room_id, len(self.devices))
        return self.room_id

    def remove_device(self, device_id: str):
        """Remove a device from the room."""
        self.devices.pop(device_id, None)
        if self.reference_device == device_id:
            self.reference_device = next(iter(self.devices), None)
            self._update_positions()

    def submit_audio(self, device_id: str, audio: np.ndarray, timestamp: float, sample_rate: int = 44100):
        """Submit an audio chunk from a device."""
        if device_id not in self.devices:
            return
        dev = self.devices[device_id]
        dev.last_audio = audio
        dev.last_timestamp = timestamp
        dev.sample_rate = sample_rate

    def can_localize(self) -> bool:
        """Check if we have enough data from enough devices to triangulate."""
        devices_with_audio = [d for d in self.devices.values() if d.last_audio is not None]
        return len(devices_with_audio) >= 2

    def localize(self) -> Optional[LocalizationResult]:
        """
        Perform TDOA-based localization using audio from all devices.

        Requires at least 2 devices (gives a bearing), 3+ gives a position.
        """
        active_devices = [d for d in self.devices.values() if d.last_audio is not None]

        if len(active_devices) < 2:
            return None

        ref = self.devices.get(self.reference_device)
        if ref is None or ref.last_audio is None:
            ref = active_devices[0]

        # Compute TDOA for each device pair (relative to reference)
        tdoa_pairs = []
        for dev in active_devices:
            if dev.device_id == ref.device_id:
                continue

            tdoa = _compute_tdoa(
                ref.last_audio, dev.last_audio,
                ref.sample_rate, dev.sample_rate,
            )

            if tdoa is not None:
                tdoa_pairs.append({
                    "device_id": dev.device_id,
                    "tdoa_seconds": tdoa["delay"],
                    "correlation_strength": tdoa["strength"],
                    "device_x": dev.x,
                    "device_y": dev.y,
                    "ref_x": ref.x,
                    "ref_y": ref.y,
                })

        if not tdoa_pairs:
            return None

        # With 2 devices: compute bearing from TDOA
        # With 3+ devices: triangulate position
        if len(tdoa_pairs) == 1:
            result = _bearing_from_single_tdoa(ref, tdoa_pairs[0])
        else:
            result = _triangulate_from_tdoa(ref, tdoa_pairs)

        if result is None:
            return None

        result.tdoa_pairs = tdoa_pairs
        return result

    def get_device_count(self) -> int:
        return len(self.devices)

    def get_device_positions(self) -> list[dict]:
        """Get all device positions for UI display."""
        return [
            {
                "device_id": d.device_id,
                "x": d.x,
                "y": d.y,
                "latitude": d.latitude,
                "longitude": d.longitude,
                "is_reference": d.device_id == self.reference_device,
            }
            for d in self.devices.values()
        ]

    def _update_positions(self):
        """Convert GPS coordinates to local XY meters relative to reference device."""
        if not self.reference_device or self.reference_device not in self.devices:
            return

        ref = self.devices[self.reference_device]
        ref.x = 0.0
        ref.y = 0.0

        for dev in self.devices.values():
            if dev.device_id == ref.device_id:
                continue
            dev.x, dev.y = _gps_to_local_xy(
                ref.latitude, ref.longitude,
                dev.latitude, dev.longitude,
            )


def _bandpass_bird(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Bandpass filter for bird vocalization frequencies."""
    nyquist = sample_rate / 2
    low = BIRD_FREQ_LOW / nyquist
    high = min(BIRD_FREQ_HIGH / nyquist, 0.99)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio)


def _compute_tdoa(
    audio_ref: np.ndarray,
    audio_dev: np.ndarray,
    sr_ref: int,
    sr_dev: int,
) -> Optional[dict]:
    """
    Compute Time Difference of Arrival between two audio signals
    using generalized cross-correlation (GCC-PHAT).

    GCC-PHAT (Phase Transform) whitens the cross-correlation to produce
    a sharper peak, making it more robust to reverb and noise.

    Returns:
        {"delay": float (seconds), "strength": float (0-1)} or None
    """
    # Ensure same sample rate
    sample_rate = sr_ref
    if sr_ref != sr_dev:
        # Simple resampling - in production use scipy.signal.resample
        ratio = sr_ref / sr_dev
        n_samples = int(len(audio_dev) * ratio)
        audio_dev = np.interp(
            np.linspace(0, len(audio_dev) - 1, n_samples),
            np.arange(len(audio_dev)),
            audio_dev,
        )

    # Bandpass filter both signals for bird frequencies
    ref_filtered = _bandpass_bird(audio_ref, sample_rate)
    dev_filtered = _bandpass_bird(audio_dev, sample_rate)

    # Ensure same length (trim to shorter)
    min_len = min(len(ref_filtered), len(dev_filtered))
    if min_len < 256:
        return None
    ref_filtered = ref_filtered[:min_len]
    dev_filtered = dev_filtered[:min_len]

    # GCC-PHAT cross-correlation
    n_fft = 2 ** int(np.ceil(np.log2(2 * min_len - 1)))
    REF = np.fft.rfft(ref_filtered, n=n_fft)
    DEV = np.fft.rfft(dev_filtered, n=n_fft)

    cross_spectrum = REF * np.conj(DEV)
    magnitude = np.abs(cross_spectrum)
    magnitude[magnitude < 1e-10] = 1e-10  # Avoid division by zero

    # PHAT weighting (phase transform)
    gcc_phat = np.fft.irfft(cross_spectrum / magnitude, n=n_fft)

    # Find the peak within a reasonable delay range
    # Max delay = max distance between devices / speed of sound
    max_delay_samples = int(0.1 * sample_rate)  # 100ms max (34m apart)
    max_delay_samples = min(max_delay_samples, n_fft // 2)

    # Search in the valid range (wrapping around FFT output)
    gcc_valid = np.concatenate([
        gcc_phat[-max_delay_samples:],
        gcc_phat[:max_delay_samples + 1],
    ])

    peak_idx = np.argmax(np.abs(gcc_valid))
    peak_value = float(np.abs(gcc_valid[peak_idx]))

    # Convert index to delay in seconds
    delay_samples = peak_idx - max_delay_samples
    delay_seconds = delay_samples / sample_rate

    # Strength: peak relative to average (higher = more confident)
    avg_value = float(np.mean(np.abs(gcc_valid)))
    strength = min(1.0, (peak_value / (avg_value + 1e-10) - 1) / 10)

    if strength < 0.05:
        return None

    return {
        "delay": round(delay_seconds, 6),
        "strength": round(strength, 4),
    }


def _bearing_from_single_tdoa(ref: DeviceInfo, pair: dict) -> Optional[LocalizationResult]:
    """
    With only 2 devices, compute the bearing to the source.
    The source lies on a hyperbola, but we can estimate the bearing.
    """
    dx = pair["device_x"] - pair["ref_x"]
    dy = pair["device_y"] - pair["ref_y"]
    baseline_dist = math.sqrt(dx**2 + dy**2)

    if baseline_dist < 0.1:
        return None

    # Baseline angle (from ref to device)
    baseline_angle = math.atan2(dy, dx)

    # TDOA gives us the path length difference
    path_diff = pair["tdoa_seconds"] * SPEED_OF_SOUND

    # The source direction bisects the baseline, offset by TDOA
    # If TDOA > 0, sound arrived at ref first (source is closer to ref)
    # cos(theta) = path_diff / baseline_dist (bounded to [-1, 1])
    cos_theta = np.clip(path_diff / baseline_dist, -1.0, 1.0)
    theta = math.acos(cos_theta)

    # Two possible bearings (above/below baseline) - report both
    bearing1 = math.degrees(baseline_angle + theta / 2) % 360
    bearing2 = math.degrees(baseline_angle - theta / 2) % 360

    # Use the bearing that's perpendicular to baseline as primary
    bearing = bearing1

    return LocalizationResult(
        x=0, y=0,  # Can't determine position with 2 devices
        latitude=ref.latitude,
        longitude=ref.longitude,
        bearing=round(bearing, 1),
        distance=-1,  # Unknown with 2 devices
        confidence=round(pair["correlation_strength"], 3),
        tdoa_pairs=[],
    )


def _triangulate_from_tdoa(ref: DeviceInfo, pairs: list[dict]) -> Optional[LocalizationResult]:
    """
    With 3+ devices, triangulate the source position using least-squares
    optimization on the TDOA hyperbolic equations.

    Each TDOA measurement defines a hyperbola:
    |d_i - d_ref| = c * tdoa_i

    where d_i = distance from source to device i, c = speed of sound.
    We solve for (x, y) that minimizes the residuals.
    """
    from scipy.optimize import least_squares

    def residuals(pos, pairs_data):
        x, y = pos
        res = []
        for p in pairs_data:
            d_ref = math.sqrt((x - p["ref_x"])**2 + (y - p["ref_y"])**2)
            d_dev = math.sqrt((x - p["device_x"])**2 + (y - p["device_y"])**2)
            expected_tdoa = (d_dev - d_ref) / SPEED_OF_SOUND
            res.append(expected_tdoa - p["tdoa_seconds"])
        return res

    # Initial guess: centroid of devices, offset in the average TDOA direction
    x0 = np.mean([ref.x] + [p["device_x"] for p in pairs])
    y0 = np.mean([ref.y] + [p["device_y"] for p in pairs])

    try:
        result = least_squares(
            residuals, [x0, y0], args=(pairs,),
            bounds=([-500, -500], [500, 500]),  # 500m search radius
            max_nfev=100,
        )
    except Exception as e:
        logger.warning("Triangulation optimization failed: %s", e)
        return None

    src_x, src_y = result.x
    residual_norm = float(np.linalg.norm(result.fun))

    # Convert back to GPS
    src_lat, src_lon = _local_xy_to_gps(ref.latitude, ref.longitude, src_x, src_y)

    # Distance and bearing from reference device
    distance = math.sqrt(src_x**2 + src_y**2)
    bearing = math.degrees(math.atan2(src_x, src_y)) % 360

    # Confidence based on residual quality and correlation strengths
    avg_strength = np.mean([p["correlation_strength"] for p in pairs])
    residual_confidence = max(0, 1 - residual_norm * 10)
    confidence = (avg_strength + residual_confidence) / 2

    return LocalizationResult(
        x=round(src_x, 2),
        y=round(src_y, 2),
        latitude=round(src_lat, 6),
        longitude=round(src_lon, 6),
        bearing=round(bearing, 1),
        distance=round(distance, 1),
        confidence=round(min(1.0, confidence), 3),
        tdoa_pairs=[],
    )


def _gps_to_local_xy(ref_lat: float, ref_lon: float, lat: float, lon: float) -> tuple[float, float]:
    """Convert GPS delta to local XY in meters (flat earth approximation for short distances)."""
    lat_rad = math.radians(ref_lat)
    x = (lon - ref_lon) * math.cos(lat_rad) * 111320  # meters per degree longitude
    y = (lat - ref_lat) * 110540  # meters per degree latitude
    return (round(x, 2), round(y, 2))


def _local_xy_to_gps(ref_lat: float, ref_lon: float, x: float, y: float) -> tuple[float, float]:
    """Convert local XY meters back to GPS coordinates."""
    lat_rad = math.radians(ref_lat)
    lon = ref_lon + x / (math.cos(lat_rad) * 111320)
    lat = ref_lat + y / 110540
    return (lat, lon)
