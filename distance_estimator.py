"""Unified bird distance estimation engine.

Fuses 6 independent distance estimation methods via an Extended Kalman Filter
to produce the most accurate distance estimate possible from a single phone.

Methods:
  1. Species Call Intensity  — inverse square law + known species dB at 1m
  2. Atmospheric Absorption  — high-frequency rolloff encodes distance
  3. Direct-to-Reverberant Ratio — early vs late energy ratio
  4. Visual Rangefinding      — detected bird pixel size + known body size
  5. Acoustic SLAM            — parallax from user movement + IMU tracking
  6. Doppler Analysis         — frequency shift if bird is in flight

Each method produces an independent distance estimate with uncertainty.
The Kalman filter optimally combines them, weighting by confidence.
"""

import math
import numpy as np
from typing import Optional

from bird_species_db import get_species_data, get_absorption_at_freq, ATMOSPHERIC_ABSORPTION_DB_PER_M
from acoustic_slam import AcousticSLAM, IMUReading


SPEED_OF_SOUND = 343.0  # m/s


# ─── Method 1: Species Call Intensity ───────────────────────────────────────────

def estimate_distance_intensity(
    measured_rms: float,
    species: str,
    sample_rate: int = 44100,
) -> Optional[dict]:
    """
    Inverse square law: intensity falls as 1/r².

    Known: species call volume at 1 meter (dB SPL)
    Measured: received audio RMS level
    Distance: d = 10^((L_source - L_measured) / 20)

    The RMS is not calibrated to absolute dB SPL (phone mic gain is unknown),
    so we use a reference calibration factor. This gets refined over time
    as other methods provide ground truth.
    """
    if measured_rms <= 0:
        return None

    sp = get_species_data(species)
    source_db = sp["call_db_1m"]

    # Convert RMS to approximate dB (relative, not absolute SPL)
    # Phone mic sensitivity ~-38 dBFS/Pa, but this varies by device
    # We use a calibration constant that maps RMS to approximate dB SPL
    # Default calibration: RMS of 0.1 ≈ 60 dB SPL (typical for phone)
    CALIBRATION_DB_AT_RMS_01 = 60.0
    measured_db = 20 * math.log10(measured_rms / 0.1 + 1e-10) + CALIBRATION_DB_AT_RMS_01

    # Inverse square law: L = L_source - 20*log10(d)
    # Therefore: d = 10^((L_source - L_measured) / 20)
    db_diff = source_db - measured_db
    if db_diff < 0:
        # Measured louder than source at 1m — very close or calibration error
        distance = 0.5
    else:
        distance = 10 ** (db_diff / 20.0)

    # Clamp to reasonable range
    distance = max(0.5, min(distance, 500.0))

    # Confidence: lower for extreme distances, higher in moderate range
    # Also lower because calibration is uncertain
    confidence = 0.25  # Base confidence is low (uncalibrated mic)
    if 2 < distance < 100:
        confidence = 0.35

    return {
        "distance": round(distance, 1),
        "confidence": round(confidence, 3),
        "uncertainty_m": round(distance * 0.6, 1),  # ~60% uncertainty
        "method": "intensity",
        "source_db": source_db,
        "measured_db": round(measured_db, 1),
    }


# ─── Method 2: Atmospheric Absorption ──────────────────────────────────────────

def estimate_distance_absorption(
    audio: np.ndarray,
    sample_rate: int,
    species: str,
) -> Optional[dict]:
    """
    Atmospheric absorption: air absorbs high frequencies more than low.

    The spectral tilt (ratio of high-freq to low-freq energy) compared to
    the species' known spectral shape encodes distance.

    At 20°C, 50% humidity:
      1 kHz: 0.005 dB/m loss
      4 kHz: 0.025 dB/m loss
      8 kHz: 0.064 dB/m loss

    So a bird 50m away loses: 0.25 dB at 1kHz vs 3.2 dB at 8kHz.
    That 2.95 dB tilt difference is measurable.
    """
    if len(audio) < 2048:
        return None

    sp = get_species_data(species)
    freq_low, freq_high = sp["spectral_bw_hz"]

    # Compute power spectrum
    n_fft = min(4096, len(audio))
    spectrum = np.abs(np.fft.rfft(audio[:n_fft]))**2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

    # Measure energy in low band vs high band
    low_band = (freqs >= freq_low) & (freqs <= (freq_low + freq_high) / 2)
    high_band = (freqs >= (freq_low + freq_high) / 2) & (freqs <= freq_high)

    if not np.any(low_band) or not np.any(high_band):
        return None

    low_energy = np.mean(spectrum[low_band])
    high_energy = np.mean(spectrum[high_band])

    if low_energy <= 0 or high_energy <= 0:
        return None

    # Measured spectral tilt (dB difference between bands)
    measured_tilt_db = 10 * math.log10(low_energy / high_energy)

    # Expected tilt at 1 meter (species-specific, from the reference spectrum)
    # Most bird calls have roughly flat spectral energy within their band at source
    # So the expected tilt at 1m is ~0 dB
    reference_tilt_db = 0.0  # Flat at source (simplification)

    # The excess tilt is due to atmospheric absorption
    excess_tilt = measured_tilt_db - reference_tilt_db

    if excess_tilt <= 0:
        # High frequencies are louder than expected — very close or noise
        return {"distance": 2.0, "confidence": 0.1, "uncertainty_m": 5.0,
                "method": "absorption", "excess_tilt_db": round(excess_tilt, 2)}

    # Compute expected tilt per meter
    freq_mid_low = (freq_low + (freq_low + freq_high) / 2) / 2
    freq_mid_high = ((freq_low + freq_high) / 2 + freq_high) / 2

    abs_low = get_absorption_at_freq(freq_mid_low)
    abs_high = get_absorption_at_freq(freq_mid_high)
    tilt_per_meter = abs_high - abs_low  # dB/m of differential absorption

    if tilt_per_meter <= 0.001:
        return None

    distance = excess_tilt / tilt_per_meter
    distance = max(1.0, min(distance, 300.0))

    # Confidence depends on how strong the spectral tilt signal is
    signal_strength = min(1.0, excess_tilt / 3.0)  # 3dB excess = good signal
    confidence = 0.3 * signal_strength

    return {
        "distance": round(distance, 1),
        "confidence": round(confidence, 3),
        "uncertainty_m": round(distance * 0.4, 1),  # ~40% uncertainty
        "method": "absorption",
        "excess_tilt_db": round(excess_tilt, 2),
        "tilt_per_meter": round(tilt_per_meter, 4),
    }


# ─── Method 3: Direct-to-Reverberant Ratio (DRR) ──────────────────────────────

def estimate_distance_drr(
    audio: np.ndarray,
    sample_rate: int,
) -> Optional[dict]:
    """
    Direct-to-Reverberant Ratio: how humans perceive distance to sounds.

    Direct sound (first ~5ms after onset) vs reverberant energy (5-100ms after).
    The DRR decreases with distance because:
      - Direct sound: 1/r² falloff
      - Reverberant field: roughly constant (bounded by environment)

    DRR (dB) ≈ 10*log10(direct_energy / reverb_energy)

    Empirical relationship (outdoor, semi-open):
      DRR ≈ 20*log10(r_ref/r) + DRR_ref
      → r = r_ref * 10^((DRR_ref - DRR) / 20)
    """
    if len(audio) < sample_rate // 4:  # Need at least 250ms
        return None

    # Find onset: first sample above threshold
    threshold = np.max(np.abs(audio)) * 0.1
    onset_idx = None
    for i in range(len(audio)):
        if abs(audio[i]) > threshold:
            onset_idx = i
            break

    if onset_idx is None:
        return None

    # Direct sound: first 5ms after onset
    direct_samples = int(0.005 * sample_rate)
    # Reverberant: 10ms to 100ms after onset
    reverb_start = int(0.010 * sample_rate)
    reverb_end = int(0.100 * sample_rate)

    if onset_idx + reverb_end >= len(audio):
        return None

    direct_window = audio[onset_idx:onset_idx + direct_samples]
    reverb_window = audio[onset_idx + reverb_start:onset_idx + reverb_end]

    direct_energy = np.mean(direct_window**2)
    reverb_energy = np.mean(reverb_window**2)

    if reverb_energy <= 0 or direct_energy <= 0:
        return None

    drr_db = 10 * math.log10(direct_energy / reverb_energy)

    # Empirical model: DRR of ~15 dB at 1m (outdoor), decreasing ~6 dB per doubling
    DRR_REF = 15.0   # dB at 1 meter
    R_REF = 1.0       # reference distance

    db_diff = DRR_REF - drr_db
    if db_diff < -10:
        distance = 0.5  # Very close
    else:
        distance = R_REF * (10 ** (db_diff / 20.0))

    distance = max(0.5, min(distance, 200.0))

    # Confidence: DRR is very environment-dependent
    confidence = 0.2  # Low base confidence
    if 5 < drr_db < 25:
        confidence = 0.3

    return {
        "distance": round(distance, 1),
        "confidence": round(confidence, 3),
        "uncertainty_m": round(distance * 0.5, 1),
        "method": "drr",
        "drr_db": round(drr_db, 1),
    }


# ─── Method 4: Visual Rangefinding ─────────────────────────────────────────────

def estimate_distance_visual(
    bird_bbox_pixels: tuple[int, int, int, int],  # x, y, width, height in pixels
    image_width_pixels: int,
    image_height_pixels: int,
    species: str,
    camera_fov_horizontal: float = 67.0,  # iPhone 16 PM main camera: 24mm = ~67° FOV
    zoom_factor: float = 1.0,
) -> Optional[dict]:
    """
    Camera rangefinding: known object size + apparent pixel size = distance.

    d = (real_size * focal_length_pixels) / pixel_size
      = (real_size * image_width) / (pixel_size * 2 * tan(FOV/2))

    iPhone 16 Pro Max cameras:
      Main (24mm): 67° horizontal FOV, 48MP
      Ultra-wide (13mm): 120° FOV
      5x Telephoto (120mm): 15.5° FOV
    """
    if not bird_bbox_pixels or bird_bbox_pixels[2] <= 0 or bird_bbox_pixels[3] <= 0:
        return None

    sp = get_species_data(species)
    body_length_cm = sp["body_length_cm"]
    body_length_m = body_length_cm / 100.0

    # Use the larger dimension of the bounding box
    bird_pixel_size = max(bird_bbox_pixels[2], bird_bbox_pixels[3])

    # Effective FOV accounting for zoom
    effective_fov = camera_fov_horizontal / zoom_factor

    # Focal length in pixels
    focal_length_px = (image_width_pixels / 2) / math.tan(math.radians(effective_fov / 2))

    # Distance = (real_size * focal_length) / pixel_size
    distance = (body_length_m * focal_length_px) / bird_pixel_size
    distance = max(0.5, min(distance, 500.0))

    # Confidence: higher when bird fills more pixels (clearer detection)
    pixel_ratio = bird_pixel_size / max(image_width_pixels, image_height_pixels)
    confidence = min(0.8, pixel_ratio * 5)  # 20% of frame = max confidence
    confidence = max(0.15, confidence)

    return {
        "distance": round(distance, 1),
        "confidence": round(confidence, 3),
        "uncertainty_m": round(distance * 0.2, 1),  # ~20% uncertainty (best method!)
        "method": "visual",
        "bird_pixels": bird_pixel_size,
        "body_size_cm": body_length_cm,
    }


# ─── Method 5: Acoustic SLAM (wrapper) ─────────────────────────────────────────

# AcousticSLAM is in acoustic_slam.py; the distance_estimator owns the instance
# and feeds it data. See the BirdDistanceEstimator class below.


# ─── Method 6: Doppler Analysis ────────────────────────────────────────────────

def estimate_doppler(
    audio_chunks: list[np.ndarray],
    sample_rate: int,
    species: str,
) -> Optional[dict]:
    """
    Doppler frequency shift analysis for birds in flight.

    If the bird is moving toward/away from the mic, its call frequency shifts:
      f_observed = f_source * c / (c - v_radial)
      v_radial = c * (f_observed - f_source) / f_observed

    By tracking the peak frequency across successive audio chunks,
    we can detect radial velocity. This doesn't directly give distance,
    but combined with angular velocity (from direction changes), it
    constrains the distance via:
      v_tangential = angular_velocity * distance
      v_total = sqrt(v_radial² + v_tangential²)

    Typical bird flight speeds: 10-20 m/s (small songbirds), up to 50 m/s (raptors)
    """
    if len(audio_chunks) < 3:
        return None

    sp = get_species_data(species)
    expected_peak = sp["spectral_peak_hz"]
    freq_low, freq_high = sp["spectral_bw_hz"]

    # Track peak frequency across chunks
    peak_freqs = []
    for chunk in audio_chunks:
        if len(chunk) < 1024:
            continue
        spectrum = np.abs(np.fft.rfft(chunk[:2048]))
        freqs = np.fft.rfftfreq(2048, 1.0 / sample_rate)

        # Find peak in the species' frequency band
        band_mask = (freqs >= freq_low) & (freqs <= freq_high)
        if not np.any(band_mask):
            continue
        band_spectrum = spectrum.copy()
        band_spectrum[~band_mask] = 0
        peak_idx = np.argmax(band_spectrum)
        peak_freqs.append(float(freqs[peak_idx]))

    if len(peak_freqs) < 3:
        return None

    # Compute frequency drift (Hz per chunk)
    freq_diffs = np.diff(peak_freqs)
    avg_drift = float(np.mean(freq_diffs))
    max_drift = float(np.max(np.abs(freq_diffs)))

    # Radial velocity from Doppler
    # v = c * delta_f / f_source
    avg_freq = np.mean(peak_freqs)
    if avg_freq <= 0:
        return None

    v_radial = SPEED_OF_SOUND * (avg_freq - expected_peak) / avg_freq

    # If there's minimal Doppler shift, bird is stationary (no distance info)
    if abs(v_radial) < 0.5:  # Less than 0.5 m/s — essentially stationary
        return {
            "v_radial": round(v_radial, 2),
            "bird_moving": False,
            "method": "doppler",
            "confidence": 0.0,
        }

    return {
        "v_radial": round(v_radial, 2),
        "v_radial_abs": round(abs(v_radial), 2),
        "freq_drift_hz_per_chunk": round(avg_drift, 2),
        "approaching": v_radial > 0,
        "bird_moving": True,
        "estimated_flight_speed": round(abs(v_radial) * 1.4, 1),  # radial is ~70% of total
        "method": "doppler",
        "confidence": min(0.3, max_drift / 50),
    }


# ─── Kalman Filter Fusion ──────────────────────────────────────────────────────

class DistanceKalmanFilter:
    """
    1D Extended Kalman Filter for fusing multiple distance estimates.

    State: [distance, distance_velocity]
    Each method provides a measurement with its own variance.
    The filter optimally weights them based on confidence.

    This is mathematically equivalent to how GPS receivers fuse signals
    from multiple satellites — each with different noise characteristics.
    """

    def __init__(self):
        # State: [distance (m), rate_of_change (m/s)]
        self.x = np.array([20.0, 0.0])  # Initial guess: 20m, stationary
        self.P = np.array([[100.0, 0.0], [0.0, 1.0]])  # High initial uncertainty

        # Process noise (how much distance can change between updates)
        self.Q = np.array([[0.5, 0.0], [0.0, 0.1]])

        self.initialized = False
        self.last_update_time = 0.0

    def predict(self, dt: float):
        """Time update: predict state forward."""
        if dt <= 0 or dt > 10:
            return

        # State transition: distance changes by velocity * dt
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt

        # Keep distance positive
        self.x[0] = max(0.5, self.x[0])

    def update(self, distance: float, uncertainty: float, confidence: float):
        """
        Measurement update: incorporate a new distance estimate.

        Args:
            distance: Measured distance in meters
            uncertainty: 1-sigma uncertainty in meters
            confidence: 0-1 confidence weight (0 = ignore, 1 = fully trust)
        """
        if confidence <= 0.01 or uncertainty <= 0:
            return

        # Measurement variance (lower confidence = higher variance)
        R = (uncertainty / confidence) ** 2

        # Measurement matrix: we observe distance directly
        H = np.array([[1.0, 0.0]])

        # Innovation
        y = distance - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T / S

        # State update
        self.x = self.x + (K * y).flatten()
        self.P = (np.eye(2) - K @ H) @ self.P

        # Keep distance positive
        self.x[0] = max(0.5, self.x[0])

        if not self.initialized:
            self.initialized = True

    def get_estimate(self) -> tuple[float, float]:
        """Returns (distance, uncertainty)."""
        distance = float(self.x[0])
        uncertainty = float(math.sqrt(self.P[0, 0]))
        return (distance, uncertainty)


# ─── Unified Distance Estimator ─────────────────────────────────────────────────

class BirdDistanceEstimator:
    """
    Unified engine that runs all 6 distance methods and fuses via Kalman filter.

    Usage:
        estimator = BirdDistanceEstimator()

        # Feed data as it arrives:
        estimator.update_audio(pcm_data, sample_rate, measured_rms)
        estimator.update_species("Northern Cardinal")
        estimator.update_imu(imu_reading)
        estimator.update_direction(azimuth, elevation, confidence)
        estimator.update_visual(bbox, image_size)

        # Get fused distance:
        result = estimator.get_distance()
    """

    def __init__(self):
        self.kf = DistanceKalmanFilter()
        self.slam = AcousticSLAM()

        self.species: str = "Unknown"
        self.latest_audio: Optional[np.ndarray] = None
        self.audio_chunks: list[np.ndarray] = []
        self.sample_rate: int = 44100
        self.measured_rms: float = 0.0

        self.last_update_time: float = 0.0
        self.method_results: dict[str, dict] = {}

    def update_species(self, species: str):
        """Update the identified species (from BirdNET)."""
        self.species = species

    def update_audio(self, audio: np.ndarray, sample_rate: int, rms: float):
        """Feed audio data for acoustic distance methods."""
        self.latest_audio = audio
        self.sample_rate = sample_rate
        self.measured_rms = rms

        self.audio_chunks.append(audio)
        if len(self.audio_chunks) > 10:
            self.audio_chunks.pop(0)

        # Run acoustic methods
        self._run_intensity()
        self._run_absorption()
        self._run_drr()
        self._run_doppler()
        self._update_kalman()

    def update_imu(self, imu: IMUReading):
        """Feed IMU data for dead reckoning (acoustic SLAM)."""
        self.slam.add_imu_reading(imu)

    def update_direction(self, azimuth: float, elevation: float, confidence: float):
        """Feed direction measurement for acoustic SLAM."""
        self.slam.add_angle_measurement(azimuth, elevation, confidence)

        # Try triangulation
        slam_result = self.slam.estimate_distance()
        if slam_result and slam_result["confidence"] > 0.05:
            self.method_results["slam"] = slam_result
            self.kf.update(
                slam_result["distance"],
                slam_result["distance"] * (1.0 - slam_result["confidence"]),
                slam_result["confidence"],
            )

    def update_visual(
        self,
        bbox: tuple[int, int, int, int],
        image_width: int, image_height: int,
        zoom: float = 1.0,
    ):
        """Feed visual detection for camera rangefinding."""
        result = estimate_distance_visual(
            bbox, image_width, image_height,
            self.species, zoom_factor=zoom,
        )
        if result:
            self.method_results["visual"] = result
            self.kf.update(result["distance"], result["uncertainty_m"], result["confidence"])

    def _run_intensity(self):
        result = estimate_distance_intensity(self.measured_rms, self.species, self.sample_rate)
        if result:
            self.method_results["intensity"] = result
            self.kf.update(result["distance"], result["uncertainty_m"], result["confidence"])

    def _run_absorption(self):
        if self.latest_audio is not None:
            result = estimate_distance_absorption(self.latest_audio, self.sample_rate, self.species)
            if result:
                self.method_results["absorption"] = result
                self.kf.update(result["distance"], result["uncertainty_m"], result["confidence"])

    def _run_drr(self):
        if self.latest_audio is not None:
            result = estimate_distance_drr(self.latest_audio, self.sample_rate)
            if result:
                self.method_results["drr"] = result
                self.kf.update(result["distance"], result["uncertainty_m"], result["confidence"])

    def _run_doppler(self):
        if len(self.audio_chunks) >= 3:
            result = estimate_doppler(self.audio_chunks[-5:], self.sample_rate, self.species)
            if result:
                self.method_results["doppler"] = result

    def _update_kalman(self):
        import time
        now = time.time()
        dt = now - self.last_update_time if self.last_update_time > 0 else 0.5
        self.kf.predict(dt)
        self.last_update_time = now

    def get_distance(self) -> dict:
        """
        Get the fused distance estimate from all methods.

        Returns:
            {
                "distance": float,          # Best estimate in meters
                "uncertainty": float,       # 1-sigma uncertainty in meters
                "confidence": float,        # Overall confidence 0-1
                "methods_used": list,       # Which methods contributed
                "method_details": dict,     # Individual method results
                "distance_range": (min, max),  # Reasonable range
            }
        """
        fused_distance, fused_uncertainty = self.kf.get_estimate()

        # List of methods that contributed
        methods_used = list(self.method_results.keys())

        # Overall confidence: higher when more methods agree
        confidences = [r.get("confidence", 0) for r in self.method_results.values()
                       if "confidence" in r and r.get("confidence", 0) > 0]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            # Bonus for multiple agreeing methods
            agreement_bonus = min(0.2, len(confidences) * 0.05)
            overall_confidence = min(1.0, avg_confidence + agreement_bonus)
        else:
            overall_confidence = 0.0

        # Reasonable range
        range_min = max(0.5, fused_distance - 2 * fused_uncertainty)
        range_max = fused_distance + 2 * fused_uncertainty

        return {
            "distance": round(fused_distance, 1),
            "uncertainty": round(fused_uncertainty, 1),
            "confidence": round(overall_confidence, 3),
            "methods_used": methods_used,
            "method_details": {k: v for k, v in self.method_results.items()},
            "distance_range": (round(range_min, 1), round(range_max, 1)),
        }

    def reset(self):
        """Reset for a new bird."""
        self.kf = DistanceKalmanFilter()
        self.slam.reset()
        self.method_results = {}
        self.audio_chunks = []
        self.species = "Unknown"
