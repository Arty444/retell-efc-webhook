"""Synthetic bird call generator for simulation testing.

Generates realistic-ish bird calls with controllable parameters:
- Species-specific frequency patterns (from bird_species_db)
- Harmonics structure mimicking real bird vocalizations
- Amplitude envelopes (attack/sustain/decay)
- Controllable SNR (noise floor)

Also provides utilities to simulate:
- 4-mic array capture with known time delays (TDOA)
- Distance-based attenuation (inverse square law + atmospheric absorption)
- Doppler shift for moving birds
"""

import sys
import os
import math
import numpy as np

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bird_species_db import get_species_data, get_absorption_at_freq

SPEED_OF_SOUND = 343.0  # m/s


def generate_bird_call(
    species: str = "Northern Cardinal",
    duration_s: float = 1.0,
    sample_rate: int = 44100,
    n_harmonics: int = 3,
    amplitude: float = 0.5,
    noise_level: float = 0.01,
    frequency_modulation_hz: float = 200.0,
    modulation_rate_hz: float = 8.0,
) -> np.ndarray:
    """
    Generate a synthetic bird call.

    Creates a frequency-modulated tone with harmonics and envelope,
    matching the spectral profile of the given species.

    Args:
        species: Bird species name (looked up in bird_species_db)
        duration_s: Call duration in seconds
        sample_rate: Sample rate in Hz
        n_harmonics: Number of harmonics to include
        amplitude: Peak amplitude (0-1)
        noise_level: Background noise RMS (0-1)
        frequency_modulation_hz: FM sweep width in Hz
        modulation_rate_hz: FM modulation rate (warble speed)

    Returns:
        np.ndarray of float32 audio samples
    """
    sp = get_species_data(species)
    f0 = sp["spectral_peak_hz"]
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Frequency modulation (bird calls have characteristic warble/sweep)
    fm = frequency_modulation_hz * np.sin(2 * np.pi * modulation_rate_hz * t)

    # Generate fundamental + harmonics
    signal = np.zeros(n_samples, dtype=np.float64)
    for h in range(1, n_harmonics + 1):
        freq = (f0 + fm) * h
        harmonic_amp = amplitude / (h ** 1.5)  # Harmonics decay as 1/h^1.5
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate
        signal += harmonic_amp * np.sin(phase)

    # Amplitude envelope: attack-sustain-decay
    attack = int(0.05 * n_samples)
    decay = int(0.15 * n_samples)
    sustain = n_samples - attack - decay
    envelope = np.concatenate([
        np.linspace(0, 1, attack),
        np.ones(sustain),
        np.linspace(1, 0, decay),
    ])
    if len(envelope) > n_samples:
        envelope = envelope[:n_samples]
    elif len(envelope) < n_samples:
        envelope = np.pad(envelope, (0, n_samples - len(envelope)))
    signal *= envelope

    # Add noise floor
    noise = np.random.randn(n_samples) * noise_level
    signal += noise

    return signal.astype(np.float32)


def generate_pulsed_call(
    species: str = "Black-capped Chickadee",
    n_pulses: int = 4,
    pulse_duration_s: float = 0.15,
    gap_duration_s: float = 0.1,
    sample_rate: int = 44100,
    amplitude: float = 0.4,
    noise_level: float = 0.01,
) -> np.ndarray:
    """Generate a pulsed/repeated call (e.g., chickadee-dee-dee)."""
    segments = []
    for i in range(n_pulses):
        pulse = generate_bird_call(
            species=species,
            duration_s=pulse_duration_s,
            sample_rate=sample_rate,
            amplitude=amplitude * (0.8 + 0.2 * (i == 0)),  # First pulse louder
            noise_level=0,
        )
        segments.append(pulse)
        if i < n_pulses - 1:
            gap = np.zeros(int(gap_duration_s * sample_rate), dtype=np.float32)
            segments.append(gap)

    signal = np.concatenate(segments)
    noise = np.random.randn(len(signal)).astype(np.float32) * noise_level
    return signal + noise


def simulate_4mic_capture(
    source_signal: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    sample_rate: int = 44100,
    mic_positions_mm: list[tuple] = None,
    noise_per_mic: float = 0.005,
) -> list[np.ndarray]:
    """
    Simulate 4-mic array capture of a far-field source.

    Given a source signal and its true direction, compute the time delays
    for each iPhone 16 Pro Max microphone and produce 4 delayed versions
    of the signal (as each mic would hear it).

    The source is assumed to be in the far field (plane wave approximation),
    so each mic receives the same signal but time-shifted.

    Args:
        source_signal: Mono audio signal
        azimuth_deg: True azimuth in phone frame (0=front, 90=right, etc.)
        elevation_deg: True elevation in phone frame (0=horizon, +90=up)
        sample_rate: Sample rate
        mic_positions_mm: Override mic positions; default = iPhone 16 PM
        noise_per_mic: Independent noise added to each mic

    Returns:
        List of 4 np.ndarray signals (one per mic)
    """
    if mic_positions_mm is None:
        mic_positions_mm = [
            (30.0,   0.0,  0.0),   # Mic 0: bottom-left
            (47.6,   0.0,  0.0),   # Mic 1: bottom-right
            (38.8, 159.0,  0.0),   # Mic 2: front-top
            (20.0, 150.0,  8.25),  # Mic 3: rear-camera
        ]

    mic_positions_m = [np.array(p) / 1000.0 for p in mic_positions_mm]

    # Direction unit vector (phone frame: X=right, Y=up/along phone, Z=toward user)
    # This matches exactly what estimate_direction_4mic uses in its cost function
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    d = np.array([
        math.cos(el) * math.sin(az),
        math.cos(el) * math.cos(az),
        math.sin(el),
    ])

    # For a plane wave arriving from direction d, a mic at position p
    # receives the signal with time-of-arrival proportional to -dot(p, d)/c
    # (negative because the wavefront hits closer mics first).
    #
    # The TDOA solver expects: tau_ij = (pos_i - pos_j) . d / c
    # which equals (dot(pos_i, d) - dot(pos_j, d)) / c
    #
    # So mic with larger dot(pos, d) hears the signal earlier.
    # We apply delay = -dot(pos, d) / c (normalized so earliest mic has delay=0).

    projections = [np.dot(pos, d) for pos in mic_positions_m]
    # Mic with MINIMUM projection is closest to source → earliest arrival → delay = 0
    # (The sound comes FROM direction d, so it hits the mic furthest in the -d direction first.)
    min_proj = min(projections)
    delays_s = [(proj - min_proj) / SPEED_OF_SOUND for proj in projections]

    # Apply fractional delay to each mic channel
    channels = []
    for i, delay in enumerate(delays_s):
        delayed = _fractional_delay(source_signal, delay * sample_rate)
        # Add independent mic noise
        noise = np.random.randn(len(delayed)).astype(np.float32) * noise_per_mic
        channels.append(delayed + noise)

    return channels


def _fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """Apply a fractional sample delay using sinc interpolation in frequency domain."""
    n = len(signal)
    n_fft = 2 ** int(np.ceil(np.log2(n)) + 1)  # Zero-pad for safety

    # FFT
    S = np.fft.rfft(signal, n=n_fft)

    # Apply phase shift (fractional delay)
    freqs = np.arange(len(S))
    phase = np.exp(-2j * np.pi * freqs * delay_samples / n_fft)
    S_delayed = S * phase

    # IFFT and trim
    delayed = np.fft.irfft(S_delayed, n=n_fft)[:n]
    return delayed.astype(np.float32)


def attenuate_for_distance(
    signal: np.ndarray,
    distance_m: float,
    species: str = "Northern Cardinal",
    sample_rate: int = 44100,
    reference_distance: float = 1.0,
) -> np.ndarray:
    """
    Apply distance-based attenuation to a signal.

    Applies:
    1. Inverse square law: amplitude *= reference_distance / distance
    2. Atmospheric absorption: frequency-dependent loss per ISO 9613-1

    Args:
        signal: Audio signal at reference distance
        distance_m: Target distance in meters
        species: Species name for spectral absorption calculation
        sample_rate: Sample rate
        reference_distance: Distance at which signal was "recorded"

    Returns:
        Attenuated signal
    """
    if distance_m <= 0:
        return signal.copy()

    # Inverse square law (amplitude goes as 1/r)
    amplitude_factor = reference_distance / distance_m

    # Atmospheric absorption (frequency-dependent)
    sp = get_species_data(species)
    freq_peak = sp["spectral_peak_hz"]
    absorption_db_per_m = get_absorption_at_freq(freq_peak)
    total_absorption_db = absorption_db_per_m * (distance_m - reference_distance)
    absorption_factor = 10 ** (-total_absorption_db / 20.0)

    attenuated = signal * amplitude_factor * absorption_factor
    return attenuated.astype(np.float32)


def apply_doppler(
    signal: np.ndarray,
    radial_velocity_ms: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Apply Doppler frequency shift to a signal.

    Positive velocity = source approaching (frequency increases).
    Negative velocity = source receding (frequency decreases).

    Args:
        signal: Input audio
        radial_velocity_ms: Radial velocity in m/s (positive = approaching)
        sample_rate: Sample rate

    Returns:
        Doppler-shifted signal
    """
    if abs(radial_velocity_ms) < 0.01:
        return signal.copy()

    # Doppler ratio: f_observed / f_source = c / (c - v_radial)
    ratio = SPEED_OF_SOUND / (SPEED_OF_SOUND - radial_velocity_ms)

    # Resample by the inverse ratio (stretch/compress time)
    n_original = len(signal)
    n_new = int(n_original / ratio)
    if n_new <= 0:
        return signal.copy()

    # Linear interpolation resampling
    old_indices = np.linspace(0, n_original - 1, n_new)
    new_signal = np.interp(old_indices, np.arange(n_original), signal)

    # Pad or trim to original length
    if len(new_signal) < n_original:
        new_signal = np.pad(new_signal, (0, n_original - len(new_signal)))
    else:
        new_signal = new_signal[:n_original]

    return new_signal.astype(np.float32)


def mix_sources(
    sources: list[np.ndarray],
    mix_ratio: list[float] = None,
) -> np.ndarray:
    """Mix multiple source signals into a single audio stream."""
    if not sources:
        return np.array([], dtype=np.float32)

    max_len = max(len(s) for s in sources)

    if mix_ratio is None:
        mix_ratio = [1.0] * len(sources)

    mixed = np.zeros(max_len, dtype=np.float64)
    for s, r in zip(sources, mix_ratio):
        padded = np.pad(s, (0, max_len - len(s)))
        mixed += padded * r

    return mixed.astype(np.float32)
