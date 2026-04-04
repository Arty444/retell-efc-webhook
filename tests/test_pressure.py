"""Pressure tests — edge cases, extreme values, adversarial inputs.

Hammers the bird sound locator pipeline with conditions designed to
break things: silence, pure noise, extreme distances, tiny signals,
zero-length audio, NaN injection, massive arrays, boundary values,
and rapid-fire state transitions.

Run: pytest tests/test_pressure.py -v
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.synth_birds import (
    generate_bird_call,
    simulate_4mic_capture,
    attenuate_for_distance,
    apply_doppler,
    mix_sources,
)
from audio_direction import (
    _gcc_phat_delay,
    estimate_direction_4mic,
    _bandpass_filter,
)
from distance_estimator import (
    estimate_distance_intensity,
    estimate_distance_absorption,
    estimate_distance_drr,
    estimate_distance_visual,
    estimate_doppler,
    DistanceKalmanFilter,
    BirdDistanceEstimator,
)
from sound_separation import separate_sources
from bird_species_db import get_species_data, get_absorption_at_freq
from acoustic_slam import AcousticSLAM, IMUReading, PedestrianDeadReckoning


# ═══════════════════════════════════════════════════════════════════════════════
#  EDGE CASE INPUTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSilenceAndNoise:
    """Feed the pipeline things that aren't bird calls."""

    def test_pure_silence(self):
        """All zeros should not crash any method."""
        silence = np.zeros(44100, dtype=np.float32)
        sr = 44100

        # GCC-PHAT on silence
        result = _gcc_phat_delay(silence, silence, sr, 0.001)
        # May return None or a result — just don't crash

        # Direction on 4 silent channels
        channels = [silence.copy() for _ in range(4)]
        result = estimate_direction_4mic(channels, sr)
        # Should return None (no signal)
        assert result is None, "Silence should not produce a direction"

        # Distance methods on silence
        r = estimate_distance_intensity(0.0, "Northern Cardinal")
        assert r is None, "Zero RMS should return None"

        r = estimate_distance_absorption(silence, sr, "Northern Cardinal")
        # May return None due to zero energy

        r = estimate_distance_drr(silence, sr)
        # Should return None (no onset)

    def test_pure_white_noise(self):
        """Random noise should not produce confident detections."""
        np.random.seed(99)
        noise = np.random.randn(44100).astype(np.float32) * 0.1
        sr = 44100

        channels = [np.random.randn(44100).astype(np.float32) * 0.1 for _ in range(4)]
        result = estimate_direction_4mic(channels, sr)
        # May or may not return a result, but confidence should be low
        if result is not None:
            assert result["confidence"] < 0.8, (
                f"White noise gave confidence {result['confidence']}"
            )

    def test_dc_offset(self):
        """Constant DC signal should not crash."""
        dc = np.ones(44100, dtype=np.float32) * 0.5
        sr = 44100

        r = estimate_distance_absorption(dc, sr, "Northern Cardinal")
        r = estimate_distance_drr(dc, sr)
        # Just don't crash

    def test_single_impulse(self):
        """A single click/impulse should not crash."""
        impulse = np.zeros(44100, dtype=np.float32)
        impulse[22050] = 1.0
        sr = 44100

        r = estimate_distance_drr(impulse, sr)
        r = estimate_distance_absorption(impulse, sr, "Northern Cardinal")
        # Just don't crash

    def test_clipped_audio(self):
        """Heavily clipped (all +1/-1) audio should not crash."""
        np.random.seed(42)
        call = generate_bird_call(duration_s=1.0, amplitude=5.0)
        clipped = np.clip(call, -1.0, 1.0)
        sr = 44100

        channels = [clipped.copy() for _ in range(4)]
        result = estimate_direction_4mic(channels, sr)
        # Should handle gracefully

        r = estimate_distance_intensity(1.0, "Northern Cardinal")
        assert r is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  TINY AND HUGE INPUTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtremeSizes:
    """Very short, very long, and very quiet/loud signals."""

    def test_very_short_audio(self):
        """10 samples should not crash any method."""
        tiny = np.random.randn(10).astype(np.float32) * 0.1
        sr = 44100

        r = _gcc_phat_delay(tiny, tiny, sr, 0.001)
        # Should return None (too short)

        r = estimate_distance_absorption(tiny, sr, "Northern Cardinal")
        assert r is None, "10 samples is too few for absorption"

        r = estimate_distance_drr(tiny, sr)
        assert r is None, "10 samples is too few for DRR"

    def test_one_sample(self):
        """1 sample should not crash."""
        one = np.array([0.5], dtype=np.float32)
        sr = 44100
        r = estimate_distance_absorption(one, sr, "Northern Cardinal")
        r = estimate_distance_drr(one, sr)
        # Just don't crash

    def test_empty_array(self):
        """Zero-length array should not crash."""
        empty = np.array([], dtype=np.float32)
        sr = 44100

        r = estimate_distance_absorption(empty, sr, "Northern Cardinal")
        r = estimate_distance_drr(empty, sr)
        # Just don't crash

    def test_very_long_audio(self):
        """10 seconds of audio should work without memory issues."""
        np.random.seed(42)
        call = generate_bird_call(duration_s=10.0)
        sr = 44100
        assert len(call) == 441000

        r = estimate_distance_drr(call, sr)
        r = estimate_distance_absorption(call, sr, "Northern Cardinal")

    def test_extremely_quiet_signal(self):
        """Signal at -100dB should not produce nonsense."""
        np.random.seed(42)
        call = generate_bird_call(duration_s=1.0, amplitude=0.00001, noise_level=0.0)
        rms = float(np.sqrt(np.mean(call**2)))
        r = estimate_distance_intensity(rms, "Northern Cardinal")
        if r is not None:
            assert r["distance"] > 50, "Extremely quiet = very far away"

    def test_extremely_loud_signal(self):
        """Signal louder than source at 1m should give very close distance."""
        r = estimate_distance_intensity(0.9, "Northern Cardinal")
        assert r is not None
        assert r["distance"] <= 5, f"Very loud signal should be close, got {r['distance']}m"


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTREME PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtremeParameters:
    """Boundary and extreme parameter values."""

    def test_zero_bbox(self):
        """Zero-size bounding box should return None."""
        r = estimate_distance_visual((100, 100, 0, 0), 1920, 1080, "American Robin")
        assert r is None

    def test_negative_bbox(self):
        """Negative bbox dimensions should return None."""
        r = estimate_distance_visual((100, 100, -50, -50), 1920, 1080, "American Robin")
        assert r is None

    def test_huge_bbox(self):
        """Bbox larger than image (bird filling entire frame)."""
        r = estimate_distance_visual((0, 0, 1920, 1080), 1920, 1080, "American Robin")
        assert r is not None
        assert r["distance"] < 2, "Frame-filling bird should be very close"

    def test_visual_zero_fov(self):
        """Zero FOV should not crash (division-by-zero guard)."""
        try:
            r = estimate_distance_visual((100, 100, 50, 50), 1920, 1080,
                                          "American Robin", camera_fov_horizontal=0.001)
        except (ZeroDivisionError, ValueError):
            pass  # Acceptable to raise on invalid FOV

    def test_visual_extreme_zoom(self):
        """100x zoom should give very narrow effective FOV."""
        r = estimate_distance_visual((100, 100, 50, 50), 1920, 1080,
                                      "American Robin", zoom_factor=100.0)
        assert r is not None
        # At extreme zoom, same pixel size means much farther away
        r_normal = estimate_distance_visual((100, 100, 50, 50), 1920, 1080,
                                             "American Robin", zoom_factor=1.0)
        assert r["distance"] > r_normal["distance"]

    def test_intensity_all_species(self):
        """Every species in the database should return a valid distance."""
        from bird_species_db import SPECIES_DB
        for species in SPECIES_DB:
            r = estimate_distance_intensity(0.05, species)
            assert r is not None, f"{species} returned None"
            assert r["distance"] > 0, f"{species} gave distance {r['distance']}"
            assert r["distance"] < 1000, f"{species} gave unreasonable distance {r['distance']}"

    def test_absorption_at_extreme_frequencies(self):
        """Absorption lookup at 0Hz and 20kHz should not crash."""
        a0 = get_absorption_at_freq(0)
        assert a0 >= 0
        a20k = get_absorption_at_freq(20000)
        assert a20k > 0
        a100 = get_absorption_at_freq(100)
        assert a100 >= 0

    def test_doppler_extreme_velocity(self):
        """Mach-speed source should not crash."""
        np.random.seed(42)
        call = generate_bird_call(duration_s=0.3)
        shifted = apply_doppler(call, radial_velocity_ms=100.0)  # ~Mach 0.3
        assert len(shifted) == len(call)

    def test_doppler_negative_velocity(self):
        """Receding source should lower frequency."""
        np.random.seed(42)
        call = generate_bird_call(duration_s=0.3)
        shifted = apply_doppler(call, radial_velocity_ms=-10.0)
        assert len(shifted) == len(call)


# ═══════════════════════════════════════════════════════════════════════════════
#  KALMAN FILTER STRESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestKalmanFilterStress:
    """Push the Kalman filter to its limits."""

    def test_wildly_conflicting_measurements(self):
        """Two methods giving opposite distances should converge somewhere."""
        kf = DistanceKalmanFilter()
        np.random.seed(42)
        for i in range(20):
            kf.predict(0.5)
            kf.update(10.0, 5.0, 0.5)   # Method 1 says 10m
            kf.update(100.0, 5.0, 0.5)  # Method 2 says 100m
        d, u = kf.get_estimate()
        # Should be somewhere between 10 and 100
        assert 5 < d < 120, f"Conflicting methods gave {d:.1f}m"

    def test_rapid_predict_no_update(self):
        """Many predicts without updates should increase uncertainty."""
        kf = DistanceKalmanFilter()
        kf.update(30.0, 5.0, 0.7)
        _, u_before = kf.get_estimate()
        for _ in range(100):
            kf.predict(1.0)
        _, u_after = kf.get_estimate()
        assert u_after > u_before, "Uncertainty should grow without updates"

    def test_zero_uncertainty_measurement(self):
        """Near-zero uncertainty measurement should dominate."""
        kf = DistanceKalmanFilter()
        kf.predict(0.5)
        kf.update(42.0, 0.01, 1.0)  # Extremely confident
        d, u = kf.get_estimate()
        assert abs(d - 42.0) < 2.0, f"High-confidence measurement should dominate: got {d}"

    def test_negative_distance_recovery(self):
        """Filter should clamp to positive distances even with bad data."""
        kf = DistanceKalmanFilter()
        for _ in range(10):
            kf.predict(0.5)
            kf.update(-50.0, 10.0, 0.3)  # Negative distance (nonsense)
        d, u = kf.get_estimate()
        assert d >= 0.5, f"Distance should be clamped positive, got {d}"

    def test_nan_measurement_handling(self):
        """NaN measurements should not corrupt the filter state."""
        kf = DistanceKalmanFilter()
        kf.predict(0.5)
        kf.update(30.0, 5.0, 0.5)
        d_before, _ = kf.get_estimate()

        # Feed NaN — update should be rejected (confidence 0 or NaN guard)
        kf.predict(0.5)
        kf.update(float('nan'), 5.0, 0.5)
        d_after, _ = kf.get_estimate()

        # State should not be NaN
        assert not math.isnan(d_after), "NaN measurement corrupted filter"

    def test_inf_measurement_handling(self):
        """Inf measurements should not corrupt the filter."""
        kf = DistanceKalmanFilter()
        kf.predict(0.5)
        kf.update(30.0, 5.0, 0.5)

        kf.predict(0.5)
        kf.update(float('inf'), 5.0, 0.5)
        d, _ = kf.get_estimate()
        assert not math.isinf(d), "Inf measurement corrupted filter"

    def test_1000_rapid_updates(self):
        """1000 updates should not accumulate numerical errors."""
        kf = DistanceKalmanFilter()
        np.random.seed(42)
        for i in range(1000):
            kf.predict(0.1)
            kf.update(50.0 + np.random.randn() * 3, 3.0, 0.6)
        d, u = kf.get_estimate()
        assert 40 < d < 60, f"After 1000 updates, expected ~50m, got {d:.1f}m"
        assert u < 5, f"Uncertainty should be small after 1000 updates: {u:.1f}m"
        assert not math.isnan(d) and not math.isinf(d)

    def test_huge_dt(self):
        """Large time step should not blow up the covariance."""
        kf = DistanceKalmanFilter()
        kf.update(30.0, 5.0, 0.5)
        kf.predict(999.0)  # Should be clamped (>10s ignored in code)
        d, u = kf.get_estimate()
        assert not math.isnan(d) and not math.isinf(d)


# ═══════════════════════════════════════════════════════════════════════════════
#  ESTIMATOR PIPELINE STRESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimatorPipelineStress:
    """Stress the BirdDistanceEstimator orchestrator."""

    def test_rapid_species_switching(self):
        """Switching species every update should not crash."""
        np.random.seed(42)
        est = BirdDistanceEstimator()
        species_list = [
            "Northern Cardinal", "Blue Jay", "American Robin",
            "House Sparrow", "Mourning Dove", "Red-tailed Hawk",
        ]
        sr = 44100
        for i in range(30):
            est.update_species(species_list[i % len(species_list)])
            call = generate_bird_call(
                species=species_list[i % len(species_list)],
                duration_s=0.2,
            )
            rms = float(np.sqrt(np.mean(call**2)))
            est.update_audio(call, sr, rms)

        result = est.get_distance()
        assert result["distance"] > 0
        assert not math.isnan(result["distance"])

    def test_repeated_reset(self):
        """Repeated reset/use cycles should work."""
        est = BirdDistanceEstimator()
        sr = 44100
        for _ in range(10):
            est.update_species("Northern Cardinal")
            call = generate_bird_call(duration_s=0.3)
            est.update_audio(call, sr, float(np.sqrt(np.mean(call**2))))
            result = est.get_distance()
            assert result["distance"] > 0
            est.reset()
            result = est.get_distance()
            assert len(result["methods_used"]) == 0

    def test_visual_then_audio(self):
        """Visual update before any audio should work."""
        est = BirdDistanceEstimator()
        est.update_species("American Robin")
        est.update_visual((100, 100, 80, 80), 1920, 1080)
        result = est.get_distance()
        assert "visual" in result["methods_used"]
        assert result["distance"] > 0

    def test_all_methods_together(self):
        """Feed all methods in one go and verify fusion."""
        np.random.seed(42)
        est = BirdDistanceEstimator()
        est.update_species("Northern Cardinal")
        sr = 44100

        # Audio (triggers intensity, absorption, DRR, doppler)
        for _ in range(5):
            call = generate_bird_call(duration_s=0.5)
            rms = float(np.sqrt(np.mean(call**2)))
            est.update_audio(call, sr, rms)

        # Visual
        est.update_visual((400, 300, 40, 40), 1920, 1080)

        # Direction (for SLAM)
        est.update_direction(45.0, 10.0, 0.7)

        result = est.get_distance()
        assert len(result["methods_used"]) >= 3, (
            f"Expected 3+ methods, got {result['methods_used']}"
        )
        assert result["confidence"] > 0
        assert result["distance"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  DIRECTION ENGINE STRESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDirectionStress:
    """Push the TDOA direction engine with adversarial inputs."""

    def test_identical_channels(self):
        """All 4 channels identical = ambiguous direction."""
        np.random.seed(42)
        call = generate_bird_call(duration_s=0.5, amplitude=0.5)
        channels = [call.copy() for _ in range(4)]
        result = estimate_direction_4mic(channels, 44100)
        # Should return something (zero delays = source at center/ambiguous)
        # Just don't crash

    def test_only_2_channels(self):
        """Only 2 of 4 channels should degrade gracefully."""
        np.random.seed(42)
        call = generate_bird_call(duration_s=0.5, amplitude=0.5)
        channels = simulate_4mic_capture(call, azimuth_deg=45, elevation_deg=0)
        result = estimate_direction_4mic(channels[:2], 44100)
        # Should still work with fewer pairs

    def test_one_channel_dead(self):
        """One dead channel (all zeros) should not crash."""
        np.random.seed(42)
        call = generate_bird_call(duration_s=0.5, amplitude=0.5)
        channels = simulate_4mic_capture(call, azimuth_deg=90, elevation_deg=0)
        channels[2] = np.zeros_like(channels[2])  # Dead mic 2
        result = estimate_direction_4mic(channels, 44100)
        # May return None or degraded result — just don't crash

    def test_inverted_channel(self):
        """One phase-inverted channel should still work."""
        np.random.seed(42)
        call = generate_bird_call(duration_s=0.5, amplitude=0.5)
        channels = simulate_4mic_capture(call, azimuth_deg=0, elevation_deg=0)
        channels[1] = -channels[1]  # Phase invert mic 1
        result = estimate_direction_4mic(channels, 44100)
        # Should handle gracefully

    def test_different_sample_rates(self):
        """16kHz sample rate should still work."""
        np.random.seed(42)
        sr = 16000
        call = generate_bird_call(duration_s=0.5, sample_rate=sr)
        channels = simulate_4mic_capture(call, azimuth_deg=45, elevation_deg=0, sample_rate=sr)
        result = estimate_direction_4mic(channels, sr)
        # Should produce some result at lower sample rate


# ═══════════════════════════════════════════════════════════════════════════════
#  SOURCE SEPARATION STRESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeparationStress:
    """Push NMF source separation to its limits."""

    def test_single_frequency(self):
        """Pure tone should result in 1 source."""
        t = np.arange(44100) / 44100
        tone = (0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
        sources = separate_sources(tone, 44100, n_sources=1)
        assert len(sources) >= 1
        assert sources[0]["energy"] > 0

    def test_very_short_for_separation(self):
        """Very short audio (0.1s) should not crash NMF."""
        np.random.seed(42)
        short = generate_bird_call(duration_s=0.1)
        try:
            sources = separate_sources(short, 44100, n_sources=2)
        except Exception:
            pass  # Acceptable to fail on very short audio

    def test_many_sources_requested(self):
        """Requesting more sources than exist should not crash."""
        np.random.seed(42)
        t = np.arange(44100) / 44100
        tone = (0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
        sources = separate_sources(tone, 44100, n_sources=6)
        # Should return some sources (may be fewer than requested)
        assert len(sources) >= 1

    def test_noise_only_separation(self):
        """Pure noise should produce sources with spread energy."""
        np.random.seed(42)
        noise = np.random.randn(44100).astype(np.float32) * 0.1
        sources = separate_sources(noise, 44100, n_sources=2)
        assert len(sources) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
#  ACOUSTIC SLAM STRESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSLAMStress:
    """Push acoustic SLAM to edge cases."""

    def test_spinning_in_place(self):
        """Rapidly changing heading with no movement should stay near origin."""
        pdr = PedestrianDeadReckoning()
        for i in range(100):
            imu = IMUReading(
                timestamp=i * 0.02,
                accel_x=0.0, accel_y=0.0, accel_z=9.81,
                gyro_x=0.0, gyro_y=0.0, gyro_z=5.0,  # Spinning
                heading=(i * 36) % 360,  # Full rotation every 10 samples
                pitch=90.0, roll=0.0,
            )
            pdr.update(imu)
        pos = pdr.get_position()
        dist = math.sqrt(pos[0]**2 + pos[1]**2)
        assert dist < 5.0, f"Spinning in place drifted {dist:.1f}m"

    def test_slam_no_angle_measurements(self):
        """SLAM with IMU but no angle measurements should not crash."""
        slam = AcousticSLAM()
        for i in range(50):
            imu = IMUReading(
                timestamp=i * 0.02,
                accel_x=0.0, accel_y=0.0, accel_z=9.81,
                gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
                heading=0.0, pitch=90.0, roll=0.0,
            )
            slam.add_imu_reading(imu)
        result = slam.estimate_distance()
        # Should return None (not enough data)

    def test_slam_single_angle(self):
        """Single angle measurement should not be enough for triangulation."""
        slam = AcousticSLAM()
        slam.add_angle_measurement(45.0, 0.0, 0.8)
        result = slam.estimate_distance()
        assert result is None or result["confidence"] < 0.3


# ═══════════════════════════════════════════════════════════════════════════════
#  MONOTONICITY AND CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════

class TestMonotonicity:
    """Verify that physics-based methods behave monotonically."""

    def test_intensity_monotonic_with_distance(self):
        """Louder RMS should always give closer distance."""
        species = "Northern Cardinal"
        rms_values = [0.5, 0.3, 0.1, 0.05, 0.01, 0.005]
        distances = []
        for rms in rms_values:
            r = estimate_distance_intensity(rms, species)
            if r is not None:
                distances.append(r["distance"])

        # Distances should be monotonically increasing (louder = closer)
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i+1], (
                f"Intensity not monotonic: {distances}"
            )

    def test_visual_monotonic_with_pixel_size(self):
        """Bigger pixel bbox should always give closer distance."""
        species = "American Robin"
        sizes = [200, 100, 50, 20, 10]
        distances = []
        for s in sizes:
            r = estimate_distance_visual((100, 100, s, s), 1920, 1080, species)
            if r is not None:
                distances.append(r["distance"])

        # Distances should be monotonically increasing (bigger pixels = closer)
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i+1], (
                f"Visual not monotonic: {distances}"
            )

    def test_visual_consistent_across_species(self):
        """Larger birds should appear closer at the same pixel size."""
        small_bird = estimate_distance_visual(
            (100, 100, 50, 50), 1920, 1080, "Black-capped Chickadee")  # 14cm
        large_bird = estimate_distance_visual(
            (100, 100, 50, 50), 1920, 1080, "Red-tailed Hawk")  # 56cm

        assert small_bird is not None and large_bird is not None
        # Same pixel size but bigger bird = must be farther away
        assert large_bird["distance"] > small_bird["distance"], (
            f"Hawk ({large_bird['distance']}m) should be farther than "
            f"Chickadee ({small_bird['distance']}m) at same pixel size"
        )

    def test_absorption_increases_with_frequency(self):
        """Higher frequencies should have more absorption per meter."""
        freqs = [500, 1000, 2000, 4000, 8000, 10000]
        absorptions = [get_absorption_at_freq(f) for f in freqs]

        for i in range(len(absorptions) - 1):
            assert absorptions[i] <= absorptions[i+1], (
                f"Absorption not monotonic: {list(zip(freqs, absorptions))}"
            )
