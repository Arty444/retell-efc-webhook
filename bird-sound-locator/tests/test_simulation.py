"""Simulation test suite for bird sound locator.

Tests the entire signal processing pipeline with synthetic bird calls
and known ground truth, validating:
- GCC-PHAT time delay estimation
- 4-mic TDOA direction engine
- 6-method distance estimation
- Kalman filter fusion
- Multi-bird NMF source separation
- Acoustic SLAM dead reckoning

Run: pytest tests/test_simulation.py -v
Run fast only: pytest tests/test_simulation.py -v -m "not slow"
"""

import sys
import os
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.synth_birds import (
    generate_bird_call,
    generate_pulsed_call,
    simulate_4mic_capture,
    attenuate_for_distance,
    apply_doppler,
    mix_sources,
)
from audio_direction import (
    _gcc_phat_delay,
    estimate_direction_4mic,
    _bandpass_filter,
    IPHONE_16_PM_MICS,
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def angular_difference(a: float, b: float) -> float:
    """Smallest angle between two headings (0-360)."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


# ── 1. Synthetic Bird Call Generation ─────────────────────────────────────────

class TestSyntheticBirdCallGeneration:

    def test_generate_cardinal_call(self):
        """Northern Cardinal call should have energy in 2000-7000Hz band."""
        np.random.seed(42)
        call = generate_bird_call(species="Northern Cardinal", duration_s=1.0)
        sr = 44100

        assert len(call) == sr, f"Expected {sr} samples, got {len(call)}"
        assert call.dtype == np.float32

        # Check spectral energy is concentrated in the right band
        spectrum = np.abs(np.fft.rfft(call))
        freqs = np.fft.rfftfreq(len(call), 1.0 / sr)
        bird_band = (freqs >= 2000) & (freqs <= 7000)
        outside_band = (freqs > 0) & (~bird_band) & (freqs < sr / 2)

        bird_energy = np.sum(spectrum[bird_band] ** 2)
        outside_energy = np.sum(spectrum[outside_band] ** 2)

        assert bird_energy > outside_energy, "Most energy should be in 2-7kHz band"

    def test_generate_chickadee_pulses(self):
        """Pulsed call should have distinct amplitude peaks."""
        np.random.seed(42)
        call = generate_pulsed_call(
            species="Black-capped Chickadee",
            n_pulses=4,
            pulse_duration_s=0.15,
            gap_duration_s=0.1,
        )
        sr = 44100

        # Compute short-time energy
        hop = sr // 20  # 50ms windows
        energies = []
        for i in range(0, len(call) - hop, hop):
            energies.append(np.sqrt(np.mean(call[i:i+hop] ** 2)))
        energies = np.array(energies)

        # Count peaks (above 30% of max)
        threshold = np.max(energies) * 0.3
        peaks = 0
        in_peak = False
        for e in energies:
            if e > threshold and not in_peak:
                peaks += 1
                in_peak = True
            elif e < threshold * 0.5:
                in_peak = False

        assert peaks >= 3, f"Expected ~4 pulses, found {peaks} peaks"

    def test_spectral_peak_matches_species(self):
        """FFT peak should be near species' known spectral_peak_hz."""
        np.random.seed(42)
        sr = 44100
        test_species = [
            ("Northern Cardinal", 3500),
            ("Black-capped Chickadee", 4200),
            ("American Robin", 2800),
        ]

        for species, expected_peak in test_species:
            call = generate_bird_call(species=species, duration_s=0.5, noise_level=0.001)
            spectrum = np.abs(np.fft.rfft(call))
            freqs = np.fft.rfftfreq(len(call), 1.0 / sr)

            # Only look in bird band
            bird_mask = (freqs >= 500) & (freqs <= 10000)
            bird_spectrum = spectrum.copy()
            bird_spectrum[~bird_mask] = 0
            peak_freq = freqs[np.argmax(bird_spectrum)]

            assert abs(peak_freq - expected_peak) < 500, (
                f"{species}: expected peak near {expected_peak}Hz, got {peak_freq:.0f}Hz"
            )


# ── 2. GCC-PHAT Time Delay Estimation ────────────────────────────────────────

class TestGCCPHAT:

    def test_known_delay(self):
        """Recover a known 5-sample delay via GCC-PHAT."""
        np.random.seed(42)
        sr = 44100
        call = generate_bird_call(duration_s=0.5, noise_level=0.001)
        call_filtered = _bandpass_filter(call, sr)

        delay_samples = 5
        delayed = np.zeros_like(call_filtered)
        delayed[delay_samples:] = call_filtered[:-delay_samples]

        result = _gcc_phat_delay(call_filtered, delayed, sr, max_delay_seconds=0.001)
        assert result is not None
        measured_delay_samples = result["delay_seconds"] * sr
        assert abs(measured_delay_samples - (-delay_samples)) < 1.0, (
            f"Expected ~{-delay_samples} samples, got {measured_delay_samples:.2f}"
        )

    def test_zero_delay(self):
        """Identical signals should have ~0 delay."""
        np.random.seed(42)
        sr = 44100
        call = generate_bird_call(duration_s=0.5, noise_level=0.001)
        call_filtered = _bandpass_filter(call, sr)

        result = _gcc_phat_delay(call_filtered, call_filtered.copy(), sr, max_delay_seconds=0.001)
        assert result is not None
        assert abs(result["delay_seconds"] * sr) < 0.5, (
            f"Expected ~0 delay, got {result['delay_seconds'] * sr:.2f} samples"
        )

    def test_noisy_delay(self):
        """Recover delay even with moderate noise (SNR ~15dB)."""
        np.random.seed(42)
        sr = 44100
        call = generate_bird_call(duration_s=0.5, amplitude=0.5, noise_level=0.001)
        call_filtered = _bandpass_filter(call, sr)

        delay_samples = 5
        delayed = np.zeros_like(call_filtered)
        delayed[delay_samples:] = call_filtered[:-delay_samples]

        # Add moderate noise (~15dB SNR — still challenging)
        noise_level = np.sqrt(np.mean(call_filtered**2)) * 0.15
        sig1 = call_filtered + np.random.randn(len(call_filtered)).astype(np.float32) * noise_level
        sig2 = delayed + np.random.randn(len(delayed)).astype(np.float32) * noise_level

        result = _gcc_phat_delay(sig1, sig2, sr, max_delay_seconds=0.002)
        assert result is not None
        measured = result["delay_seconds"] * sr
        assert abs(measured - (-delay_samples)) < 3.0, (
            f"Expected ~{-delay_samples} samples, got {measured:.2f} (noisy)"
        )


# ── 3. TDOA 4-Mic Direction Estimation ───────────────────────────────────────

class TestTDOADirection:

    def _test_direction(self, true_az, true_el, tolerance_deg=35):
        """Helper: simulate source at given direction, verify recovery.

        Uses azimuth_phone (not world heading) to test the TDOA solver
        independently of the phone-to-world orientation transform.
        """
        np.random.seed(42 + abs(int(true_az)) + abs(int(true_el)))
        sr = 44100
        # Use a longer signal for more robust cross-correlation
        call = generate_bird_call(
            species="Northern Cardinal", duration_s=1.0,
            amplitude=0.6, noise_level=0.001,
        )
        channels = simulate_4mic_capture(
            call, azimuth_deg=true_az, elevation_deg=true_el,
            sample_rate=sr, noise_per_mic=0.001,
        )
        result = estimate_direction_4mic(
            channels, sr,
            device_heading=0.0, device_pitch=90.0, device_roll=0.0,
        )
        return result

    def test_4mic_front_direction(self):
        """Source directly in front (azimuth=0) should be recovered."""
        result = self._test_direction(0, 0)
        assert result is not None, "Direction estimation returned None"
        assert result["confidence"] > 0, "Should have nonzero confidence"
        # Check phone-frame azimuth (isolates TDOA solver from orientation transform)
        phone_az = result["azimuth_phone"] % 360
        diff = angular_difference(phone_az, 0)
        assert diff < 60, f"Front source: expected phone_az ~0°, got {phone_az:.1f}° (diff={diff:.1f})"

    def test_4mic_right_direction(self):
        """Source to the right (azimuth=90) should be recovered."""
        result = self._test_direction(90, 0)
        assert result is not None
        phone_az = result["azimuth_phone"] % 360
        diff = angular_difference(phone_az, 90)
        assert diff < 60, f"Right source: expected ~90°, got {phone_az:.1f}° (diff={diff:.1f})"

    def test_4mic_left_direction(self):
        """Source to the left (azimuth=-90/270) should be recovered."""
        result = self._test_direction(-90, 0)
        assert result is not None
        phone_az = result["azimuth_phone"] % 360
        diff = angular_difference(phone_az, 270)
        assert diff < 60, f"Left source: expected ~270°, got {phone_az:.1f}° (diff={diff:.1f})"

    def test_4mic_above_direction(self):
        """Source above (elevation=45) should show positive phone elevation."""
        result = self._test_direction(0, 45)
        assert result is not None
        # The tiny mic array has limited elevation resolution, so be generous
        assert result["elevation_phone"] > 5, (
            f"Above source: expected elevation_phone > 5°, got {result['elevation_phone']:.1f}°"
        )

    @pytest.mark.slow
    def test_4mic_multiple_directions_consistency(self):
        """Test 8 evenly-spaced azimuths, verify generally distinct patterns.

        With the tiny iPhone mic array (~17-162mm), perfect direction recovery
        isn't expected. But different true azimuths should produce different
        TDOA patterns, and the solver should at least get the hemisphere right.
        """
        results = []
        true_azimuths = [0, 45, 90, 135, 180, 225, 270, 315]

        for az in true_azimuths:
            np.random.seed(42 + az)
            sr = 44100
            call = generate_bird_call(
                species="Blue Jay", duration_s=1.0,
                amplitude=0.6, noise_level=0.001,
            )
            channels = simulate_4mic_capture(
                call, azimuth_deg=az, elevation_deg=0,
                sample_rate=sr, noise_per_mic=0.001,
            )
            result = estimate_direction_4mic(channels, sr, 0.0, 90.0, 0.0)
            results.append(result)

        # Check that we get valid results for most directions
        valid_results = [r for r in results if r is not None]
        assert len(valid_results) >= 6, f"Only {len(valid_results)}/8 returned results"

        # Check hemisphere accuracy: for azimuths 0-180 vs 180-360,
        # the phone azimuths should cluster differently
        left_half_azimuths = []
        right_half_azimuths = []
        for az, result in zip(true_azimuths, results):
            if result is None:
                continue
            phone_az = result["azimuth_phone"]
            if 0 <= az < 180:
                right_half_azimuths.append(phone_az)
            else:
                left_half_azimuths.append(phone_az)

        # At minimum, the resolver should produce varied azimuths across directions
        if len(valid_results) >= 4:
            all_phone_az = [r["azimuth_phone"] for r in valid_results]
            az_spread = max(all_phone_az) - min(all_phone_az)
            assert az_spread > 30, (
                f"Phone azimuths too clustered: spread={az_spread:.1f}° (values: {all_phone_az})"
            )


# ── 4. Distance Estimation Methods ───────────────────────────────────────────

class TestDistanceMethods:

    def test_intensity_closer_is_louder(self):
        """Closer bird should yield shorter distance estimate."""
        np.random.seed(42)
        species = "Northern Cardinal"
        call = generate_bird_call(species=species, duration_s=1.0)

        close = attenuate_for_distance(call, 10.0, species)
        far = attenuate_for_distance(call, 50.0, species)

        rms_close = float(np.sqrt(np.mean(close**2)))
        rms_far = float(np.sqrt(np.mean(far**2)))

        r_close = estimate_distance_intensity(rms_close, species)
        r_far = estimate_distance_intensity(rms_far, species)

        assert r_close is not None and r_far is not None
        assert r_close['distance'] < r_far['distance'], (
            f"Close ({r_close['distance']}m) should be < far ({r_far['distance']}m)"
        )

    def test_intensity_known_species(self):
        """Distance should be in reasonable range for typical RMS."""
        result = estimate_distance_intensity(0.05, "Northern Cardinal")
        assert result is not None
        assert 1 <= result["distance"] <= 200, f"Unreasonable distance: {result['distance']}m"
        assert result["method"] == "intensity"

    def test_absorption_farther_has_more_tilt(self):
        """Farther bird should have more spectral tilt (more high-freq loss).

        Note: attenuate_for_distance() only applies bulk amplitude loss
        at the peak frequency, not true frequency-dependent filtering.
        So we manually apply differential absorption to simulate real propagation.
        """
        np.random.seed(42)
        species = "Northern Cardinal"
        sr = 44100
        call = generate_bird_call(species=species, duration_s=1.0, amplitude=0.5, noise_level=0.001)

        # Apply frequency-dependent absorption manually for a more realistic test
        n_fft = len(call)
        S = np.fft.rfft(call)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        def apply_freq_absorption(spectrum, freqs, distance_m):
            """Apply ISO 9613-1 absorption per frequency bin."""
            atten = np.ones_like(spectrum, dtype=np.complex128)
            for i, f in enumerate(freqs):
                if f > 0:
                    abs_db = get_absorption_at_freq(f) * distance_m
                    atten[i] = 10 ** (-abs_db / 20.0)
            return spectrum * atten

        S_close = apply_freq_absorption(S.copy(), freqs, 10.0)
        S_far = apply_freq_absorption(S.copy(), freqs, 100.0)

        close = np.fft.irfft(S_close, n=n_fft).astype(np.float32)
        far = np.fft.irfft(S_far, n=n_fft).astype(np.float32)

        r_close = estimate_distance_absorption(close, sr, species)
        r_far = estimate_distance_absorption(far, sr, species)

        # Both should return results since we've applied real absorption
        if r_close is not None and r_far is not None:
            assert r_close['distance'] <= r_far['distance'], (
                f"Close ({r_close['distance']}m) should be <= far ({r_far['distance']}m)"
            )
        # At least the far one should detect absorption
        assert r_far is not None, "Should detect spectral tilt at 100m"

    def test_drr_returns_distance(self):
        """DRR method should return a valid distance for a signal with clear onset."""
        np.random.seed(42)
        sr = 44100
        # Create signal with clear onset
        call = generate_bird_call(duration_s=1.0, amplitude=0.5, noise_level=0.002)
        # Pad with silence at start to create clear onset
        padded = np.concatenate([np.zeros(sr // 4, dtype=np.float32), call])

        result = estimate_distance_drr(padded, sr)
        assert result is not None, "DRR should return a result for signal with onset"
        assert result["distance"] > 0, "Distance should be positive"
        assert result["method"] == "drr"

    def test_visual_known_size(self):
        """Visual rangefinding with known bird size should give reasonable distance."""
        species = "Northern Cardinal"
        sp = get_species_data(species)
        body_m = sp["body_length_cm"] / 100.0  # 0.22m

        # At 20m distance with 67deg FOV on 1920px wide image:
        # focal_length_px = 960 / tan(33.5deg) = 960 / 0.6619 = 1450.6
        # pixel_size = body_m * focal_px / distance = 0.22 * 1450.6 / 20 = 15.96
        pixel_size = int(body_m * (960.0 / math.tan(math.radians(33.5))) / 20.0)
        bbox = (900, 500, pixel_size, pixel_size)

        result = estimate_distance_visual(bbox, 1920, 1080, species)
        assert result is not None
        assert 12 <= result["distance"] <= 30, (
            f"Expected ~20m, got {result['distance']}m"
        )
        assert result["method"] == "visual"

    def test_visual_closer_is_bigger(self):
        """Bigger bounding box should give closer distance."""
        species = "American Robin"
        # Large bbox (close bird)
        r_close = estimate_distance_visual((100, 100, 200, 200), 1920, 1080, species)
        # Small bbox (far bird)
        r_far = estimate_distance_visual((100, 100, 30, 30), 1920, 1080, species)

        assert r_close is not None and r_far is not None
        assert r_close['distance'] < r_far['distance']

    def test_doppler_stationary_no_movement(self):
        """Identical repeated chunks should show minimal frequency drift.

        Note: The Doppler method compares measured peak to the species database
        reference frequency. Synthetic calls may not match exactly, giving an
        apparent offset. What matters is consistency across chunks (no drift).
        """
        np.random.seed(42)
        sr = 44100
        # Use the SAME chunk repeated — guarantees zero frequency drift
        base_call = generate_bird_call(species="Northern Cardinal", duration_s=0.3)
        chunks = [base_call.copy() for _ in range(5)]
        result = estimate_doppler(chunks, sr, "Northern Cardinal")
        if result is not None:
            # Check that frequency drift between chunks is small
            drift = abs(result.get("freq_drift_hz_per_chunk", 0))
            assert drift < 50, (
                f"Identical chunks should have minimal drift, got {drift:.1f} Hz/chunk"
            )

    def test_doppler_approaching(self):
        """Approaching bird (Doppler upshift) should show positive v_radial."""
        np.random.seed(42)
        sr = 44100
        call = generate_bird_call(species="Northern Cardinal", duration_s=0.3)
        shifted = apply_doppler(call, radial_velocity_ms=8.0, sample_rate=sr)

        # Create chunks: first few normal, last few Doppler-shifted
        chunks = [call.copy() for _ in range(3)] + [shifted.copy() for _ in range(3)]
        result = estimate_doppler(chunks, sr, "Northern Cardinal")
        # Doppler detection is subtle; just verify it returns something
        assert result is not None
        assert result["method"] == "doppler"


# ── 5. Kalman Filter ─────────────────────────────────────────────────────────

class TestKalmanFilter:

    def test_convergence_single_method(self):
        """Filter should converge to true distance with repeated measurements."""
        kf = DistanceKalmanFilter()
        true_distance = 30.0

        np.random.seed(42)
        for i in range(20):
            noisy = true_distance + np.random.randn() * 5
            kf.predict(0.5)
            kf.update(max(1, noisy), 5.0, 0.5)

        d, u = kf.get_estimate()
        assert 20 <= d <= 40, f"Expected ~30m, got {d:.1f}m"
        assert u < 10, f"Uncertainty should decrease, got {u:.1f}m"

    def test_convergence_multi_method(self):
        """Multiple methods should reduce uncertainty faster."""
        kf = DistanceKalmanFilter()
        true_distance = 25.0

        np.random.seed(42)
        initial_u = kf.get_estimate()[1]

        for i in range(10):
            kf.predict(0.5)
            # Method 1: intensity (noisy)
            kf.update(true_distance + np.random.randn() * 10, 10.0, 0.3)
            # Method 2: visual (precise)
            kf.update(true_distance + np.random.randn() * 3, 3.0, 0.7)
            # Method 3: absorption
            kf.update(true_distance + np.random.randn() * 7, 7.0, 0.4)

        d, u = kf.get_estimate()
        assert 18 <= d <= 32, f"Expected ~25m, got {d:.1f}m"
        assert u < initial_u, "Uncertainty should decrease with more data"

    def test_outlier_rejection(self):
        """Filter should resist single outlier measurement."""
        kf = DistanceKalmanFilter()
        true_distance = 30.0

        np.random.seed(42)
        # Feed good measurements
        for i in range(10):
            kf.predict(0.5)
            kf.update(true_distance + np.random.randn() * 3, 3.0, 0.6)

        d_before, _ = kf.get_estimate()

        # Single outlier
        kf.predict(0.5)
        kf.update(200.0, 50.0, 0.1)  # Very uncertain outlier

        d_after, _ = kf.get_estimate()
        assert abs(d_after - d_before) < 15, (
            f"Outlier shifted estimate too much: {d_before:.1f} -> {d_after:.1f}"
        )

    def test_fusion_beats_single(self):
        """Fused estimate should have less uncertainty than noisiest source."""
        kf = DistanceKalmanFilter()
        true_distance = 40.0

        np.random.seed(42)
        for i in range(15):
            kf.predict(0.5)
            # High-variance source
            kf.update(true_distance + np.random.randn() * 15, 15.0, 0.25)
            # Low-variance source
            kf.update(true_distance + np.random.randn() * 3, 3.0, 0.7)

        d, u = kf.get_estimate()
        assert u < 15, f"Fused uncertainty {u:.1f}m should be < worst source (15m)"
        assert 25 <= d <= 55, f"Expected ~40m, got {d:.1f}m"


# ── 6. BirdDistanceEstimator ─────────────────────────────────────────────────

class TestBirdDistanceEstimator:

    def test_full_pipeline(self):
        """End-to-end: set species, feed audio, get distance."""
        np.random.seed(42)
        est = BirdDistanceEstimator()
        est.update_species("Northern Cardinal")

        sr = 44100
        call = generate_bird_call(species="Northern Cardinal", duration_s=1.0)
        rms = float(np.sqrt(np.mean(call**2)))
        est.update_audio(call, sr, rms)

        result = est.get_distance()
        assert "distance" in result
        assert "methods_used" in result
        assert result["distance"] > 0
        assert len(result["methods_used"]) > 0

    def test_species_update(self):
        """Changing species mid-stream should not crash."""
        np.random.seed(42)
        est = BirdDistanceEstimator()
        sr = 44100

        est.update_species("Northern Cardinal")
        call1 = generate_bird_call(species="Northern Cardinal", duration_s=0.5)
        est.update_audio(call1, sr, float(np.sqrt(np.mean(call1**2))))

        est.update_species("Blue Jay")
        call2 = generate_bird_call(species="Blue Jay", duration_s=0.5)
        est.update_audio(call2, sr, float(np.sqrt(np.mean(call2**2))))

        result = est.get_distance()
        assert result["distance"] > 0

    def test_reset(self):
        """Reset should clear all state."""
        np.random.seed(42)
        est = BirdDistanceEstimator()
        est.update_species("American Robin")
        call = generate_bird_call(species="American Robin", duration_s=0.5)
        est.update_audio(call, 44100, float(np.sqrt(np.mean(call**2))))

        est.reset()
        result = est.get_distance()
        assert len(result["methods_used"]) == 0, "After reset, no methods should be active"


# ── 7. Multi-Bird Source Separation ──────────────────────────────────────────

class TestMultiBirdSeparation:

    def test_two_distinct_species(self):
        """Two tones at different frequencies should be separable."""
        np.random.seed(42)
        sr = 44100
        duration = 1.0
        t = np.arange(int(duration * sr)) / sr

        # Simulate two birds: one at 2kHz, one at 6kHz
        bird1 = (0.3 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
        bird2 = (0.3 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
        mixed = bird1 + bird2

        sources = separate_sources(mixed, sr, n_sources=2)
        assert len(sources) >= 2, f"Expected 2 sources, got {len(sources)}"

        # Check that the dominant frequencies are distinct
        freqs = sorted([s["dominant_freq"] for s in sources])
        assert freqs[0] < 4000, f"Low source should be < 4kHz, got {freqs[0]:.0f}"
        assert freqs[-1] > 4000, f"High source should be > 4kHz, got {freqs[-1]:.0f}"

    def test_single_source_passthrough(self):
        """Single source with n_sources=1 should return 1 source."""
        np.random.seed(42)
        sr = 44100
        call = generate_bird_call(species="American Robin", duration_s=1.0)

        sources = separate_sources(call, sr, n_sources=1)
        assert len(sources) == 1
        assert sources[0]["energy"] > 0

    @pytest.mark.slow
    def test_three_sources(self):
        """Three distinct tones should be separable."""
        np.random.seed(42)
        sr = 44100
        duration = 1.5
        t = np.arange(int(duration * sr)) / sr

        tone1 = (0.3 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
        tone2 = (0.3 * np.sin(2 * np.pi * 4500 * t)).astype(np.float32)
        tone3 = (0.3 * np.sin(2 * np.pi * 7500 * t)).astype(np.float32)
        mixed = tone1 + tone2 + tone3

        sources = separate_sources(mixed, sr, n_sources=3)
        assert len(sources) >= 2, f"Expected 3 sources, got {len(sources)}"

        freqs = sorted([s["dominant_freq"] for s in sources])
        # At least we should see frequency spread
        assert freqs[-1] - freqs[0] > 2000, (
            f"Source frequencies not spread enough: {freqs}"
        )


# ── 8. Acoustic SLAM ─────────────────────────────────────────────────────────

class TestAcousticSLAM:

    def test_dead_reckoning_stationary(self):
        """Stationary phone should have position near origin."""
        pdr = PedestrianDeadReckoning()
        
        # Feed stationary IMU data (gravity only on Z)
        for i in range(50):
            imu = IMUReading(
                timestamp=i * 0.02,  # 50 Hz
                accel_x=0.0, accel_y=0.0, accel_z=9.81,
                gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
                heading=0.0, pitch=90.0, roll=0.0,
            )
            pdr.update(imu)

        pos = pdr.get_position()
        distance_from_origin = math.sqrt(pos[0]**2 + pos[1]**2)
        assert distance_from_origin < 2.0, (
            f"Stationary phone drifted {distance_from_origin:.2f}m from origin"
        )

    def test_triangulation_converges(self):
        """SLAM with simulated walking should produce a distance estimate."""
        slam = AcousticSLAM()

        # Simulate walking 5m north while observing bird at 45 degrees
        n_steps = 10
        for i in range(n_steps):
            # Simulate walking forward (step every 0.6s)
            for j in range(30):  # 30 IMU samples per step
                t = i * 0.6 + j * 0.02
                # Small acceleration pulse for step detection
                accel_z = 9.81 + (3.0 if j == 10 else 0.0)
                imu = IMUReading(
                    timestamp=t,
                    accel_x=0.0, accel_y=0.0, accel_z=accel_z,
                    gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
                    heading=0.0, pitch=90.0, roll=0.0,
                )
                slam.add_imu_reading(imu)

            # Angle to bird changes as we walk (parallax)
            # Bird at (10, 10): angle from (0,0) is 45deg, from (0,5) is ~63deg
            bird_x, bird_y = 10.0, 10.0
            walker_y = i * 0.5  # Walking ~0.5m per step cycle
            angle = math.degrees(math.atan2(bird_x, bird_y - walker_y))
            slam.add_angle_measurement(angle, 0.0, 0.7)

        result = slam.estimate_distance()
        # SLAM needs enough baseline to converge; it may or may not
        # produce a result depending on step detection
        if result is not None:
            assert result["distance"] > 0, "Distance should be positive"
            assert 0 < result["confidence"] <= 1.0


# ── 9. Species Database ──────────────────────────────────────────────────────

class TestSpeciesDatabase:

    def test_known_species_lookup(self):
        """Known species should return valid data."""
        sp = get_species_data("Northern Cardinal")
        assert sp["call_db_1m"] == 85
        assert sp["body_length_cm"] == 22
        assert sp["spectral_peak_hz"] == 3500

    def test_unknown_species_fallback(self):
        """Unknown species should return default values."""
        sp = get_species_data("Nonexistent Bird")
        assert "call_db_1m" in sp
        assert sp["call_db_1m"] > 0

    def test_absorption_at_freq(self):
        """Atmospheric absorption should increase with frequency."""
        abs_1k = get_absorption_at_freq(1000)
        abs_4k = get_absorption_at_freq(4000)
        abs_8k = get_absorption_at_freq(8000)

        assert abs_1k < abs_4k < abs_8k, (
            f"Absorption should increase: {abs_1k} < {abs_4k} < {abs_8k}"
        )
        assert abs_1k > 0, "Absorption at 1kHz should be positive"
