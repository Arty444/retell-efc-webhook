"""Bird call sound source separation.

Isolates individual bird calls from mixed audio using spectral analysis
and Non-negative Matrix Factorization (NMF). This is a frequency-domain
approach to the "cocktail party problem" for bird sounds.

Science: Computational Auditory Scene Analysis (CASA)
- Uses spectrogram decomposition to separate overlapping sound sources
- NMF factorizes the magnitude spectrogram into basis vectors (spectral
  templates) and activation matrices (when each source is active)
- Each separated source can then be independently classified by BirdNET
"""

import numpy as np
from scipy.signal import stft, istft, butter, sosfilt
from sklearn.decomposition import NMF


# Bird vocalization band
BIRD_FREQ_LOW = 800    # Hz
BIRD_FREQ_HIGH = 10000 # Hz

# STFT parameters
NPERSEG = 2048
NOVERLAP = 1536  # 75% overlap for good time resolution
HOP_LENGTH = NPERSEG - NOVERLAP


def _bird_band_mask(frequencies: np.ndarray) -> np.ndarray:
    """Create a boolean mask for bird vocalization frequency bins."""
    return (frequencies >= BIRD_FREQ_LOW) & (frequencies <= BIRD_FREQ_HIGH)


def separate_sources(
    audio: np.ndarray,
    sample_rate: int = 44100,
    n_sources: int = 4,
    max_sources: int = 6,
) -> list[dict]:
    """
    Separate overlapping bird calls from a mixed audio signal.

    Uses Non-negative Matrix Factorization (NMF) on the magnitude spectrogram
    to decompose the audio into individual source components.

    Args:
        audio: Float32 mono audio array
        sample_rate: Sample rate in Hz
        n_sources: Expected number of sources to separate (auto-estimated if 0)
        max_sources: Maximum number of sources to extract

    Returns:
        List of separated sources:
        [
            {
                "audio": np.ndarray (float32),     # Isolated audio signal
                "dominant_freq": float,             # Peak frequency in Hz
                "energy": float,                    # Relative energy of this source
                "temporal_pattern": list[float],    # Amplitude envelope over time
                "freq_range": (float, float),       # Frequency range (low, high) in Hz
            },
            ...
        ]
    """
    if len(audio) < NPERSEG:
        return [{"audio": audio, "dominant_freq": 0, "energy": 1.0,
                 "temporal_pattern": [], "freq_range": (0, 0)}]

    # Compute STFT
    frequencies, times, Zxx = stft(audio, fs=sample_rate, nperseg=NPERSEG, noverlap=NOVERLAP)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Focus on bird frequency band
    bird_mask = _bird_band_mask(frequencies)
    bird_magnitude = magnitude[bird_mask, :]
    bird_freqs = frequencies[bird_mask]

    if bird_magnitude.size == 0 or bird_magnitude.max() == 0:
        return []

    # Auto-estimate number of sources if needed
    if n_sources <= 0:
        n_sources = _estimate_source_count(bird_magnitude, bird_freqs, max_sources)

    n_sources = min(n_sources, max_sources, bird_magnitude.shape[0], bird_magnitude.shape[1])
    if n_sources < 1:
        n_sources = 1

    # Apply NMF to decompose the spectrogram
    # V ≈ W * H where:
    #   V = magnitude spectrogram (freq_bins x time_frames)
    #   W = spectral basis vectors (freq_bins x n_sources) - "what each source sounds like"
    #   H = activation matrix (n_sources x time_frames) - "when each source is active"
    nmf = NMF(n_components=n_sources, init="nndsvda", max_iter=300, random_state=42)

    # NMF needs non-negative input; add small epsilon to avoid zeros
    bird_mag_safe = bird_magnitude + 1e-10
    W = nmf.fit_transform(bird_mag_safe)  # (freq_bins, n_sources)
    H = nmf.components_                    # (n_sources, time_frames)

    # Reconstruct each source via Wiener filtering (soft masking)
    sources = []
    reconstructed_sum = W @ H + 1e-10  # Avoid division by zero

    for i in range(n_sources):
        # Soft mask for this source
        source_reconstruction = np.outer(W[:, i], H[i, :])
        mask = source_reconstruction / reconstructed_sum

        # Apply mask to the full spectrogram (only in bird band)
        source_magnitude = np.zeros_like(magnitude)
        source_magnitude[bird_mask, :] = mask * bird_magnitude

        # Reconstruct time-domain signal using original phase
        source_complex = source_magnitude * np.exp(1j * phase)
        _, source_audio = istft(source_complex, fs=sample_rate, nperseg=NPERSEG, noverlap=NOVERLAP)

        # Trim to original length
        source_audio = source_audio[:len(audio)]

        # Compute source characteristics
        spectral_profile = W[:, i]
        peak_freq_idx = np.argmax(spectral_profile)
        dominant_freq = float(bird_freqs[peak_freq_idx])

        # Frequency range (where the source has >10% of peak energy)
        threshold = spectral_profile.max() * 0.1
        active_bins = bird_freqs[spectral_profile > threshold]
        freq_range = (float(active_bins.min()), float(active_bins.max())) if len(active_bins) > 0 else (0.0, 0.0)

        # Energy relative to total
        energy = float(np.sum(source_reconstruction)) / float(np.sum(reconstructed_sum))

        # Temporal activation pattern (when this source is loud)
        temporal_pattern = H[i, :].tolist()

        sources.append({
            "audio": source_audio.astype(np.float32),
            "dominant_freq": round(dominant_freq, 1),
            "energy": round(energy, 4),
            "temporal_pattern": [round(t, 5) for t in temporal_pattern],
            "freq_range": (round(freq_range[0], 1), round(freq_range[1], 1)),
        })

    # Sort by energy (loudest first), filter out very weak sources
    sources.sort(key=lambda s: s["energy"], reverse=True)
    sources = [s for s in sources if s["energy"] > 0.02]

    return sources


def _estimate_source_count(magnitude: np.ndarray, frequencies: np.ndarray, max_sources: int) -> int:
    """
    Estimate the number of distinct bird sound sources in the spectrogram.

    Uses spectral peak detection across time frames - each distinct frequency
    band with consistent energy likely represents a separate bird.
    """
    # Average spectrum across time
    avg_spectrum = np.mean(magnitude, axis=1)
    if avg_spectrum.max() == 0:
        return 1

    # Normalize
    avg_spectrum = avg_spectrum / avg_spectrum.max()

    # Find peaks (local maxima above 15% threshold)
    threshold = 0.15
    peaks = []
    for i in range(1, len(avg_spectrum) - 1):
        if (avg_spectrum[i] > avg_spectrum[i-1] and
            avg_spectrum[i] > avg_spectrum[i+1] and
            avg_spectrum[i] > threshold):
            peaks.append(i)

    # Merge peaks that are within 500 Hz of each other (likely same bird)
    if len(peaks) > 1 and len(frequencies) > 0:
        freq_resolution = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
        merged = [peaks[0]]
        for p in peaks[1:]:
            if (p - merged[-1]) * freq_resolution > 500:
                merged.append(p)
        peaks = merged

    n = max(1, min(len(peaks), max_sources))
    return n


def compute_source_directions(
    sources: list[dict],
    chunks_by_source: list[list[dict]],
    sample_rate: int = 44100,
) -> list[dict]:
    """
    For each separated source, estimate its direction using the amplitude-heading
    correlation from scan mode data.

    Args:
        sources: Output from separate_sources()
        chunks_by_source: For each source, list of {"audio": np.ndarray, "heading": float}
                         from different compass headings

    Returns:
        List of {"source_idx": int, "heading": float, "confidence": float}
    """
    from audio_direction import estimate_direction

    results = []
    for idx, (source, chunks) in enumerate(zip(sources, chunks_by_source)):
        if not chunks:
            results.append({"source_idx": idx, "heading": None, "confidence": 0.0})
            continue

        direction = estimate_direction(
            [{"pcm_data": c["audio"], "heading": c["heading"]} for c in chunks],
            sample_rate,
        )

        if direction:
            results.append({
                "source_idx": idx,
                "heading": direction["heading"],
                "confidence": direction["confidence"],
            })
        else:
            results.append({"source_idx": idx, "heading": None, "confidence": 0.0})

    return results
