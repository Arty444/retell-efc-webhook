"""Bird species acoustic and physical reference database.

Contains known call volumes (dB SPL at 1 meter) and body dimensions
for distance estimation. Sources: Cornell Lab of Ornithology,
Brumm & Todt (2002), Brackenbury (1979), various field studies.

Call volume (dB SPL at 1m) enables inverse-square-law distance estimation.
Body length (cm) enables camera-based visual rangefinding.
Spectral peak frequencies enable atmospheric absorption modeling.
"""

# Reference: dB SPL at 1 meter distance, typical call/song
# Body length in cm (beak to tail)
# Spectral peak in Hz (dominant frequency of primary call)
# Spectral bandwidth in Hz (range of vocalization)

SPECIES_DB = {
    # ── Common North American Birds ──────────────────────────────────────
    "American Robin": {
        "call_db_1m": 82, "body_length_cm": 25, "wingspan_cm": 36,
        "spectral_peak_hz": 2800, "spectral_bw_hz": (1800, 5500),
    },
    "Northern Cardinal": {
        "call_db_1m": 85, "body_length_cm": 22, "wingspan_cm": 30,
        "spectral_peak_hz": 3500, "spectral_bw_hz": (2000, 7000),
    },
    "Blue Jay": {
        "call_db_1m": 92, "body_length_cm": 28, "wingspan_cm": 40,
        "spectral_peak_hz": 3000, "spectral_bw_hz": (1500, 6000),
    },
    "Black-capped Chickadee": {
        "call_db_1m": 78, "body_length_cm": 14, "wingspan_cm": 20,
        "spectral_peak_hz": 4200, "spectral_bw_hz": (3000, 7500),
    },
    "House Sparrow": {
        "call_db_1m": 80, "body_length_cm": 16, "wingspan_cm": 24,
        "spectral_peak_hz": 3800, "spectral_bw_hz": (2500, 6500),
    },
    "Song Sparrow": {
        "call_db_1m": 80, "body_length_cm": 15, "wingspan_cm": 22,
        "spectral_peak_hz": 3500, "spectral_bw_hz": (2000, 7000),
    },
    "American Goldfinch": {
        "call_db_1m": 76, "body_length_cm": 12, "wingspan_cm": 20,
        "spectral_peak_hz": 5000, "spectral_bw_hz": (3000, 8000),
    },
    "Mourning Dove": {
        "call_db_1m": 75, "body_length_cm": 30, "wingspan_cm": 45,
        "spectral_peak_hz": 600, "spectral_bw_hz": (300, 1200),
    },
    "American Crow": {
        "call_db_1m": 95, "body_length_cm": 45, "wingspan_cm": 85,
        "spectral_peak_hz": 1500, "spectral_bw_hz": (800, 3500),
    },
    "Red-winged Blackbird": {
        "call_db_1m": 88, "body_length_cm": 22, "wingspan_cm": 33,
        "spectral_peak_hz": 3200, "spectral_bw_hz": (1500, 6000),
    },
    "European Starling": {
        "call_db_1m": 84, "body_length_cm": 21, "wingspan_cm": 38,
        "spectral_peak_hz": 3000, "spectral_bw_hz": (1000, 8000),
    },
    "Downy Woodpecker": {
        "call_db_1m": 80, "body_length_cm": 16, "wingspan_cm": 28,
        "spectral_peak_hz": 4000, "spectral_bw_hz": (2000, 8000),
    },
    "Red-bellied Woodpecker": {
        "call_db_1m": 86, "body_length_cm": 24, "wingspan_cm": 40,
        "spectral_peak_hz": 3500, "spectral_bw_hz": (1500, 6000),
    },
    "White-breasted Nuthatch": {
        "call_db_1m": 79, "body_length_cm": 15, "wingspan_cm": 27,
        "spectral_peak_hz": 3800, "spectral_bw_hz": (2500, 6000),
    },
    "Tufted Titmouse": {
        "call_db_1m": 82, "body_length_cm": 16, "wingspan_cm": 25,
        "spectral_peak_hz": 3500, "spectral_bw_hz": (2000, 6000),
    },
    "Carolina Wren": {
        "call_db_1m": 90, "body_length_cm": 13, "wingspan_cm": 18,
        "spectral_peak_hz": 3200, "spectral_bw_hz": (2000, 7000),
    },
    "House Finch": {
        "call_db_1m": 80, "body_length_cm": 14, "wingspan_cm": 24,
        "spectral_peak_hz": 4000, "spectral_bw_hz": (2500, 7500),
    },
    "Dark-eyed Junco": {
        "call_db_1m": 78, "body_length_cm": 15, "wingspan_cm": 23,
        "spectral_peak_hz": 4500, "spectral_bw_hz": (3000, 7000),
    },
    "Northern Mockingbird": {
        "call_db_1m": 88, "body_length_cm": 25, "wingspan_cm": 35,
        "spectral_peak_hz": 3000, "spectral_bw_hz": (1000, 8000),
    },
    "Eastern Bluebird": {
        "call_db_1m": 76, "body_length_cm": 18, "wingspan_cm": 30,
        "spectral_peak_hz": 3500, "spectral_bw_hz": (2000, 6000),
    },
    "Cedar Waxwing": {
        "call_db_1m": 74, "body_length_cm": 17, "wingspan_cm": 30,
        "spectral_peak_hz": 7000, "spectral_bw_hz": (5000, 9000),
    },
    "Ruby-throated Hummingbird": {
        "call_db_1m": 70, "body_length_cm": 8, "wingspan_cm": 10,
        "spectral_peak_hz": 6000, "spectral_bw_hz": (3000, 9000),
    },
    "Bald Eagle": {
        "call_db_1m": 92, "body_length_cm": 80, "wingspan_cm": 200,
        "spectral_peak_hz": 2500, "spectral_bw_hz": (1500, 5000),
    },
    "Red-tailed Hawk": {
        "call_db_1m": 94, "body_length_cm": 50, "wingspan_cm": 120,
        "spectral_peak_hz": 2800, "spectral_bw_hz": (1500, 5000),
    },
    "Great Horned Owl": {
        "call_db_1m": 88, "body_length_cm": 55, "wingspan_cm": 130,
        "spectral_peak_hz": 500, "spectral_bw_hz": (200, 1500),
    },
    "Barred Owl": {
        "call_db_1m": 86, "body_length_cm": 50, "wingspan_cm": 110,
        "spectral_peak_hz": 600, "spectral_bw_hz": (250, 1800),
    },
    "Wild Turkey": {
        "call_db_1m": 96, "body_length_cm": 100, "wingspan_cm": 140,
        "spectral_peak_hz": 1200, "spectral_bw_hz": (500, 3000),
    },
    "Canada Goose": {
        "call_db_1m": 98, "body_length_cm": 100, "wingspan_cm": 170,
        "spectral_peak_hz": 800, "spectral_bw_hz": (400, 2000),
    },
    "Mallard": {
        "call_db_1m": 88, "body_length_cm": 58, "wingspan_cm": 90,
        "spectral_peak_hz": 1500, "spectral_bw_hz": (500, 3500),
    },
    "Great Blue Heron": {
        "call_db_1m": 90, "body_length_cm": 115, "wingspan_cm": 180,
        "spectral_peak_hz": 800, "spectral_bw_hz": (300, 2000),
    },

    # ── Common European Birds ────────────────────────────────────────────
    "Eurasian Blackbird": {
        "call_db_1m": 85, "body_length_cm": 25, "wingspan_cm": 36,
        "spectral_peak_hz": 2500, "spectral_bw_hz": (1500, 6000),
    },
    "European Robin": {
        "call_db_1m": 78, "body_length_cm": 14, "wingspan_cm": 21,
        "spectral_peak_hz": 4000, "spectral_bw_hz": (2000, 8000),
    },
    "Great Tit": {
        "call_db_1m": 82, "body_length_cm": 14, "wingspan_cm": 24,
        "spectral_peak_hz": 4500, "spectral_bw_hz": (2500, 7000),
    },
    "Common Chaffinch": {
        "call_db_1m": 84, "body_length_cm": 15, "wingspan_cm": 26,
        "spectral_peak_hz": 3500, "spectral_bw_hz": (2000, 7000),
    },
    "Eurasian Wren": {
        "call_db_1m": 90, "body_length_cm": 10, "wingspan_cm": 15,
        "spectral_peak_hz": 4500, "spectral_bw_hz": (2500, 8500),
    },
    "Common Nightingale": {
        "call_db_1m": 90, "body_length_cm": 16, "wingspan_cm": 25,
        "spectral_peak_hz": 3000, "spectral_bw_hz": (1000, 8000),
    },
    "Common Cuckoo": {
        "call_db_1m": 92, "body_length_cm": 33, "wingspan_cm": 58,
        "spectral_peak_hz": 700, "spectral_bw_hz": (500, 1200),
    },
    "Eurasian Magpie": {
        "call_db_1m": 90, "body_length_cm": 46, "wingspan_cm": 56,
        "spectral_peak_hz": 2000, "spectral_bw_hz": (1000, 5000),
    },
    "Common Wood Pigeon": {
        "call_db_1m": 78, "body_length_cm": 41, "wingspan_cm": 75,
        "spectral_peak_hz": 500, "spectral_bw_hz": (300, 900),
    },
    "Barn Swallow": {
        "call_db_1m": 76, "body_length_cm": 18, "wingspan_cm": 33,
        "spectral_peak_hz": 4500, "spectral_bw_hz": (2000, 8000),
    },

    # ── Common Australian Birds ──────────────────────────────────────────
    "Australian Magpie": {
        "call_db_1m": 90, "body_length_cm": 40, "wingspan_cm": 76,
        "spectral_peak_hz": 2000, "spectral_bw_hz": (800, 5000),
    },
    "Laughing Kookaburra": {
        "call_db_1m": 96, "body_length_cm": 42, "wingspan_cm": 60,
        "spectral_peak_hz": 1800, "spectral_bw_hz": (800, 4000),
    },
    "Sulphur-crested Cockatoo": {
        "call_db_1m": 100, "body_length_cm": 50, "wingspan_cm": 95,
        "spectral_peak_hz": 2000, "spectral_bw_hz": (500, 5000),
    },
    "Rainbow Lorikeet": {
        "call_db_1m": 92, "body_length_cm": 28, "wingspan_cm": 46,
        "spectral_peak_hz": 3500, "spectral_bw_hz": (1500, 7000),
    },
}

# Default values for species not in the database
DEFAULT_SPECIES = {
    "call_db_1m": 84,       # Average songbird
    "body_length_cm": 20,
    "wingspan_cm": 30,
    "spectral_peak_hz": 3500,
    "spectral_bw_hz": (2000, 7000),
}

# Atmospheric absorption coefficients at 20°C, 50% relative humidity
# Source: ISO 9613-1, Bass et al. (1995)
# Values in dB per meter
ATMOSPHERIC_ABSORPTION_DB_PER_M = {
    250: 0.0011,
    500: 0.0025,
    1000: 0.0050,
    2000: 0.0110,
    3000: 0.0180,
    4000: 0.0250,
    5000: 0.0380,
    6000: 0.0480,
    7000: 0.0560,
    8000: 0.0640,
    10000: 0.0900,
}


def get_species_data(species_name: str) -> dict:
    """Look up species data, with fuzzy matching fallback."""
    # Exact match
    if species_name in SPECIES_DB:
        return SPECIES_DB[species_name]

    # Case-insensitive match
    lower = species_name.lower()
    for name, data in SPECIES_DB.items():
        if name.lower() == lower:
            return data

    # Partial match (species name contains query or vice versa)
    for name, data in SPECIES_DB.items():
        if lower in name.lower() or name.lower() in lower:
            return data

    return DEFAULT_SPECIES


def get_absorption_at_freq(freq_hz: float) -> float:
    """Interpolate atmospheric absorption coefficient for a given frequency."""
    freqs = sorted(ATMOSPHERIC_ABSORPTION_DB_PER_M.keys())
    coeffs = [ATMOSPHERIC_ABSORPTION_DB_PER_M[f] for f in freqs]

    if freq_hz <= freqs[0]:
        return coeffs[0]
    if freq_hz >= freqs[-1]:
        return coeffs[-1]

    # Linear interpolation
    for i in range(len(freqs) - 1):
        if freqs[i] <= freq_hz <= freqs[i + 1]:
            t = (freq_hz - freqs[i]) / (freqs[i + 1] - freqs[i])
            return coeffs[i] + t * (coeffs[i + 1] - coeffs[i])

    return 0.05  # fallback
