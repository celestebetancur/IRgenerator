"""Physical acoustics helpers: air absorption, materials, reverberation formulas,
frequency-curve handling and directivity patterns."""

from __future__ import annotations

import numpy as np

# Octave bands used for all band-wise data (materials, scattering, analysis).
OCTAVE_BANDS = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0])

# --------------------------------------------------------------------------------------
# Material absorption coefficients (random incidence), octave bands 63 Hz .. 16 kHz.
# Values 125 Hz-4 kHz follow commonly published tables (Everest, Vorländer, Kuttruff);
# 63 Hz and 8/16 kHz are extrapolated. Treat them as reasonable starting points.
# --------------------------------------------------------------------------------------
MATERIALS = {
    #                     63    125   250   500   1k    2k    4k    8k    16k
    "concrete_rough":    [0.01, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14],
    "concrete_painted":  [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04],
    "brick":             [0.02, 0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09],
    "brick_painted":     [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04],
    "marble":            [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
    "plaster":           [0.02, 0.02, 0.03, 0.04, 0.05, 0.04, 0.03, 0.03, 0.03],
    "plaster_on_lath":   [0.15, 0.14, 0.10, 0.06, 0.05, 0.04, 0.03, 0.03, 0.03],
    "gypsum_board":      [0.35, 0.29, 0.10, 0.05, 0.04, 0.07, 0.09, 0.10, 0.10],
    "wood_panel":        [0.30, 0.28, 0.22, 0.17, 0.09, 0.10, 0.11, 0.11, 0.12],
    "wood_floor":        [0.15, 0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07, 0.08],
    "wood_thick":        [0.20, 0.19, 0.14, 0.09, 0.06, 0.06, 0.05, 0.05, 0.05],
    "linoleum":          [0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02],
    "carpet_thin":       [0.02, 0.02, 0.06, 0.14, 0.37, 0.60, 0.65, 0.65, 0.65],
    "carpet_thick":      [0.05, 0.08, 0.24, 0.57, 0.69, 0.71, 0.73, 0.73, 0.73],
    "glass_window":      [0.40, 0.35, 0.25, 0.18, 0.12, 0.07, 0.04, 0.03, 0.03],
    "glass_thick":       [0.20, 0.18, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02],
    "curtain_light":     [0.03, 0.03, 0.04, 0.11, 0.17, 0.24, 0.35, 0.40, 0.45],
    "curtain_heavy":     [0.10, 0.14, 0.35, 0.55, 0.72, 0.70, 0.65, 0.65, 0.65],
    "acoustic_tile":     [0.40, 0.50, 0.70, 0.60, 0.70, 0.70, 0.50, 0.50, 0.50],
    "foam_50mm":         [0.05, 0.08, 0.25, 0.60, 0.90, 0.95, 0.95, 0.95, 0.95],
    "mineral_wool_100mm":[0.40, 0.60, 0.95, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    "bass_trap":         [0.60, 0.80, 0.90, 0.95, 0.95, 0.95, 0.90, 0.90, 0.90],
    "audience_seated":   [0.50, 0.60, 0.74, 0.88, 0.96, 0.93, 0.85, 0.85, 0.85],
    "seats_upholstered": [0.40, 0.49, 0.66, 0.80, 0.88, 0.82, 0.70, 0.70, 0.70],
    "seats_wood_empty":  [0.10, 0.15, 0.19, 0.22, 0.39, 0.38, 0.30, 0.30, 0.30],
    "water":             [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02],
    "metal_sheet":       [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
    "anechoic":          [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    "rigid":             [0.001]*9,
}

SURFACES = ["west", "east", "south", "north", "floor", "ceiling"]  # x=0, x=Lx, y=0, y=Ly, z=0, z=Lz

PATTERNS = {  # first-order pattern g(theta) = a + (1-a) cos(theta)
    "omni": 1.0,
    "subcardioid": 0.7,
    "wide_cardioid": 0.7,
    "cardioid": 0.5,
    "supercardioid": 0.366,
    "hypercardioid": 0.25,
    "figure8": 0.0,
}


def speed_of_sound(temperature_c: float) -> float:
    return 331.3 * np.sqrt(1.0 + temperature_c / 273.15)


def air_absorption_db_per_m(f, temperature_c=20.0, humidity_pct=50.0, pressure_kpa=101.325):
    """ISO 9613-1 atmospheric absorption coefficient in dB/m for frequencies f (Hz)."""
    f = np.asarray(f, dtype=float)
    T = temperature_c + 273.15
    T0 = 293.15
    T01 = 273.16
    pa = pressure_kpa
    pr = 101.325
    psat = pr * 10.0 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    h = humidity_pct * psat / pa  # molar concentration of water vapour (%)
    frO = (pa / pr) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (pa / pr) * (T / T0) ** -0.5 * (9.0 + 280.0 * h * np.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1.0)))
    alpha = 8.686 * f ** 2 * (
        1.84e-11 * (pr / pa) * (T / T0) ** 0.5
        + (T / T0) ** -2.5 * (
            0.01275 * np.exp(-2239.1 / T) / (frO + f ** 2 / frO)
            + 0.1068 * np.exp(-3352.0 / T) / (frN + f ** 2 / frN)
        )
    )
    return alpha


def resolve_band_values(spec, name="value", allow_material=True):
    """Turn a number, a list of 9 band values, a material name or {freqs, values} into
    an array over OCTAVE_BANDS."""
    if spec is None:
        return None
    if isinstance(spec, str):
        if allow_material and spec in MATERIALS:
            return np.array(MATERIALS[spec], dtype=float)
        raise ValueError(f"Unknown material '{spec}' for {name}. Known: {sorted(MATERIALS)}")
    if isinstance(spec, (int, float)):
        return np.full(len(OCTAVE_BANDS), float(spec))
    if isinstance(spec, dict):
        if "material" in spec:
            base = resolve_band_values(spec["material"], name)
            return base * float(spec.get("scale", 1.0)) + float(spec.get("offset", 0.0))
        return interp_curve(spec["freqs"], spec["values"], OCTAVE_BANDS)
    arr = np.asarray(spec, dtype=float)
    if arr.size == 1:
        return np.full(len(OCTAVE_BANDS), float(arr))
    if arr.size != len(OCTAVE_BANDS):
        raise ValueError(f"{name}: expected {len(OCTAVE_BANDS)} octave values (63 Hz..16 kHz), got {arr.size}")
    return arr


def interp_curve(freqs, values, f_out):
    """Interpolate values on a log-frequency axis; values held constant beyond the ends."""
    freqs = np.asarray(freqs, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(freqs)
    freqs, values = freqs[order], values[order]
    lf = np.log2(np.maximum(np.asarray(f_out, dtype=float), 1e-3))
    return np.interp(lf, np.log2(freqs), values)


def curve_from_spec(spec, f_out, default=None):
    """Evaluate a frequency curve spec on f_out.

    Accepted specs: number, {freqs, values}, list of 9 octave values,
    or the 'shelf' form {mid, low_ratio, high_ratio, low_hz, high_hz}."""
    f_out = np.asarray(f_out, dtype=float)
    if spec is None:
        return None if default is None else np.full(f_out.shape, float(default))
    if isinstance(spec, (int, float)):
        return np.full(f_out.shape, float(spec))
    if isinstance(spec, dict) and "mid" in spec:
        mid = float(spec["mid"])
        lo_r = float(spec.get("low_ratio", 1.0))
        hi_r = float(spec.get("high_ratio", 1.0))
        lo_hz = float(spec.get("low_hz", 250.0))
        hi_hz = float(spec.get("high_hz", 2000.0))
        lf = np.log2(np.maximum(f_out, 1e-3))
        # smooth transitions one octave wide around each corner
        lo_w = 0.5 * (1 - np.tanh(2.0 * (lf - np.log2(lo_hz))))
        hi_w = 0.5 * (1 + np.tanh(2.0 * (lf - np.log2(hi_hz))))
        log_ratio = lo_w * np.log(lo_r) + hi_w * np.log(hi_r)
        return mid * np.exp(log_ratio)
    if isinstance(spec, dict):
        return interp_curve(spec["freqs"], spec["values"], f_out)
    arr = np.asarray(spec, dtype=float)
    if arr.size == len(OCTAVE_BANDS):
        return interp_curve(OCTAVE_BANDS, arr, f_out)
    raise ValueError(f"Cannot interpret frequency curve spec: {spec!r}")


def pattern_coeff(p):
    if isinstance(p, (int, float)):
        return float(p)
    if p not in PATTERNS:
        raise ValueError(f"Unknown directivity '{p}'. Known: {sorted(PATTERNS)} or a number 0..1")
    return PATTERNS[p]


def orientation_vector(orientation):
    """[azimuth_deg, elevation_deg] -> unit vector. Azimuth 0 = +x, 90 = +y."""
    az, el = np.deg2rad(orientation[0]), np.deg2rad(orientation[1] if len(orientation) > 1 else 0.0)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def directivity_gain(a, axis, directions):
    """First-order pattern gain for unit direction vectors (N,3)."""
    if a >= 1.0:
        return np.ones(directions.shape[0])
    cos_t = directions @ axis
    return a + (1.0 - a) * cos_t


def diffuse_power(a):
    """Mean of g^2 over the sphere for a first-order pattern (1 for omni)."""
    return a * a + (1.0 - a) ** 2 / 3.0


def mixing_time_s(volume, surface):
    """Perceptual mixing time model (Lindau, Kosanke & Weinzierl 2012): 20 V/S + 12 ms."""
    return (20.0 * volume / surface + 12.0) / 1000.0


def reverberation_time(volume, surface, alpha_mean, m_air, formula="eyring", c=343.0):
    """Sabine / Eyring reverberation time. alpha_mean: (bands,), m_air: energy attenuation (1/m)."""
    alpha_mean = np.clip(alpha_mean, 1e-6, 0.999999)
    if formula == "sabine":
        absorption = surface * alpha_mean
    elif formula == "eyring":
        absorption = -surface * np.log(1.0 - alpha_mean)
    else:
        raise ValueError("reverb_formula must be 'eyring' or 'sabine'")
    return (24.0 * np.log(10.0) / c) * volume / (absorption + 4.0 * m_air * volume)
