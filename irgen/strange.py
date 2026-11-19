"""Strange spaces: artificial, bizarre and tonal impulse responses.

A second synthesis layer next to the room engine. It builds an IR from

- cavity modes, solved analytically for a box, sphere, cylinder, pipe (open / closed /
  half-open) or bottle (cylinder + Helmholtz neck), filled with any medium and scaled
  from a ping-pong ball up to a mountain ("gigantify": every length x scale),
- a diffuse tail above the highest synthesised mode, level-matched to the modal density,
- plate modes of the shell itself (material wave speed, thickness, loss factor),
- a pitched or dispersive (spring-like) echo train,
- a musical tuning: modes snapped towards a scale, replaced by a scale, or both, and a
  "tonal imprint" (resonant peaks at the scale pitches) on the whole reverberant part,
- motion: the space expands / shrinks while it rings (glide) or breathes (wobble),
- an imported IR (last standard render or a file), resampled by the same scale factor.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
from scipy import signal

from . import acoustics as ac
from .config import deep_merge
from .eq import apply_eq, biquad, dc_block
from .late import band_centers, shape_noise_multi, shoebox_modes, velvet_growing

DEFAULTS = {
    "name": "strange",
    "seed": 1,
    "output": {
        "sample_rate": 48000,
        "channels": 2,               # 1 | 2
        "width": 0.7,                # 0 = mono-identical channels, 1 = fully decorrelated
        "length_s": "auto",
        "max_length_s": 20.0,
        "trim_db": 80.0,
        "lead_ms": 1.0,
        "fade_out_ms": 60.0,
        "dc_highpass_hz": 15.0,
        "normalize": "peak",         # peak | energy | none
        "peak_dbfs": -1.0,
        "format": "PCM_24",
        "dither": True,
    },
    "space": {
        "shape": "sphere",           # box | sphere | cylinder | pipe | bottle
        "radius_m": 0.12,            # sphere / cylinder / pipe / bottle body
        "length_m": 0.3,             # cylinder / pipe / bottle body
        "dimensions": [0.33, 0.2, 0.12],  # box
        "ends": "closed",            # cylinder / pipe: closed | open | half
        "neck_length_m": 0.08,       # bottle
        "neck_radius_m": 0.01,
        "scale": 1.0,                # multiplies every length (gigantify / miniaturise)
        "medium": "air",
        "speed_of_sound": "auto",    # m/s, overrides the medium
        "medium_absorption_scale": 1.0,
        "void_fraction": 0.01,       # bubbly_water: gas volume fraction (Wood's equation)
        "bubble_radius_mm": 1.0,     # bubbly_water: sets the Minnaert absorption peak (scales with the space)
        "boundary": "auto",          # auto (from the wall / medium impedance ratio) | rigid | soft
        "wall_thickness_mm": "auto", # auto = thick wall (rho c); a number = thin shell (mass law at 1 kHz)
        "transmission": 1.0,         # 0..1: how much of the impedance-mismatch loss (sound leaving through the wall) applies
        "wall": "rubber",            # STRANGE_MATERIALS or any irgen material
        "source": [0.3, 0.4, 0.55],  # relative position inside the bounding box, 0..1
        "receiver": [0.65, 0.6, 0.45],
        "t60": "auto",               # "auto" (from wall + medium) | seconds | 9 octave values
        "decay_scale": 1.0,
        "t60_floor_s": 0.0,          # > 0: no band rings shorter than this ("freeze")
    },
    "modes": {"enabled": True, "gain_db": 0.0, "max_modes": 600, "f_max_hz": "auto", "detune_cents": 0.0},
    "tuning": {
        "mode": "off",               # off | snap | replace | add
        "root": "A2",                # note name or Hz
        "scale": "minor_pentatonic",
        "custom_cents": [0, 200, 350, 700, 950],
        "custom_period_cents": 1200.0,
        "amount": 1.0,               # snap strength 0..1
        "range_hz": [40.0, 8000.0],  # pitches used by replace / add / imprint
        "partials": 3,               # replace / add: partials per pitch
        "inharmonicity": 0.0,        # partial h at h^(1 + x)
        "tilt_db_per_oct": -3.0,
        "ring_s": "auto",            # replace / add: "auto" = space T60
        "root_emphasis_db": 3.0,
    },
    "diffuse": {"enabled": True, "gain_db": 0.0, "range": "above_modes"},  # above_modes | full
    "echoes": {"enabled": False, "period": "auto", "period_ms": 25.0, "dispersion": 0.0, "gain_db": -6.0},
    "shell": {"enabled": False, "material": "auto", "thickness_mm": 2.0, "f0_hz": "auto", "count": 60, "gain_db": -10.0},
    "motion": {"glide_oct_per_s": 0.0, "wobble_cents": 0.0, "wobble_hz": 0.3},
    "imprint": {"gain_db": 0.0, "q": 40.0},
    "envelope": {"bloom_ms": 0.0, "gate_ms": 0.0, "reverse": False},
    "imported": {"enabled": False, "source": "last_render", "scale": 1.0, "gain_db": 0.0},
    "direct": {"enabled": True, "level_db": -6.0},   # direct energy re reverberant energy
    "eq": [],
}

# alpha: absorption 63 Hz..16 kHz; c_plate: bending-wave speed proxy (m/s); eta: loss factor;
# rho: density (kg/m^3). rho * c_plate is the wall impedance used against the medium.
STRANGE_MATERIALS = {
    "rubber":    {"alpha": [0.10, 0.08, 0.06, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08], "c_plate": 900, "eta": 0.08, "rho": 1100},
    "steel":     {"alpha": [0.005, 0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.015, 0.02], "c_plate": 5100, "eta": 0.0003, "rho": 7850},
    "brass":     {"alpha": [0.006, 0.006, 0.007, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025], "c_plate": 3500, "eta": 0.0005, "rho": 8500},
    "aluminium": {"alpha": [0.005, 0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.015, 0.02], "c_plate": 5100, "eta": 0.0002, "rho": 2700},
    "glass":     {"alpha": [0.03, 0.03, 0.025, 0.02, 0.02, 0.02, 0.02, 0.025, 0.03], "c_plate": 5300, "eta": 0.001, "rho": 2500},
    "ceramic":   {"alpha": [0.01, 0.01, 0.01, 0.012, 0.015, 0.02, 0.02, 0.025, 0.03], "c_plate": 4500, "eta": 0.002, "rho": 2400},
    "ice":       {"alpha": [0.005, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02, 0.02], "c_plate": 3200, "eta": 0.002, "rho": 917},
    "stone":     {"alpha": [0.01, 0.01, 0.01, 0.015, 0.02, 0.02, 0.025, 0.03, 0.03], "c_plate": 4500, "eta": 0.004, "rho": 2600},
    "concrete":  {"alpha": [0.01, 0.01, 0.015, 0.02, 0.02, 0.025, 0.03, 0.035, 0.04], "c_plate": 3500, "eta": 0.006, "rho": 2400},
    "wood":      {"alpha": [0.15, 0.12, 0.10, 0.08, 0.07, 0.07, 0.08, 0.09, 0.10], "c_plate": 4000, "eta": 0.012, "rho": 600},
    "bamboo":    {"alpha": [0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.07, 0.08, 0.09], "c_plate": 3500, "eta": 0.01, "rho": 700},
    "plastic":   {"alpha": [0.04, 0.04, 0.035, 0.03, 0.03, 0.03, 0.035, 0.04, 0.05], "c_plate": 1700, "eta": 0.02, "rho": 1200},
    "cardboard": {"alpha": [0.35, 0.30, 0.25, 0.20, 0.15, 0.12, 0.12, 0.12, 0.12], "c_plate": 1800, "eta": 0.04, "rho": 700},
    "bone":      {"alpha": [0.02, 0.02, 0.025, 0.03, 0.03, 0.035, 0.04, 0.045, 0.05], "c_plate": 3000, "eta": 0.01, "rho": 1900},
    "chitin":    {"alpha": [0.05, 0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.09], "c_plate": 2500, "eta": 0.015, "rho": 1300},
    "skin":      {"alpha": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.60], "c_plate": 60, "eta": 0.15, "rho": 1100},
    "felt":      {"alpha": [0.05, 0.10, 0.25, 0.50, 0.75, 0.85, 0.90, 0.90, 0.90], "c_plate": 150, "eta": 0.3, "rho": 300},
    "mirror":    {"alpha": [0.0005] * 9, "c_plate": 6000, "eta": 0.00005, "rho": 1e7},   # impossible perfect reflector
}

# c (m/s), rho (kg/m^3), group, absorption:
#   ("iso", k):            ISO 9613 air absorption x k (gases)
#   ("pow", a1k, y, eta):  a1k * (f / 1 kHz)^y dB/m (scattering / relaxation) plus classical viscous
#                          absorption from the shear viscosity eta (Pa s): w^2 (4/3 eta) / (2 rho c^3)
# Liquids and soft solids are nearly lossless at audio frequencies; the walls dominate.
# medium_absorption_scale exaggerates (or removes) the medium's own loss.
MEDIA = {
    "air":               {"c": 343.0, "rho": 1.204, "group": "gas", "abs": ("iso", 1.0)},
    "helium":            {"c": 1007.0, "rho": 0.166, "group": "gas", "abs": ("iso", 0.6)},
    "hydrogen":          {"c": 1310.0, "rho": 0.084, "group": "gas", "abs": ("iso", 0.5)},
    "co2":               {"c": 267.0, "rho": 1.84, "group": "gas", "abs": ("iso", 4.0)},
    "sf6":               {"c": 136.0, "rho": 6.1, "group": "gas", "abs": ("iso", 3.0)},
    "xenon":             {"c": 178.0, "rho": 5.9, "group": "gas", "abs": ("iso", 1.5)},
    "mars_atmosphere":   {"c": 240.0, "rho": 0.020, "group": "gas", "abs": ("iso", 12.0)},
    "titan_atmosphere":  {"c": 194.0, "rho": 5.4, "group": "gas", "abs": ("iso", 2.0)},
    "water":             {"c": 1482.0, "rho": 998.0, "group": "liquid", "abs": ("pow", 2e-7, 2.0, 0.001)},
    "seawater":          {"c": 1500.0, "rho": 1025.0, "group": "liquid", "abs": ("pow", 6e-5, 1.5, 0.0011)},
    "bubbly_water":      {"c": None, "rho": None, "group": "liquid", "abs": ("pow", 2e-7, 2.0, 0.001)},
    "oil":               {"c": 1430.0, "rho": 920.0, "group": "liquid", "abs": ("pow", 1e-4, 2.0, 0.08)},
    "glycerol":          {"c": 1904.0, "rho": 1260.0, "group": "liquid", "abs": ("pow", 2e-4, 2.0, 1.4)},
    "honey":             {"c": 2030.0, "rho": 1420.0, "group": "liquid", "abs": ("pow", 1e-3, 2.0, 10.0)},
    "liquid_nitrogen":   {"c": 853.0, "rho": 807.0, "group": "liquid", "abs": ("pow", 1e-6, 2.0, 0.00016)},
    "superfluid_helium": {"c": 238.0, "rho": 145.0, "group": "liquid", "abs": ("pow", 0.0, 2.0, 0.0)},
    "mercury":           {"c": 1450.0, "rho": 13534.0, "group": "liquid", "abs": ("pow", 6e-7, 2.0, 0.0015)},
    "lava":              {"c": 2500.0, "rho": 2700.0, "group": "molten", "abs": ("pow", 0.05, 1.0, 300.0)},
    "molten_glass":      {"c": 2800.0, "rho": 2400.0, "group": "molten", "abs": ("pow", 0.01, 1.0, 100.0)},
    "molten_iron":       {"c": 3800.0, "rho": 7000.0, "group": "molten", "abs": ("pow", 1e-5, 2.0, 0.006)},
    "molten_aluminium":  {"c": 4600.0, "rho": 2375.0, "group": "molten", "abs": ("pow", 1e-5, 2.0, 0.0013)},
    "molten_gold":       {"c": 2570.0, "rho": 17300.0, "group": "molten", "abs": ("pow", 1e-5, 2.0, 0.005)},
    "jelly":             {"c": 1520.0, "rho": 1040.0, "group": "soft solid", "abs": ("pow", 0.05, 1.0, 0.0)},
    "flesh":             {"c": 1540.0, "rho": 1060.0, "group": "soft solid", "abs": ("pow", 0.07, 1.1, 0.0)},
    "silicone":          {"c": 1000.0, "rho": 1100.0, "group": "soft solid", "abs": ("pow", 0.1, 1.2, 0.0)},
    "aerogel":           {"c": 100.0, "rho": 100.0, "group": "exotic", "abs": ("pow", 2.0, 1.0, 0.0)},
    "steel":             {"c": 5960.0, "rho": 7850.0, "group": "solid", "abs": ("pow", 1e-5, 1.0, 0.0)},
}


def wood_mixture(beta):
    """Bubbly water below the bubble resonance (Wood's equation): (c, rho)."""
    rw, cw, ra, ca = 998.0, 1482.0, 1.2, 343.0
    b = float(np.clip(beta, 1e-7, 0.6))
    rho = (1 - b) * rw + b * ra
    kappa = (1 - b) / (rw * cw * cw) + b / (ra * ca * ca)
    return 1.0 / np.sqrt(rho * kappa), rho


def medium_props(sc, k=1.0):
    """-> c, rho, energy attenuation m(f) (1/m) on OCTAVE_BANDS, extra info."""
    name = sc.get("medium", "air")
    if name not in MEDIA:
        raise ValueError(f"space.medium '{name}' unknown. Known: {sorted(MEDIA)}")
    md = MEDIA[name]
    f = ac.OCTAVE_BANDS
    extra = {}
    if name == "bubbly_water":
        beta = float(sc.get("void_fraction", 0.01))
        c, rho = wood_mixture(beta)
        R = float(sc.get("bubble_radius_mm", 1.0)) / 1000 * k
        fM = 3.26 / max(R, 1e-6)    # Minnaert resonance (air bubbles in water, 1 atm)
        extra = {"void_fraction": beta, "minnaert_hz": round(fM, 1)}
    else:
        c, rho = md["c"], md["rho"]
    kind = md["abs"]
    if kind[0] == "iso":
        db = ac.air_absorption_db_per_m(f) * kind[1]
    else:
        _, a1k, y, eta = kind
        w = 2 * np.pi * f
        db = a1k * (f / 1000.0) ** y + 8.686 * w * w * (4 / 3 * eta) / (2 * rho * c ** 3)
    if name == "bubbly_water":
        # polydisperse cloud: broad absorption peak around the Minnaert frequency
        db = db + min(200.0, 2000.0 * extra["void_fraction"]) / (1 + (2 * (f / fM - fM / f)) ** 2)
    db = db * float(sc.get("medium_absorption_scale", 1.0))
    if not _auto(sc.get("speed_of_sound")):
        c = float(sc["speed_of_sound"])
    return float(c), float(rho), db / (10 * np.log10(np.e)), extra


def wall_impedance(sc, wall):
    """Wall impedance against the medium: thick (rho c) or a thin shell (mass law at 1 kHz)."""
    zw = wall.get("rho", 2000.0) * wall.get("c_plate", 2500.0)
    th = sc.get("wall_thickness_mm", "auto")
    if not _auto(th):
        zw = min(zw, 2 * np.pi * 1000.0 * wall.get("rho", 2000.0) * float(th) / 1000 * float(sc.get("scale", 1.0)))
    return zw


# cents within one period, period in cents
SCALES = {
    "chromatic": (list(range(0, 1200, 100)), 1200.0),
    "major": ([0, 200, 400, 500, 700, 900, 1100], 1200.0),
    "dorian": ([0, 200, 300, 500, 700, 900, 1000], 1200.0),
    "phrygian": ([0, 100, 300, 500, 700, 800, 1000], 1200.0),
    "lydian": ([0, 200, 400, 600, 700, 900, 1100], 1200.0),
    "mixolydian": ([0, 200, 400, 500, 700, 900, 1000], 1200.0),
    "aeolian": ([0, 200, 300, 500, 700, 800, 1000], 1200.0),
    "locrian": ([0, 100, 300, 500, 600, 800, 1000], 1200.0),
    "harmonic_minor": ([0, 200, 300, 500, 700, 800, 1100], 1200.0),
    "double_harmonic": ([0, 100, 400, 500, 700, 800, 1100], 1200.0),
    "major_pentatonic": ([0, 200, 400, 700, 900], 1200.0),
    "minor_pentatonic": ([0, 300, 500, 700, 1000], 1200.0),
    "blues": ([0, 300, 500, 600, 700, 1000], 1200.0),
    "whole_tone": ([0, 200, 400, 600, 800, 1000], 1200.0),
    "octatonic": ([0, 200, 300, 500, 600, 800, 900, 1100], 1200.0),
    "hirajoshi": ([0, 200, 300, 700, 800], 1200.0),
    "in_sen": ([0, 100, 500, 700, 1000], 1200.0),
    "pelog": ([0, 120, 270, 540, 670, 780, 950], 1200.0),
    "slendro": ([0, 240, 480, 720, 960], 1200.0),
    "maqam_rast": ([0, 200, 350, 500, 700, 900, 1050], 1200.0),
    "maqam_bayati": ([0, 150, 300, 500, 700, 800, 1000], 1200.0),
    "just_major": ([0, 203.91, 386.31, 498.04, 701.96, 884.36, 1088.27], 1200.0),
    "harmonic_series": ([1200 * np.log2(k / 8) for k in range(8, 16)], 1200.0),
    "bohlen_pierce": ([i * 1901.955 / 13 for i in range(13)], 1901.955),
    "tet_19": ([i * 1200 / 19 for i in range(19)], 1200.0),
    "quarter_tone": ([i * 50.0 for i in range(24)], 1200.0),
    "major_triad": ([0, 400, 700], 1200.0),
    "minor_triad": ([0, 300, 700], 1200.0),
    "fifths": ([0, 700], 1200.0),
    "drone": ([0], 1200.0),
}

SHAPES = ["box", "sphere", "cylinder", "pipe", "bottle"]
KINDS = ["cavity", "tonal", "shell", "helmholtz"]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _auto(v):
    return v is None or v == "auto"


_NOTE = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}


def note_hz(v):
    """'A4' / 'C#3' / 'Eb2' / 110 -> Hz (A4 = 440)."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"([A-Ga-g])([#b]*)(-?\d+)", s)
    if not m:
        raise ValueError(f"tuning.root: cannot read note '{v}' (use e.g. A2, C#3, Eb4 or Hz)")
    semi = _NOTE[m.group(1).upper()] + m.group(2).count("#") - m.group(2).count("b") + 12 * (int(m.group(3)) - 4)
    return 440.0 * 2 ** (semi / 12)


def scale_def(tc):
    if tc.get("scale") == "custom":
        cents = [float(x) for x in (tc.get("custom_cents") or [0])]
        return cents, float(tc.get("custom_period_cents", 1200.0))
    if tc.get("scale") not in SCALES:
        raise ValueError(f"tuning.scale '{tc.get('scale')}' unknown. Known: {sorted(SCALES)} or custom")
    return SCALES[tc["scale"]]


def scale_grid(root, cents, period, lo, hi):
    """All scale pitches in [lo, hi] -> (freqs, degree index)."""
    lo, hi = max(lo, 1.0), max(hi, lo * 1.001)
    k0 = int(np.floor(1200 * np.log2(lo / root) / period)) - 1
    k1 = int(np.ceil(1200 * np.log2(hi / root) / period)) + 1
    f, d = [], []
    for k in range(k0, k1 + 1):
        for i, cc in enumerate(cents):
            x = root * 2 ** ((cc + k * period) / 1200)
            if lo <= x <= hi:
                f.append(x)
                d.append(i)
    o = np.argsort(f)
    return np.asarray(f)[o], np.asarray(d, int)[o]


def wall_props(name):
    if name in STRANGE_MATERIALS:
        return STRANGE_MATERIALS[name]
    if name in ac.MATERIALS:   # irgen database: absorption only, generic plate
        return {"alpha": ac.MATERIALS[name], "c_plate": 3000, "eta": 0.01, "rho": 2000}
    raise ValueError(f"unknown wall material '{name}'")


def _rel_pos(shape, rel, g):
    """Relative 0..1 position -> coordinates in the shape's own frame."""
    u = np.clip(np.asarray(rel, float), 0.02, 0.98)
    if shape == "box":
        return u * g["dims"]
    a = g["a"]
    p = (2 * u - 1) * a
    if shape == "sphere":
        r = np.linalg.norm(p)
        return p * (0.95 * a / r) if r > 0.95 * a else p
    xy = p[:2]
    r = np.linalg.norm(xy)
    if r > 0.95 * a:
        xy = xy * 0.95 * a / r
    return np.array([np.hypot(*xy), np.arctan2(xy[1], xy[0]), u[2] * g["L"]])   # (r, theta, z)


# --------------------------------------------------------------------------------------
# analytic cavity modes -> (freq, coupling psi(s) psi(r) / Lambda)
# --------------------------------------------------------------------------------------
def _sphere_modes(a, c, f_lim, ps, pr, soft=False):
    """rigid: zeros of j_l'(x); soft (pressure release): zeros of j_l(x)."""
    from scipy.special import eval_legendre, spherical_jn
    X = 2 * np.pi * f_lim * a / c
    rs, rr = np.linalg.norm(ps), np.linalg.norm(pr)
    cosg = float(ps @ pr / (rs * rr)) if rs > 1e-9 and rr > 1e-9 else 1.0
    fs_, cs_ = [], []
    for l in range(0, int(X) + 2):
        x = np.arange(max(0.02, 0.8 * l), X + 0.04, 0.02)
        if x.size < 2:
            break
        d = spherical_jn(l, x, derivative=not soft)
        i = np.flatnonzero(d[:-1] * d[1:] < 0)
        if i.size == 0:
            continue
        z = x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i])   # linear refinement
        if soft:
            norm = (3 / (8 * np.pi)) * spherical_jn(l + 1, z) ** 2
        else:
            norm = (3 / (8 * np.pi)) * spherical_jn(l, z) ** 2 * (1 - l * (l + 1) / z ** 2)
        coup = spherical_jn(l, z * rs / a) * spherical_jn(l, z * rr / a) * (2 * l + 1) / (4 * np.pi) \
            * eval_legendre(l, cosg) / np.maximum(norm, 1e-12)
        fs_.append(z * c / (2 * np.pi * a))
        cs_.append(coup)
    if not fs_:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(fs_), np.concatenate(cs_)


def _cylinder_modes(a, L, ends, c, f_lim, ps, pr, soft=False):
    """rigid: zeros of J_m' and cos axial modes at closed ends; soft: zeros of J_m and
    pressure release at every end. Open ends are pressure release (with end correction)."""
    from scipy.special import jn_zeros, jnp_zeros, jv
    kmax = 2 * np.pi * f_lim / c
    X = kmax * a
    # axial part
    e = 0.61 * a   # open-end correction
    if soft:
        n_open = {"open": 2, "half": 1}.get(ends, 0)
        Le, off = L + n_open * e, (e if ends == "open" else 0.0)
        p = np.arange(1, int(kmax * Le / np.pi) + 2)
        kz = p * np.pi / Le
        Zs, Zr, Zn = np.sin(kz * (ps[2] + off)), np.sin(kz * (pr[2] + off)), np.full(p.size, Le / 2)
    elif ends == "open":
        Le = L + 2 * e
        p = np.arange(1, int(kmax * Le / np.pi) + 2)
        kz = p * np.pi / Le
        Zs, Zr, Zn = np.sin(kz * (ps[2] + e)), np.sin(kz * (pr[2] + e)), np.full(p.size, Le / 2)
    elif ends == "half":   # closed at z = 0, open at z = L
        Le = L + e
        p = np.arange(0, int(kmax * Le / np.pi) + 2)
        kz = (p + 0.5) * np.pi / Le
        Zs, Zr, Zn = np.cos(kz * ps[2]), np.cos(kz * pr[2]), np.full(p.size, Le / 2)
    else:
        p = np.arange(0, int(kmax * L / np.pi) + 2)
        kz = p * np.pi / L
        Zs, Zr, Zn = np.cos(kz * ps[2]), np.cos(kz * pr[2]), np.where(p == 0, L, L / 2)
    # radial part (m, x_mn)
    rad = [] if soft else [(0, 0.0)]
    for m in range(0, int(X) + 2):
        nt = int(max(X - m, 0) / np.pi) + 2
        for z in (jn_zeros(m, nt) if soft else jnp_zeros(m, nt)):
            if z <= X:
                rad.append((m, float(z)))
    if not rad:
        return np.zeros(0), np.zeros(0)
    m = np.array([r[0] for r in rad])
    x = np.array([r[1] for r in rad])
    if soft:
        Rn = (a * a / 2) * jv(m + 1, x) ** 2
    else:
        Jx = np.where(x > 0, jv(m, x), 1.0)
        Rn = np.where(x > 0, (a * a / 2) * (1 - m ** 2 / np.maximum(x, 1e-9) ** 2) * Jx ** 2, a * a / 2)
    Ath = np.where(m == 0, 2 * np.pi, np.pi)
    Rs, Rr = jv(m, x * ps[0] / a), jv(m, x * pr[0] / a)
    ang = np.cos(m * (ps[1] - pr[1]))
    V = np.pi * a * a * L
    kr = x / a
    F = (c / (2 * np.pi)) * np.sqrt(kr[:, None] ** 2 + kz[None, :] ** 2)
    lam = (Ath * Rn)[:, None] * Zn[None, :] / V
    coup = (Rs * Rr * ang)[:, None] * (Zs * Zr)[None, :] / np.maximum(lam, 1e-12)
    sel = (F > 0) & (F <= f_lim)
    return F[sel], coup[sel]


def geometry(sc):
    """Scaled geometry: volume, surface, absorbing extra area, flutter length, mode solver."""
    k = float(sc.get("scale", 1.0))
    shape = sc.get("shape", "sphere")
    g = {"shape": shape, "scale": k}
    if shape == "box":
        dims = np.asarray(sc["dimensions"], float) * k
        g.update(dims=dims, V=float(np.prod(dims)), S=float(2 * (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2])),
                 flutter=float(2 * dims.max()), open_area=0.0, size=dims.tolist())
    elif shape == "sphere":
        a = float(sc["radius_m"]) * k
        g.update(a=a, V=4 / 3 * np.pi * a ** 3, S=4 * np.pi * a ** 2, flutter=2 * a, open_area=0.0, size=[a])
    elif shape in ("cylinder", "pipe", "bottle"):
        a, L = float(sc["radius_m"]) * k, float(sc["length_m"]) * k
        ends = sc.get("ends", "open" if shape == "pipe" else "closed")
        if shape == "bottle":
            ends = "closed"
        n_open = {"open": 2, "half": 1}.get(ends, 0)
        g.update(a=a, L=L, ends=ends, V=np.pi * a * a * L, S=2 * np.pi * a * L + 2 * np.pi * a * a * (1 - n_open / 2),
                 open_area=n_open * np.pi * a * a, flutter=2 * max(L, 2 * a), size=[a, L])
        if shape == "bottle":
            g["neck"] = (float(sc["neck_radius_m"]) * k, float(sc["neck_length_m"]) * k)
    else:
        raise ValueError(f"space.shape must be one of {SHAPES}")
    return g


def _soft_box_modes(dims, src, rcv, c, f_max):
    """Pressure-release box: sin modes, n >= 1 on every axis, Lambda = 1/8."""
    Lx, Ly, Lz = dims
    ax = [np.arange(1, int(2 * f_max * L / c) + 2) for L in dims]
    NX, NY, NZ = (q.ravel() for q in np.meshgrid(*ax, indexing="ij"))
    f = 0.5 * c * np.sqrt((NX / Lx) ** 2 + (NY / Ly) ** 2 + (NZ / Lz) ** 2)
    sel = f <= f_max
    NX, NY, NZ, f = NX[sel], NY[sel], NZ[sel], f[sel]

    def psi(p):
        return np.sin(np.pi * NX * p[0] / Lx) * np.sin(np.pi * NY * p[1] / Ly) * np.sin(np.pi * NZ * p[2] / Lz)
    return f, psi(src) * psi(rcv) * 8.0


def cavity_modes(g, c, f_lim, ps, pr):
    soft = g.get("soft", False)
    if g["shape"] == "box":
        return _soft_box_modes(g["dims"], ps, pr, c, f_lim) if soft else shoebox_modes(g["dims"], ps, pr, c, f_lim)
    if g["shape"] == "sphere":
        return _sphere_modes(g["a"], c, f_lim, ps, pr, soft)
    return _cylinder_modes(g["a"], g["L"], g["ends"], c, f_lim, ps, pr, soft)


# --------------------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------------------
def synth_modes(n, fs, i0, f, amp, t60, motion, trim_db, rng, block=None):
    """Sum of decaying sinusoids from sample i0. amp: (C, K). The whole space glides
    (2^(-glide t)) and breathes (per-mode wobble phase) together.

    Rendered in time blocks: per block only the modes still ringing are evaluated
    (float32, phase re-based per block) and mixed to the C channels with one matrix
    product, so memory stays bounded for any channel count."""
    C, K = amp.shape
    m = n - i0
    out = np.zeros((C, n))
    if K == 0 or m <= 0:
        return out
    t = np.arange(m) / fs
    gl = float(motion.get("glide_oct_per_s", 0.0) or 0.0)
    G = t if abs(gl) < 1e-9 else (1.0 - 2.0 ** (-gl * t)) / (gl * np.log(2.0))
    depth = 2 ** (float(motion.get("wobble_cents", 0.0) or 0.0) / 1200) - 1
    wr = max(float(motion.get("wobble_hz", 0.3) or 0.3), 1e-3)
    phi = rng.uniform(0, 2 * np.pi, K)
    L = np.minimum(m, np.ceil(t60 * trim_db / 60.0 * fs)).astype(int)
    if gl < 0:   # rising pitch: stop each mode before it aliases
        L = np.minimum(L, (np.log2(0.45 * fs / f) / -gl * fs).astype(int))
    L[L < 8] = 0
    nfade = np.maximum(np.minimum(L // 8, int(0.02 * fs)), 1).astype(np.float32)
    delta = 6.9078 / t60
    w0 = 2 * np.pi * f
    wob = (f * depth / wr).astype(np.float32) if depth else None
    amp32 = amp.astype(np.float32)
    if block is None:   # ~8 MB per float32 temporary, whatever the mode count
        block = int(np.clip(2 ** np.floor(np.log2(2e6 / K)), 1024, 8192))

    def work(b0):
        b1 = min(b0 + block, m)
        act = np.flatnonzero(L > b0)
        if act.size == 0:
            return
        tl = (t[b0:b1] - t[b0]).astype(np.float32)
        base = np.mod(w0[act] * G[b0], 2 * np.pi).astype(np.float32)
        ph = (w0[act, None] * (G[b0:b1] - G[b0])[None, :]).astype(np.float32) + base[:, None]
        if wob is not None:
            ph += wob[act, None] * (np.cos(phi[act]).astype(np.float32)[:, None]
                                   - np.cos((2 * np.pi * wr * t[b0:b1])[None, :] + phi[act, None]).astype(np.float32))
        env = np.exp(-delta[act] * t[b0]).astype(np.float32)[:, None] * np.exp(-delta[act].astype(np.float32)[:, None] * tl[None, :])
        rem = (L[act, None] - np.arange(b0, b1)[None, :]).astype(np.float32)
        fade = 0.5 * (1 - np.cos(np.pi * np.clip(rem / nfade[act, None], 0, 1)))
        out[:, i0 + b0:i0 + b1] = amp32[:, act] @ (np.sin(ph) * env * fade)

    W = max(1, min(8, os.cpu_count() or 1))
    with ThreadPoolExecutor(W) as ex:
        list(ex.map(work, range(0, m, block)))
    return out


def echo_train(n, fs, T, t60_fn, dispersion, n_rep_cut=None):
    """h = sum_k (g D z^-T)^k, k >= 1, evaluated on an FFT grid (frequency-dependent loss
    from T60, optional dispersion: high (+) or low (-) frequencies delayed per pass)."""
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    T60 = t60_fn(np.maximum(f, 20.0))
    gmag = np.minimum(10 ** (-3.0 * T / T60), 0.9995)
    ph = 2 * np.pi * f * T
    if dispersion:
        u = np.clip(np.log2(np.maximum(f, 20) / 20) / np.log2(fs / 2 / 20), 0, 1)
        tau = abs(dispersion) * T * (u if dispersion > 0 else 1 - u)
        ph = ph + 2 * np.pi * np.cumsum(tau) * (f[1] - f[0])
    q = gmag * np.exp(-1j * ph)
    q[-1] = q[-1].real
    return np.fft.irfft(q / (1 - q), nfft)[:n]


def _resample(x, k):
    """Play x slower by factor k (k > 1: bigger, lower, longer)."""
    fr = Fraction(float(k)).limit_denominator(64)
    up, down = fr.numerator, fr.denominator
    if up == down:
        return x
    return signal.resample_poly(x, up, down, axis=-1)


# --------------------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------------------
def load_strange(preset):
    if not isinstance(preset, dict):
        with open(preset) as fh:
            preset = json.load(fh)
    return deep_merge(DEFAULTS, {k: v for k, v in preset.items() if not k.startswith("_")})


def generate_strange(preset, imported=None):
    """Render a strange-space preset. imported = (ir, fs) for the imported-IR layer.
    Returns (ir (n,) or (C, n), fs, info, cfg, modes_for_display)."""
    cfg = load_strange(preset)
    oc, sc, mc, tc = cfg["output"], cfg["space"], cfg["modes"], cfg["tuning"]
    fs = int(oc["sample_rate"])
    C = int(np.clip(int(oc.get("channels", 2)), 1, 64))
    width = float(np.clip(oc.get("width", 0.7), 0, 1))
    trim = float(oc.get("trim_db", 80.0))
    seed = int(cfg.get("seed", 1))
    r_mode, r_ch, r_noise, r_tone, r_shell, r_mot = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
    info = {"name": cfg.get("name"), "sample_rate": fs, "channels": C}

    # ---------------- space ----------------
    g = geometry(sc)
    med = sc.get("medium", "air")
    c, rho_m, m_air, med_info = medium_props(sc, g["scale"])
    wall = wall_props(sc.get("wall", "rubber"))
    # impedance mismatch: a wall that matches the medium lets the sound through (absorbs);
    # a wall softer than the medium is a pressure-release ("soft") boundary
    zr = wall_impedance(sc, wall) / (rho_m * c)
    refl = (zr - 1) / (zr + 1)
    bnd = sc.get("boundary", "auto")
    soft = (zr < 1) if bnd == "auto" else (bnd == "soft")
    g["soft"] = bool(soft)
    alpha = np.clip(np.maximum(np.asarray(wall["alpha"], float), float(sc.get("transmission", 1.0)) * (1 - refl ** 2)), 1e-4, 0.999)
    S_tot = g["S"] + g["open_area"]
    alpha_mean = (alpha * g["S"] + 1.0 * g["open_area"]) / S_tot
    if _auto(sc.get("t60")):
        T_bands = ac.reverberation_time(g["V"], S_tot, alpha_mean, m_air, "eyring", c)
    else:
        T_bands = ac.curve_from_spec(sc["t60"], ac.OCTAVE_BANDS)
    T_bands = np.maximum(T_bands * float(sc.get("decay_scale", 1.0)), float(sc.get("t60_floor_s", 0.0) or 0.0))
    T_bands = np.clip(T_bands, 0.005, 600.0)
    lgb, logT = np.log2(ac.OCTAVE_BANDS), np.log(T_bands)

    def t60_fn(f):
        return np.exp(np.interp(np.log2(np.maximum(f, 1e-3)), lgb, logT))

    ps, pr = _rel_pos(g["shape"], sc["source"], g), _rel_pos(g["shape"], sc["receiver"], g)
    if g["shape"] == "box":
        dist = float(np.linalg.norm(ps - pr))
    elif g["shape"] == "sphere":
        dist = float(np.linalg.norm(ps - pr))
    else:
        cart = lambda p: np.array([p[0] * np.cos(p[1]), p[0] * np.sin(p[1]), p[2]])
        dist = float(np.linalg.norm(cart(ps) - cart(pr)))
    t_mix = ac.mixing_time_s(g["V"], S_tot)
    info.update({"shape": g["shape"], "scale": g["scale"], "size_m": np.round(g["size"], 4).tolist(),
                 "volume_m3": g["V"], "surface_m2": S_tot, "speed_of_sound": c, "medium": med, **med_info,
                 "medium_density": rho_m, "boundary": "soft" if soft else "rigid", "impedance_ratio": round(float(zr), 4),
                 "wall_reflection": round(float(refl), 4),
                 "source_receiver_m": round(dist, 4), "t60_bands": dict(zip(ac.OCTAVE_BANDS.tolist(), np.round(T_bands, 3).tolist())),
                 "mixing_time_ms": round(t_mix * 1000, 1)})

    # ---------------- modes ----------------
    K = int(np.clip(mc.get("max_modes", 600), 1, 5000))
    f_top = 0.45 * fs if _auto(mc.get("f_max_hz")) else min(float(mc["f_max_hz"]), 0.45 * fs)
    f_lim = min(f_top, c * (3 * K / (4 * np.pi * g["V"])) ** (1 / 3) * 1.1)
    for _ in range(40):
        fm, cm = cavity_modes(g, c, f_lim, ps, pr)
        if fm.size >= K or f_lim >= f_top:
            break
        f_lim = min(f_top, f_lim * 1.35)
    o = np.argsort(fm)
    fm, cm = fm[o], cm[o]
    if fm.size > K:
        f_x = 0.5 * (fm[K - 1] + fm[K])
        fm, cm = fm[:K], cm[:K]
    else:
        f_x = f_top
    am = cm * (4 * np.pi * c * c / g["V"]) / (2 * np.pi * fm)
    kind = np.zeros(fm.size, int)
    if g["shape"] == "bottle" and not soft:   # compliant walls: no closed neck resonator
        rn, ln = g["neck"]
        A = np.pi * rn * rn
        Le = ln + 1.7 * rn
        fH = c / (2 * np.pi) * np.sqrt(A / (g["V"] * Le))
        Q = min(120.0, 2 * np.pi * np.sqrt(g["V"] * (Le / A) ** 3))
        aH = (4 * np.pi * c * c / g["V"]) / (2 * np.pi * fH) * 2.0
        fm, am, kind = np.append(fm, fH), np.append(am, aH), np.append(kind, 3)
        info.update({"helmholtz_hz": round(fH, 2), "helmholtz_q": round(Q, 1)})
    tm = t60_fn(fm)
    if g["shape"] == "bottle" and not soft:
        tm[-1] = min(tm[-1], 2.2 * Q / fH)
    info.update({"cavity_modes": int((kind == 0).sum()), "lowest_cavity_mode_hz": round(float(fm.min()), 3) if fm.size else None,
                 "modal_crossover_hz": round(float(f_x), 1)})
    # crossover taper of the top modal octave (power complementary with the diffuse tail)
    xw = np.clip(np.log2(np.maximum(fm, 1e-3) / f_x) + 1, 0, 1)
    taper = np.where(kind == 0, np.cos(0.5 * np.pi * xw), 1.0) if f_x < f_top else np.ones(fm.size)

    # modal density in the top octave -> diffuse level
    band = (fm >= f_x / 2) & (fm <= f_x) & (kind == 0)
    if band.sum() < 8:
        band = (fm >= f_x / 4) & (fm <= f_x) & (kind == 0)
    bw = f_x - (f_x / 2 if band.sum() >= 8 else f_x / 4)
    g_diff = np.sqrt(fs * np.sum(am[band] ** 2) / (4 * max(bw, 1.0))) if band.any() else 0.0
    e_phys = float(np.sum(am ** 2 * tm)) if fm.size else 0.0

    # ---------------- tuning ----------------
    tmode = tc.get("mode", "off")
    cents, period = scale_def(tc)
    root = note_hz(tc.get("root", "A2"))
    grid_lo, grid_hi = (float(x) for x in tc.get("range_hz", [40, 8000]))
    full_grid, full_deg = scale_grid(root, cents, period, 20.0, 0.45 * fs)

    def snap(f):
        if not len(f) or not full_grid.size:
            return f
        lg = np.log2(full_grid)
        lf = np.log2(f)
        j = np.clip(np.searchsorted(lg, lf), 1, lg.size - 1)
        near = np.where(np.abs(lg[j] - lf) < np.abs(lg[j - 1] - lf), lg[j], lg[j - 1])
        near = np.where((lf < lg[0] - 0.5) | (lf > lg[-1] + 0.5), lf, near)
        return 2 ** (lf + float(tc.get("amount", 1.0)) * (near - lf))

    F, A, T, Kd = [], [], [], []
    if mc.get("enabled", True) and tmode in ("off", "snap", "add"):
        F.append(snap(fm) if tmode == "snap" else fm)
        A.append(am * taper)
        T.append(tm)
        Kd.append(kind)
    if tmode in ("replace", "add"):
        gf, gd = scale_grid(root, cents, period, grid_lo, grid_hi)
        P = max(1, int(tc.get("partials", 3)))
        h = np.arange(1, P + 1)
        ff = (gf[:, None] * h[None, :] ** (1 + float(tc.get("inharmonicity", 0.0))))
        aa = (h[None, :] ** -0.7) * 10 ** (float(tc.get("tilt_db_per_oct", -3.0)) / 20 * np.log2(ff / gf.min()))
        aa = aa * np.where(gd == 0, 10 ** (float(tc.get("root_emphasis_db", 3.0)) / 20), 1.0)[:, None]
        ff, aa = ff.ravel(), aa.ravel() * r_tone.uniform(0.6, 1.0, ff.size) * np.where(r_tone.random(ff.size) < 0.5, -1, 1)
        keep = ff < 0.45 * fs
        ff, aa = ff[keep], aa[keep]
        tt = t60_fn(ff) if _auto(tc.get("ring_s")) else np.full(ff.size, float(tc["ring_s"]))
        if e_phys > 0 and mc.get("enabled", True):
            aa = aa * np.sqrt(e_phys / max(np.sum(aa ** 2 * tt), 1e-30))
        else:
            aa = aa / np.sqrt(max(np.sum(aa ** 2 * tt), 1e-30))
        F.append(ff)
        A.append(aa)
        T.append(tt)
        Kd.append(np.ones(ff.size, int))
        info["tonal_modes"] = int(ff.size)
    F = np.concatenate(F) if F else np.zeros(0)
    A = np.concatenate(A) if A else np.zeros(0)
    T = np.concatenate(T) if T else np.zeros(0)
    Kd = np.concatenate(Kd) if Kd else np.zeros(0, int)
    det = float(mc.get("detune_cents", 0.0) or 0.0)
    if det and F.size:
        F = F * 2 ** (r_mode.normal(0, det, F.size) / 1200)
    sel = (F > 1.0) & (F < 0.45 * fs)
    F, A, T, Kd = F[sel], A[sel], np.maximum(T[sel], 0.003), Kd[sel]
    A = A * 10 ** (float(mc.get("gain_db", 0.0)) / 20)
    if F.size:
        info["lowest_mode_hz"] = round(float(F.min()), 3)

    # ---------------- shell ----------------
    shc = cfg["shell"]
    Fs_, As_, Ts_ = np.zeros(0), np.zeros(0), np.zeros(0)
    if shc.get("enabled"):
        sw = wall if _auto(shc.get("material")) else wall_props(shc["material"])
        Lc = np.sqrt(g["S"] / 6)
        hth = float(shc.get("thickness_mm", 2.0)) / 1000 * g["scale"]
        f11 = (np.pi / 2) * (hth * sw["c_plate"] / np.sqrt(12)) * (2 / Lc ** 2) if _auto(shc.get("f0_hz")) else float(shc["f0_hz"])
        mm = np.arange(1, 40)
        MM, NN = np.meshgrid(mm, mm, indexing="ij")
        keep = MM <= NN
        fr = (f11 / 2 * (MM ** 2 + NN ** 2))[keep]
        cp = (np.sin(MM * np.pi * 0.37) * np.sin(NN * np.pi * 0.61) * np.sin(MM * np.pi * 0.71) * np.sin(NN * np.pi * 0.23))[keep]
        o = np.argsort(fr)
        fr, cp = fr[o], cp[o]
        ok = (fr > 20) & (fr < 0.45 * fs)
        fr, cp = fr[ok][: int(shc.get("count", 60))], cp[ok][: int(shc.get("count", 60))]
        if tmode != "off":
            fr = snap(fr)
        Fs_, As_ = fr, cp / np.sqrt(fr)
        Ts_ = np.minimum(2.2 / (fr * sw["eta"]), 120.0)
        info.update({"shell_f11_hz": round(float(f11), 2), "shell_modes": int(fr.size)})

    # ---------------- timeline ----------------
    lead = int(round(float(oc.get("lead_ms", 1.0)) * fs / 1000))
    t_d = dist / c          # direct sound arrival (emission at 0)
    audible = ac.OCTAVE_BANDS <= min(16000, fs / 2)
    t_ring = float(np.max(T_bands[audible]))
    if T.size:
        t_ring = max(t_ring, float(np.max(T[F > 40])) if np.any(F > 40) else t_ring)
    if Ts_.size:
        t_ring = max(t_ring, float(np.max(Ts_)))
    t_end = t_ring * trim / 60.0
    imc = cfg["imported"]
    imp = None
    if imc.get("enabled"):
        if imported is None:
            raise ValueError("imported layer enabled but no IR was given (render a standard IR or load a file)")
        x, ifs = imported
        x = np.atleast_2d(np.asarray(x, float))
        if x.shape[0] > x.shape[1]:
            x = x.T
        if ifs != fs:
            from math import gcd
            gg = gcd(int(fs), int(ifs))
            x = signal.resample_poly(x, fs // gg, int(ifs) // gg, axis=-1)
        a0 = np.abs(x).max(axis=0)
        on = int(np.argmax(a0 >= a0.max() * 0.1))
        ksc = float(imc.get("scale", 1.0))
        x = x[:, on: on + int(float(oc.get("max_length_s", 20.0)) * fs / max(ksc, 1e-3)) + 1]
        imp = _resample(x, ksc)
        t_end = max(t_end, imp.shape[1] / fs)
        info["imported_length_s"] = round(imp.shape[1] / fs, 3)
    if _auto(oc.get("length_s")):
        length = min(t_end + lead / fs + 0.05, float(oc.get("max_length_s", 20.0)))
    else:
        length = float(oc["length_s"])
    gate = float(cfg["envelope"].get("gate_ms", 0) or 0) / 1000
    if gate > 0:
        length = min(length, gate + 0.03 + lead / fs)
    n = int(np.ceil(length * fs))
    i0 = lead
    info["length_s"] = n / fs
    # modes start ringing at emission; everything before the direct sound is gated, and the
    # output is aligned so that the direct sound lands at the lead-in
    i_emit = max(0, i0 - int(round(t_d * fs)))

    def ch_gain(a, rng):
        if C == 1:
            return a[None, :]
        z = rng.standard_normal((C, a.size))
        return a[None, :] * ((1 - width) + width * z)

    rev = np.zeros((C, n))
    nr = max(1, min(i0 - i_emit, int(0.002 * fs)))
    pre = np.sin(0.5 * np.pi * np.clip((np.arange(n) - (i0 - nr)) / nr, 0, 1)) ** 2
    M = synth_modes(n, fs, i_emit, F, ch_gain(A, r_ch), T, cfg["motion"], trim, r_mot) * pre if F.size else np.zeros((C, n))
    rev += M
    info["synth_modes"] = int(F.size)

    # ---------------- diffuse tail ----------------
    dc_ = cfg["diffuse"]
    drange = dc_.get("range", "above_modes")
    if dc_.get("enabled", True) and (drange == "full" or f_x < f_top):
        exc = np.zeros((C, n))
        base = velvet_growing(n, fs, i0, r_noise, 100.0, 0.25 * fs, t_mix, 2.0, max(0.01, 0.3 * t_mix))
        for ch in range(C):
            own = velvet_growing(n, fs, i0, r_noise, 100.0, 0.25 * fs, t_mix, 2.0, max(0.01, 0.3 * t_mix))
            exc[ch] = np.sqrt(1 - width) * base + np.sqrt(width) * own if C > 1 else base
        centers = band_centers(fs, 2)   # decays vary smoothly; 1/2 octave is enough here
        tt = np.maximum(np.arange(n) - i0, 0) / fs

        def env_fn(i, fc):
            w = 1.0 if drange == "full" else float(np.sin(0.5 * np.pi * np.clip(np.log2(fc / f_x) + 1, 0, 1)))
            if w <= 1e-4:
                return None
            return np.exp(-6.9078 * tt / float(t60_fn(fc))) * w

        D = shape_noise_multi(exc, fs, centers, env_fn, None, None)
        if g_diff <= 0:
            em = np.sum(M[0] ** 2)
            g_diff = np.sqrt(em / max(np.sum(D[0] ** 2), 1e-30)) if em > 0 else 1.0 / np.sqrt(max(np.sum(D[0] ** 2), 1e-30))
        rev += D * g_diff * 10 ** (float(dc_.get("gain_db", 0.0)) / 20)
        del D, exc

    e_ref = float(np.mean(np.sum(rev ** 2, axis=1)))
    if e_ref <= 0:
        e_ref = 1.0

    def add_layer(x, gain_db):
        e = float(np.mean(np.sum(x ** 2, axis=1)))
        return x * np.sqrt(e_ref * 10 ** (gain_db / 10) / e) if e > 0 else x

    # ---------------- shell layer ----------------
    if Fs_.size:
        Sh = synth_modes(n, fs, i_emit, Fs_, ch_gain(As_, r_ch), Ts_, cfg["motion"], trim, r_mot) * pre
        rev += add_layer(Sh, float(shc.get("gain_db", -10.0)))
        del Sh

    # ---------------- echo train ----------------
    ec = cfg["echoes"]
    if ec.get("enabled"):
        per = ec.get("period", "auto")
        if per == "note":
            Tp = 1.0 / root
            while Tp < 0.001:
                Tp *= 2
        elif per == "ms":
            Tp = float(ec.get("period_ms", 25.0)) / 1000
        else:
            Tp = g["flutter"] / c
        Tp = float(np.clip(Tp, 1.0 / fs * 4, 2.0))
        E = np.zeros((C, n))
        for ch in range(C):
            Tc = Tp * (1 + 0.004 * width * (2 * ch - (C - 1))) if C > 1 else Tp
            E[ch, i0:] = echo_train(n - i0, fs, Tc, t60_fn, float(ec.get("dispersion", 0.0)))
        rev += add_layer(E, float(ec.get("gain_db", -6.0)))
        info["echo_period_ms"] = round(Tp * 1000, 3)
        del E

    # ---------------- imported IR ----------------
    if imp is not None:
        X = np.zeros((C, n))
        y = imp if imp.shape[0] == C else (np.repeat(imp[:1], C, 0) if imp.shape[0] < C else imp.mean(0, keepdims=True))
        m_ = min(n - i0, y.shape[1])
        X[:, i0:i0 + m_] = y[:, :m_]
        rev += add_layer(X, float(imc.get("gain_db", 0.0)))
        del X

    # ---------------- tonal imprint ----------------
    ic = cfg["imprint"]
    ig = float(ic.get("gain_db", 0.0) or 0.0)
    if ig > 0:
        pf, _ = scale_grid(root, cents, period, max(25.0, grid_lo), min(grid_hi, 0.45 * fs))
        if pf.size:
            pf = pf[:: max(1, pf.size // 240 + 1)]
            sos = np.vstack([biquad("peak", f_, fs, ig, float(ic.get("q", 40.0))) for f_ in pf])
            rev = signal.sosfilt(sos, rev, axis=-1)
            info["imprint_peaks"] = int(pf.size)

    # ---------------- envelope ----------------
    en = cfg["envelope"]
    tt = (np.arange(n) - i0) / fs
    bloom = float(en.get("bloom_ms", 0) or 0) / 1000
    if bloom > 0:
        rev *= np.sin(0.5 * np.pi * np.clip(tt / bloom, 0, 1)) ** 2
    if gate > 0:
        rev *= np.cos(0.5 * np.pi * np.clip((tt - gate) / 0.015, 0, 1)) ** 2

    ir = rev
    dcfg = cfg["direct"]
    if dcfg.get("enabled", True):
        e_rev = float(np.mean(np.sum(rev ** 2, axis=1)))
        ir[:, i0] += np.sqrt(max(e_rev, 1e-30) * 10 ** (float(dcfg.get("level_db", -6.0)) / 10))
    if en.get("reverse"):
        thr = np.max(np.abs(ir)) * 10 ** (-trim / 20)
        last = int(np.flatnonzero(np.max(np.abs(ir), axis=0) > thr)[-1]) + 1
        body = ir[:, i0:last][:, ::-1]
        ir = np.zeros((C, n))
        ir[:, i0:i0 + body.shape[1]] = body
    ir = apply_eq(ir, fs, cfg.get("eq"))
    ir = dc_block(ir, fs, oc.get("dc_highpass_hz", 15.0))
    fo = int(float(oc.get("fade_out_ms", 60.0)) * fs / 1000)
    if 0 < fo < n:
        ir[:, -fo:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fo)))
    norm = oc.get("normalize", "peak")   # one common gain for all channels
    if norm == "energy":   # unit mean channel energy: equal loudness across IR sets
        gn = 1.0 / np.sqrt(max(float(np.mean(np.sum(ir ** 2, axis=1))), 1e-24))
    elif norm == "none":
        gn = 1.0
    else:
        gn = 10 ** (float(oc.get("peak_dbfs", -1.0)) / 20) / max(np.max(np.abs(ir)), 1e-12)
    ir *= gn
    info["normalize_gain_db"] = round(20 * np.log10(gn), 2)
    info["direct_index"] = i0

    # ---------------- display data ----------------
    allf = np.concatenate([F, Fs_])
    alle = np.concatenate([A ** 2 * T, (As_ ** 2 * Ts_) * (np.sum(A ** 2 * T) / max(np.sum(As_ ** 2 * Ts_), 1e-30)
                                                         * 10 ** (float(shc.get("gain_db", -10)) / 10) if As_.size else 1)])
    allk = np.concatenate([Kd, np.full(Fs_.size, 2)])
    allt = np.concatenate([T, Ts_])
    disp = {"modes": [], "grid": [], "crossover_hz": float(f_x) if f_x < f_top else None}
    if allf.size:
        lvl = 10 * np.log10(np.maximum(alle, 1e-30) / max(alle.max(), 1e-30))
        top = np.argsort(-lvl)[:1500]
        disp["modes"] = [[round(float(allf[i]), 2), round(float(max(lvl[i], -90)), 1), round(float(allt[i]), 3), int(allk[i])] for i in top]
    if tmode != "off" or ig > 0:
        gf_, gd_ = scale_grid(root, cents, period, 20.0, min(20000.0, 0.45 * fs))
        if gf_.size <= 400:
            disp["grid"] = [[round(float(a), 2), int(b)] for a, b in zip(gf_, gd_)]
        info["root_hz"] = round(root, 3)
    return (ir[0] if C == 1 else ir), fs, info, cfg, disp


def render_strange(preset, outdir, name=None, imported=None, quiet=True, analysis=True):
    """Render, write wav + json; with analysis also the analysis report. Returns a summary dict."""
    from .export import write_json, write_wav

    t0 = time.time()
    ir, fs, info, cfg, disp = generate_strange(preset, imported)
    dt = time.time() - t0
    base = name or cfg.get("name") or "strange"
    os.makedirs(outdir, exist_ok=True)
    oc = cfg["output"]
    wav = os.path.join(outdir, f"{base}.wav")
    write_wav(wav, ir if ir.ndim == 1 else ir.T, fs, oc.get("format", "PCM_24"), oc.get("dither", True), cfg.get("seed", 1))
    info["render_seconds"] = round(dt, 2)
    jpath = os.path.join(outdir, f"{base}.json")
    write_json(jpath, {"info": info, "resolved_preset": cfg})
    res = {"ir": ir, "fs": fs, "info": info, "cfg": cfg, "analysis": None, "table": "", "display": disp,
           "outdir": outdir, "base": base, "files": {"wav": wav, "report": None, "json": jpath}}
    if analysis:
        res.update(analyze_strange(ir, fs, info, cfg, outdir, base))
        res["files"]["report"] = res["report"]
    if not quiet:
        if res["table"]:
            print(res["table"])
        print(f"-> {wav}  ({ir.shape[-1] / fs:.2f} s, rendered in {dt:.1f} s)")
    return res


def analyze_strange(ir, fs, info, cfg, outdir, base):
    """Analysis + report PNG of a rendered strange IR (channel 1); adds it to <base>.json."""
    from .analysis import analyze, plot_report, summary_table, to_json
    from .export import write_json

    ref = ir if ir.ndim == 1 else ir[0]
    res = analyze(ref, fs)
    rep = os.path.join(outdir, f"{base}_report.png")
    plot_report(ref, fs, res, rep, title=f"{base}  (strange space, {fs} Hz, {ref.size / fs:.2f} s)",
                target_t60=(ac.OCTAVE_BANDS, list(info["t60_bands"].values())))
    write_json(os.path.join(outdir, f"{base}.json"), {"info": info, "resolved_preset": cfg, "analysis": to_json(res)})
    return {"analysis": to_json(res), "table": summary_table(res), "report": rep}
