"""Preset schema with defaults. Every key can be overridden from a JSON preset."""

from __future__ import annotations

import copy
import json

DEFAULTS = {
    "name": "untitled",
    "seed": 1,
    "output": {
        "sample_rate": 48000,
        "length_s": "auto",          # seconds or "auto" (decay to -trim_db)
        "trim_db": 90.0,             # auto length: stop when tail is this far below its start
        "max_length_s": 30.0,
        "lead_ms": 0.7,              # silence before the direct sound (room for sinc interpolation)
        "align_to_direct": True,     # room mode: remove propagation delay before direct sound
        "fade_out_ms": 30.0,
        "dc_highpass_hz": 8.0,
        "normalize": "peak",         # "peak" | "energy" | "none"
        "peak_dbfs": -1.0,
        "format": "PCM_24",          # PCM_16 | PCM_24 | PCM_32 | FLOAT | DOUBLE
        "dither": True,
        "write_report": True,
    },
    # Physical room (used by any component whose source is "room")
    "room": {
        "dimensions": [12.0, 8.0, 4.0],
        "materials": {"default": "plaster", "floor": "wood_floor"},
        "scattering": 0.3,           # number | 9 octave values | {surface: ...}
        "source": {"position": [3.0, 4.0, 1.5], "directivity": "omni", "orientation": [0, 0]},
        "receiver": {"position": [8.0, 3.2, 1.6], "directivity": "omni", "orientation": [180, 0]},
        "air": {"temperature_c": 20.0, "humidity_pct": 50.0, "pressure_kpa": 101.325},
        "reverb_formula": "eyring",  # "eyring" | "sabine"
        "max_order": "auto",
        "image_jitter_m": 0.04,      # randomised image positions (reduces sweeping echoes)
        "modal": {
            "enabled": "auto",       # true | false | "auto" (on when Schroeder freq > 60 Hz)
            "crossover_hz": "auto",  # "auto" = Schroeder frequency, clamped 50..400 Hz
            "detune_cents": 0.0,
            "gain_db": 0.0,
        },
    },
    "direct": {
        "enabled": True,
        "gain_db": 0.0,
        "tone": None,                # frequency curve in dB (see README)
    },
    "early": {
        "source": "room",            # "room" | "parametric" | "list" | "none"
        "gain_db": 0.0,
        "tone": None,
        "crossover_ms": "auto",      # room mode: image model -> statistical tail (default: mixing time)
        "crossover_width_ms": "auto",
        "parametric": {
            "count": 24,
            "start_ms": 6.0,
            "end_ms": 70.0,
            "distribution": "power", # "power" | "regular"
            "density_exponent": 2.0, # 0 uniform, 2 = Polack t^2 growth
            "level_start_db": -5.0,
            "level_end_db": -22.0,
            "level_curve": 1.0,
            "level_jitter_db": 2.0,
            "polarity": "random",
            "color_from_decay": True,
            "hf_damping_db_per_oct": 0.0,
            "absorption": None,      # material name or 9 values: per-bounce absorption colouring
            "bounces_end": 3.0,
            "cluster_size": 1,
            "cluster_spread_ms": 1.5,
            "air": True,
        },
        "list": [],
    },
    "late": {
        "source": "room",            # "room" | "parametric" | "none"
        "t60": None,                 # override: number, 9 octave values, {freqs, values} or {mid, low_ratio, high_ratio, ...}
        "t60_scale": 1.0,
        "calibrate": 3,              # iterations matching measured octave T30 to the target (0 = off)
        "gain_db": 0.0,
        "tone": None,                # dB curve on the tail's level spectrum
        "drr_db": None,              # parametric: direct-to-late energy ratio (dB); room: optional override
        "onset_ms": "auto",          # parametric: tail start re direct (auto = start of early-reflection window)
        "attack_ms": 15.0,           # parametric: tail fade-in duration
        "attack_shape": 1.0,         # >1 slower bloom, <1 faster
        "noise": "auto",             # "gaussian" | "velvet" | "auto" (velvet in parametric mode)
        "density": {"initial_per_s": 300.0, "max_per_s": "auto", "full_ms": "auto", "exponent": 2.0,
                    "pulse_amplitude": "gaussian"},  # max auto = fs/4; pulse_amplitude gaussian | uniform
        "double_slope": {"enabled": False, "t60": None, "t60_ratio": 2.5, "level_db": -18.0},
        "bands_per_octave": 3,
        "envelope_flatten": True,     # remove slow random level fluctuations of the noise (better T/EDT match)
        "flatten_cycles": 8.0,        # smoothing window = cycles / bandwidth
    },
    # 1-64 statistically decorrelated, perceptually similar outputs
    "multichannel": {
        "channels": 1,
        "coherence": {"model": "independent"},   # independent | constant | diffuse | diffuse_2d | custom
                                                 # constant: "value" (number or curve); custom: "matrix";
                                                 # any model: "coherent_blend" 0..1 (curve) -> e.g. mono bass
        "array": {"type": "none"},               # none | line | circle | sphere | grid | random | custom
        "early": "jitter",                       # same | jitter | independent
        "direct": "auto",                        # auto | same | per_channel
        "receiver_jitter_m": 0.15,               # room mode, no array: random mic displacement per channel
        "time_jitter_ms": 0.4,                   # parametric/list reflections, 'jitter' mode
        "gain_jitter_db": 1.0,
        "export": "interleaved",                 # interleaved | separate | both
    },
    "eq": [],                         # global causal EQ
    "eq_late": [],                    # causal EQ on the late tail only
}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_preset(path_or_dict):
    if isinstance(path_or_dict, dict):
        user = path_or_dict
    else:
        with open(path_or_dict) as fh:
            user = json.load(fh)
    user = {k: v for k, v in user.items() if not k.startswith("_")}
    cfg = deep_merge(DEFAULTS, user)
    mode = user.get("mode")
    if mode == "perceptual":
        # perceptual default: no room needed
        cfg["early"]["source"] = user.get("early", {}).get("source", "parametric")
        cfg["late"]["source"] = user.get("late", {}).get("source", "parametric")
    return cfg
