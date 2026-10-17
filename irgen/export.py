"""WAV export with optional TPDF dither."""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf

_BITS = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}


def write_wav(path, ir, fs, fmt="PCM_24", dither=True, seed=0):
    x = np.asarray(ir, dtype=np.float64)
    if fmt in _BITS and dither and _BITS[fmt] < 32:
        lsb = 2.0 ** -(_BITS[fmt] - 1)
        rng = np.random.default_rng(seed)
        x = x + (rng.random(x.shape) - rng.random(x.shape)) * lsb
    if fmt in _BITS:
        x = np.clip(x, -1.0, 1.0 - 2.0 ** -(_BITS[fmt] - 1))
    sf.write(path, x, fs, subtype=fmt)


def write_json(path, obj):
    def conv(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=conv)
