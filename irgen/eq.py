"""Causal EQ (RBJ biquads), Linkwitz-Riley crossovers and helpers."""

from __future__ import annotations

import numpy as np
from scipy import signal


def biquad(kind, f0, fs, gain_db=0.0, q=0.7071):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    if kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    elif kind == "highshelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    elif kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "notch":
        b = [1, -2 * cw, 1]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(f"Unknown EQ type '{kind}'")
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return np.concatenate([b, a])[None, :]


def apply_eq(x, fs, bands):
    """bands: list of {type, freq, gain_db?, q?, order?}. lowpass/highpass with
    'order' > 2 are Butterworth."""
    if not bands:
        return x
    for bd in bands:
        kind = bd["type"]
        f0 = float(bd["freq"])
        if f0 >= fs / 2:
            continue
        order = int(bd.get("order", 2))
        if kind in ("lowpass", "highpass") and order != 2:
            sos = signal.butter(order, f0, btype="low" if kind == "lowpass" else "high", fs=fs, output="sos")
        else:
            sos = biquad(kind, f0, fs, bd.get("gain_db", 0.0), bd.get("q", 0.7071))
        x = signal.sosfilt(sos, x)
    return x


def lr4(fs, fc):
    lp = signal.butter(2, fc, "low", fs=fs, output="sos")
    hp = signal.butter(2, fc, "high", fs=fs, output="sos")
    return np.vstack([lp, lp]), np.vstack([hp, hp])


def dc_block(x, fs, fc):
    if not fc or fc <= 0:
        return x
    sos = signal.butter(2, fc, "high", fs=fs, output="sos")
    return signal.sosfilt(sos, x)
