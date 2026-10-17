"""Multichannel helpers: receiver arrays, target coherence models and mixing matrices."""

from __future__ import annotations

import numpy as np
from scipy.special import j0

from . import acoustics as ac


def array_offsets(acfg, n):
    """Receiver offsets (n, 3) in metres relative to the main receiver position,
    plus per-channel orientation vectors (n, 3) or None (= keep main orientation)."""
    acfg = acfg or {}
    kind = acfg.get("type", "none")
    if n == 1 or kind == "none":
        return np.zeros((n, 3)), None
    if kind == "line":
        d = float(acfg.get("spacing_m", 0.2))
        axis = ac.orientation_vector(acfg.get("axis", [90, 0]))
        k = np.arange(n) - (n - 1) / 2
        off = k[:, None] * d * axis[None, :]
    elif kind == "circle":
        r = float(acfg.get("radius_m", 0.3))
        a = 2 * np.pi * np.arange(n) / n + np.deg2rad(acfg.get("rotation_deg", 0.0))
        off = np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], axis=1)
    elif kind == "sphere":
        r = float(acfg.get("radius_m", 0.3))
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        th = np.pi * (1 + 5 ** 0.5) * i
        off = r * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)
    elif kind == "grid":
        d = float(acfg.get("spacing_m", 0.2))
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        gx, gy = np.meshgrid(np.arange(cols) - (cols - 1) / 2, np.arange(rows) - (rows - 1) / 2)
        off = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)[:n] * d
    elif kind == "random":
        r = float(acfg.get("radius_m", 0.5))
        rng = np.random.default_rng(int(acfg.get("seed", 0)))
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        off = v * r * rng.random((n, 1)) ** (1 / 3)
    elif kind == "custom":
        off = np.asarray(acfg["positions"], float)
        if off.shape != (n, 3):
            raise ValueError(f"multichannel.array.positions must be {n} x 3")
    else:
        raise ValueError("array.type must be none | line | circle | sphere | grid | random | custom")
    orient = None
    if acfg.get("orientation", "same") == "outward":
        nr = np.linalg.norm(off, axis=1, keepdims=True)
        orient = np.where(nr > 1e-9, off / np.maximum(nr, 1e-9), np.array([[1.0, 0, 0]]))
    return off, orient


def coherence_matrix(ccfg, fc, n, offsets, c=343.0):
    """Target coherence matrix (n x n) at frequency fc."""
    ccfg = ccfg or {}
    model = ccfg.get("model", "independent")
    if model == "independent":
        G = np.eye(n)
    elif model == "constant":
        rho = float(ac.curve_from_spec(ccfg.get("value", 0.0), np.array([fc]))[0])
        rho = np.clip(rho, -1.0 / max(n - 1, 1) + 1e-6, 1.0)
        G = (1 - rho) * np.eye(n) + rho * np.ones((n, n))
    elif model in ("diffuse", "diffuse_3d", "diffuse_2d"):
        d = np.linalg.norm(offsets[:, None, :] - offsets[None, :, :], axis=2)
        if model == "diffuse_2d":
            G = j0(2 * np.pi * fc * d / c)
        else:
            G = np.sinc(2 * fc * d / c)
    elif model == "custom":
        G = np.asarray(ccfg["matrix"], float)
        if G.shape != (n, n):
            raise ValueError(f"multichannel.coherence.matrix must be {n} x {n}")
    else:
        raise ValueError("coherence.model must be independent | constant | diffuse | diffuse_2d | custom")
    blend = ccfg.get("coherent_blend")  # 0..1 (curve allowed): push towards full coherence, e.g. mono bass
    if blend is not None:
        b = float(np.clip(ac.curve_from_spec(blend, np.array([fc]))[0], 0, 1))
        G = (1 - b) * G + b * np.ones((n, n))
    np.fill_diagonal(G, 1.0)
    return G


def mixing_matrix(G):
    """Symmetric square root of the (PSD-projected) coherence matrix, rows normalised to unit power."""
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    w = np.maximum(w, 0.0)
    M = (V * np.sqrt(w)) @ V.T
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    return M
