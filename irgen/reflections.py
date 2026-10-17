"""Discrete reflections: high-quality renderer, shoebox image-source model and a
parametric early-reflection generator."""

from __future__ import annotations

import numpy as np

from .acoustics import (
    OCTAVE_BANDS, air_absorption_db_per_m, directivity_gain, interp_curve,
)


# ======================================================================================
# Renderer
# ======================================================================================
class ReflectionRenderer:
    """Renders each reflection as  gain x minimum-phase filter (its own frequency
    response) x band-limited fractional delay (Kaiser-windowed sinc).

    Every reflection gets its own causal filter, so absorption / air / tone are
    applied without pre-ringing, and arrival times are sub-sample accurate."""

    def __init__(self, fs, air=None, nfft=None, sinc_half=32, kaiser_beta=9.0, batch=1024):
        self.fs = fs
        if nfft is None:
            nfft = int(2 ** np.ceil(np.log2(max(512, 0.025 * fs))))
        self.nfft = nfft
        self.K = sinc_half
        self.beta = kaiser_beta
        self.batch = batch
        self.f = np.fft.rfftfreq(nfft, 1.0 / fs)
        # interpolation matrix octave-band dB -> FFT grid (log-frequency linear)
        eye = np.eye(len(OCTAVE_BANDS))
        self.W = np.stack([interp_curve(OCTAVE_BANDS, eye[i], self.f) for i in range(len(OCTAVE_BANDS))], axis=1)
        air = air or {}
        self.air_db_per_m = air_absorption_db_per_m(
            self.f, air.get("temperature_c", 20.0), air.get("humidity_pct", 50.0), air.get("pressure_kpa", 101.325)
        )
        # cepstral folding window for minimum phase
        w = np.zeros(nfft)
        w[0] = 1.0
        w[1:nfft // 2] = 2.0
        w[nfft // 2] = 1.0
        self.fold = w
        self.L = nfft - (2 * sinc_half + 1)  # truncated min-phase length (no circular wrap)
        fade = int(self.L * 0.25)
        self.win = np.ones(self.L)
        self.win[-fade:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade)))
        self.kwin = np.kaiser(2 * sinc_half + 1, kaiser_beta)

    def render(self, out, samples, amps, band_db=None, air_dist=None, tone_db=None, use_air=True):
        """Add reflections into `out`.

        samples : (B,) fractional sample positions in the output buffer
        amps    : (B,) signed broadband gains
        band_db : (B, 9) octave-band gain in dB (absorption etc.), optional
        air_dist: (B,) propagation distance in metres for air absorption, optional
        tone_db : (F,) or None, extra common response on the FFT grid
        """
        samples = np.asarray(samples, float)
        amps = np.asarray(amps, float)
        n_out = out.shape[0]
        keep = (samples > -self.K) & (samples < n_out) & (np.abs(amps) > 0)
        samples, amps = samples[keep], amps[keep]
        if band_db is not None:
            band_db = np.asarray(band_db)[keep]
        if air_dist is not None:
            air_dist = np.asarray(air_dist)[keep]
        B = samples.size
        nf = self.f.size
        jj = np.arange(2 * self.K + 1)
        for s0 in range(0, B, self.batch):
            sl = slice(s0, min(B, s0 + self.batch))
            b = sl.stop - sl.start
            mag_db = np.zeros((b, nf))
            if band_db is not None:
                mag_db += band_db[sl] @ self.W.T
            if use_air and air_dist is not None:
                mag_db -= air_dist[sl, None] * self.air_db_per_m[None, :]
            if tone_db is not None:
                mag_db += tone_db[None, :]
            pos = samples[sl]
            n0 = np.floor(pos).astype(np.int64)
            frac = pos - n0
            # fractional-delay kernel centred at K + frac
            x = jj[None, :] - self.K - frac[:, None]
            kern = np.sinc(x) * self.kwin[None, :]
            if not np.any(mag_db):
                h = np.zeros((b, self.nfft))
                h[:, : 2 * self.K + 1] = kern
            else:
                logmag = np.clip(mag_db, -200, 60) * (np.log(10.0) / 20.0)
                cep = np.fft.irfft(logmag, n=self.nfft, axis=1)
                hmin = np.fft.irfft(np.exp(np.fft.rfft(cep * self.fold[None, :], axis=1)), n=self.nfft, axis=1)
                hmin = hmin[:, : self.L] * self.win[None, :]
                H = np.fft.rfft(hmin, n=self.nfft, axis=1) * np.fft.rfft(kern, n=self.nfft, axis=1)
                h = np.fft.irfft(H, n=self.nfft, axis=1)
            h *= amps[sl, None]
            idx = (n0 - self.K)[:, None] + np.arange(self.nfft)[None, :]
            valid = (idx >= 0) & (idx < n_out)
            out += np.bincount(idx[valid], weights=h[valid], minlength=n_out)[:n_out]
        return B


# ======================================================================================
# Shoebox image-source model
# ======================================================================================
def shoebox_images(dims, src, rcv, beta_db, c, t_max, max_order=None, src_dir=None,
                   rcv_dir=None, jitter_m=0.0, rng=None):
    """Enumerate image sources of a rectangular room.

    beta_db : (6, 9) specular reflection gain per surface (dB, <= 0)
              surface order: west(x=0), east(x=Lx), south(y=0), north(y=Ly), floor, ceiling
    Returns dict with arrays: time (emission->arrival, s), dist, amp (1/r * directivity),
    band_db (B,9), order."""
    dims = np.asarray(dims, float)
    src = np.asarray(src, float)
    rcv = np.asarray(rcv, float)
    d_max = c * t_max
    if max_order is None:
        max_order = int(np.ceil(d_max / dims.min())) + 1
    N = int(max_order)
    m = np.arange(-N, N + 1)
    out = {k: [] for k in ("time", "dist", "amp", "band_db", "order", "counts")}
    for px in (0, 1):
        for py in (0, 1):
            for pz in (0, 1):
                p = np.array([px, py, pz])
                # bound each axis by the distance limit to keep memory small
                mx, my, mz = np.meshgrid(m, m, m, indexing="ij")
                mx, my, mz = mx.ravel(), my.ravel(), mz.ravel()
                cx = np.abs(mx - px) + np.abs(mx)
                cy = np.abs(my - py) + np.abs(my)
                cz = np.abs(mz - pz) + np.abs(mz)
                order = cx + cy + cz
                sel = order <= N
                mx, my, mz, order = mx[sel], my[sel], mz[sel], order[sel]
                img = np.stack([
                    (1 - 2 * px) * src[0] + 2 * mx * dims[0],
                    (1 - 2 * py) * src[1] + 2 * my * dims[1],
                    (1 - 2 * pz) * src[2] + 2 * mz * dims[2],
                ], axis=1)
                if jitter_m > 0 and rng is not None:
                    jit = rng.normal(0.0, jitter_m, img.shape) * (order[:, None] > 0)
                    img = img + jit
                vec = img - rcv[None, :]
                dist = np.linalg.norm(vec, axis=1)
                sel = dist <= d_max
                if not np.any(sel):
                    continue
                mx, my, mz, order, vec, dist = mx[sel], my[sel], mz[sel], order[sel], vec[sel], dist[sel]
                counts = np.stack([
                    np.abs(mx - px), np.abs(mx), np.abs(my - py), np.abs(my), np.abs(mz - pz), np.abs(mz)
                ], axis=1).astype(float)
                band_db = counts @ beta_db  # (B, 9)
                u_arr = vec / np.maximum(dist[:, None], 1e-9)  # direction sound arrives from
                amp = 1.0 / np.maximum(dist, 1e-3)
                if rcv_dir is not None:
                    amp = amp * directivity_gain(rcv_dir[0], rcv_dir[1], u_arr)
                if src_dir is not None:
                    emit = -u_arr * (1 - 2 * p)[None, :]  # mirrored direction leaving the real source
                    amp = amp * directivity_gain(src_dir[0], src_dir[1], emit)
                out["time"].append(dist / c)
                out["dist"].append(dist)
                out["amp"].append(amp)
                out["band_db"].append(band_db)
                out["order"].append(order)
                out["counts"].append(counts)
    res = {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}
    if res["band_db"].ndim == 1:
        res["band_db"] = res["band_db"].reshape(0, len(OCTAVE_BANDS))
        res["counts"] = res["counts"].reshape(0, 6)
    o = np.argsort(res["time"])
    return {k: v[o] for k, v in res.items()}, N


# ======================================================================================
# Parametric early reflections
# ======================================================================================
def parametric_early(cfg, t60_bands, rng, c=343.0):
    """Generate reflection times (relative to direct, s), signed gains (re direct),
    band_db and air distances from perceptual parameters."""
    n = int(cfg.get("count", 24))
    if n <= 0:
        return None
    t0 = cfg.get("start_ms", 4.0) / 1000.0
    t1 = cfg.get("end_ms", 80.0) / 1000.0
    k = float(cfg.get("density_exponent", 2.0))  # 0 = uniform, 2 = Polack (density ~ t^2)
    u = np.sort(rng.random(n))
    if cfg.get("distribution", "power") == "regular":
        u = (np.arange(n) + 0.5 + cfg.get("regular_jitter", 0.3) * (rng.random(n) - 0.5)) / n
        u = np.sort(np.clip(u, 0, 1))
    times = (t0 ** (k + 1) + u * (t1 ** (k + 1) - t0 ** (k + 1))) ** (1.0 / (k + 1))
    times[0] = t0 if cfg.get("pin_first", True) else times[0]
    # level envelope (dB re direct), linear in dB across the window
    l0 = cfg.get("level_start_db", -6.0)
    l1 = cfg.get("level_end_db", -24.0)
    x = (times - t0) / max(t1 - t0, 1e-9)
    curve = float(cfg.get("level_curve", 1.0))  # 1 = linear in dB, <1 faster early drop, >1 slower
    lev = l0 + (l1 - l0) * x ** curve
    lev += rng.normal(0.0, cfg.get("level_jitter_db", 2.0), n)
    gains = 10 ** (lev / 20.0)
    pol = cfg.get("polarity", "random")
    if pol == "random":
        gains *= np.where(rng.random(n) < 0.5, -1.0, 1.0)
    elif pol == "negative":
        gains *= -1.0
    band_db = np.zeros((n, len(OCTAVE_BANDS)))
    # colour each reflection consistently with the frequency-dependent decay
    if cfg.get("color_from_decay", True) and t60_bands is not None:
        t_ref = np.interp(np.log2(1000.0), np.log2(OCTAVE_BANDS), t60_bands)
        band_db += -60.0 * times[:, None] * (1.0 / t60_bands[None, :] - 1.0 / t_ref)
    hf = cfg.get("hf_damping_db_per_oct", 0.0)  # extra per-reflection tilt above 1 kHz, grows with time
    if hf:
        oct_above = np.maximum(np.log2(OCTAVE_BANDS / 1000.0), 0.0)
        band_db += -abs(hf) * oct_above[None, :] * (0.5 + x[:, None])
    if cfg.get("absorption") is not None:
        from .acoustics import resolve_band_values
        a = np.clip(resolve_band_values(cfg["absorption"], "early.absorption"), 0, 0.999)
        bounces = 1.0 + x * float(cfg.get("bounces_end", 3.0))
        band_db += bounces[:, None] * 10 * np.log10(1 - a)[None, :]
    air_dist = times * c if cfg.get("air", True) else None
    # optional clusters (diffuse reflections split into a few smeared pulses)
    cs = int(cfg.get("cluster_size", 1))
    if cs > 1:
        spread = cfg.get("cluster_spread_ms", 1.5) / 1000.0
        offs = rng.random((n, cs)) * spread
        offs[:, 0] = 0.0
        wts = rng.normal(0, 1, (n, cs)) * np.exp(-offs / max(spread, 1e-9) * 2)
        wts[:, 0] = 1.0
        wts /= np.sqrt(np.sum(wts ** 2, axis=1, keepdims=True))
        times = (times[:, None] + offs).ravel()
        gains = (gains[:, None] * wts).ravel()
        band_db = np.repeat(band_db, cs, axis=0)
        air_dist = None if air_dist is None else np.repeat(air_dist, cs)
    return {"time": times, "gain": gains, "band_db": band_db, "air_dist": air_dist}


def list_early(entries, c=343.0, use_air=True):
    """Explicit reflection list: [{time_ms, gain_db, polarity?, tone?(9 bands or {freqs,values})}]."""
    if not entries:
        return None
    from .acoustics import curve_from_spec
    times = np.array([e["time_ms"] for e in entries]) / 1000.0
    gains = np.array([10 ** (e.get("gain_db", 0.0) / 20.0) * (-1 if e.get("polarity", 1) < 0 else 1) for e in entries])
    band_db = np.zeros((len(entries), len(OCTAVE_BANDS)))
    for i, e in enumerate(entries):
        if e.get("tone") is not None:
            band_db[i] = curve_from_spec(e["tone"], OCTAVE_BANDS)
    return {"time": times, "gain": gains, "band_db": band_db, "air_dist": times * c if use_air else None}
