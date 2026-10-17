"""Late reverberation: excitation noise (Gaussian or velvet with growing echo density),
a perfect-reconstruction fractional-octave filterbank for frequency-dependent decay,
and low-frequency modal synthesis for rectangular rooms."""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# Excitation
# --------------------------------------------------------------------------------------
def velvet_growing(n, fs, onset_idx, rng, d0=200.0, dmax=4000.0, t_full=0.08, k=2.0, xfade_s=0.02, amp="gaussian"):
    """Unit-power excitation whose echo density grows from d0 to dmax pulses/s
    (density ~ t^k, Polack: k=2) over t_full seconds after onset, then crossfades
    to Gaussian noise. Before onset the signal is zero."""
    x = np.zeros(n)
    m = n - onset_idx
    if m <= 0:
        return x
    t = np.arange(m) / fs
    dens = d0 + (dmax - d0) * np.minimum(1.0, t / max(t_full, 1e-6)) ** k
    lam = np.cumsum(dens / fs)                      # expected pulse count
    seg = np.floor(lam).astype(np.int64)
    starts = np.flatnonzero(np.diff(np.concatenate([[-1], seg])))  # first sample of each segment
    ends = np.concatenate([starts[1:], [m]])
    lens = ends - starts
    pos = starts + np.floor(rng.random(starts.size) * lens).astype(np.int64)
    if amp == "uniform":   # classic velvet: +-1 pulses
        a = np.where(rng.random(starts.size) < 0.5, -1.0, 1.0)
    else:                  # Gaussian pulse heights: more natural reflection statistics
        a = rng.standard_normal(starts.size)
    v = np.zeros(m)
    v[pos] = a * np.sqrt(lens)                      # unit mean power per sample
    g = rng.standard_normal(m)
    # equal-power crossfade velvet -> gaussian around t_full
    xf = np.clip((t - t_full) / max(xfade_s, 1e-6) + 0.5, 0.0, 1.0)
    w_g = np.sin(0.5 * np.pi * xf)
    w_v = np.cos(0.5 * np.pi * xf)
    x[onset_idx:] = w_v * v + w_g * g
    return x


def gaussian(n, onset_idx, rng):
    x = np.zeros(n)
    x[onset_idx:] = rng.standard_normal(n - onset_idx)
    return x


# --------------------------------------------------------------------------------------
# Filterbank
# --------------------------------------------------------------------------------------
def band_centers(fs, bands_per_octave=3, f_lo=20.0):
    nyq = fs / 2.0
    n = int(np.floor(np.log2(nyq * 0.98 / f_lo) * bands_per_octave)) + 1
    return f_lo * 2.0 ** (np.arange(n) / bands_per_octave)


def band_masks(f, centers):
    """cos^2 masks on log-frequency; they sum exactly to 1 at every frequency."""
    u = np.log2(np.maximum(f, 1e-3))
    uc = np.log2(centers)
    du = uc[1] - uc[0]
    masks = np.zeros((centers.size, f.size))
    for i in range(centers.size):
        d = (u - uc[i]) / du
        m = np.where(np.abs(d) < 1.0, np.cos(0.5 * np.pi * d) ** 2, 0.0)
        if i == 0:
            m = np.where(u <= uc[0], 1.0, m)
        if i == centers.size - 1:
            m = np.where(u >= uc[-1], 1.0, m)
        masks[i] = m
    return masks


def _flatten(band, fs, fc, bpo, onset_idx, cycles):
    """Remove slow random power fluctuations of a narrow-band noise (keeps the fine
    structure random). Makes decay parameters of different seeds / channels match."""
    from scipy.ndimage import uniform_filter1d
    B = fc * (2 ** (0.5 / bpo) - 2 ** (-0.5 / bpo))
    L = int(np.clip(cycles / B, 0.005, 0.4) * fs)
    if L < 3:
        return band
    act = np.zeros(band.shape[-1])
    act[onset_idx:] = 1.0
    sm_a = uniform_filter1d(uniform_filter1d(act, L), L)
    p = uniform_filter1d(uniform_filter1d(band.astype(np.float64) ** 2, L, axis=-1), L, axis=-1)
    p = p / np.maximum(sm_a, 1e-6)
    ref = np.mean(p[..., onset_idx:], axis=-1, keepdims=True) if band.ndim > 1 else np.mean(p[onset_idx:])
    g = np.sqrt(ref / np.maximum(p, 1e-30))
    return band * np.clip(g, 0.5, 2.0)


def shape_noise(noise, fs, centers, env_fn, flatten=None):
    """Split noise into bands (zero-phase masks), multiply band i by env_fn(i, fc)
    (amplitude envelope, length n) and sum. flatten = (bands_per_octave, onset_idx, cycles)."""
    n = noise.size
    nfft = int(2 ** np.ceil(np.log2(n)))
    X = np.fft.rfft(noise, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    masks = band_masks(f, centers)
    out = np.zeros(n)
    for i, fc in enumerate(centers):
        env = env_fn(i, fc)
        if env is None:
            continue
        band = np.fft.irfft(X * masks[i], nfft)[:n]
        if flatten:
            band = _flatten(band, fs, fc, *flatten)
        out += band * env
    return out


def shape_noise_multi(noises, fs, centers, env_fn, mix_fn=None, flatten=None):
    """Multichannel version: noises (N, n) independent excitations. In each band the
    channels are mixed by mix_fn(fc) -> (N, N) so that their correlation matches the
    target coherence, then all channels get the same envelope (perceptual similarity)."""
    N, n = noises.shape
    nfft = int(2 ** np.ceil(np.log2(n)))
    dt = np.float32 if N > 8 else np.float64
    X = np.fft.rfft(noises, nfft, axis=1)
    if dt is np.float32:
        X = X.astype(np.complex64)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    masks = band_masks(f, centers)
    out = np.zeros((N, n), dtype=dt)
    for i, fc in enumerate(centers):
        env = env_fn(i, fc)
        if env is None:
            continue
        Xb = X * masks[i][None, :].astype(X.real.dtype)
        if mix_fn is not None:
            M = mix_fn(fc).astype(X.real.dtype)
            Xb = M @ Xb
        band = np.fft.irfft(Xb, nfft, axis=1)[:, :n]
        if flatten:
            band = _flatten(band, fs, fc, *flatten)
        out += (band * env[None, :]).astype(dt)
    return out.astype(np.float64)


# --------------------------------------------------------------------------------------
# Modal synthesis (rectangular room, below the Schroeder frequency)
# --------------------------------------------------------------------------------------
def shoebox_modes(dims, src, rcv, c, f_max):
    Lx, Ly, Lz = dims
    nx = np.arange(int(2 * f_max * Lx / c) + 2)
    ny = np.arange(int(2 * f_max * Ly / c) + 2)
    nz = np.arange(int(2 * f_max * Lz / c) + 2)
    NX, NY, NZ = np.meshgrid(nx, ny, nz, indexing="ij")
    NX, NY, NZ = NX.ravel(), NY.ravel(), NZ.ravel()
    f = 0.5 * c * np.sqrt((NX / Lx) ** 2 + (NY / Ly) ** 2 + (NZ / Lz) ** 2)
    sel = (f > 0) & (f <= f_max)
    NX, NY, NZ, f = NX[sel], NY[sel], NZ[sel], f[sel]

    def psi(p):
        return (np.cos(np.pi * NX * p[0] / Lx) * np.cos(np.pi * NY * p[1] / Ly) * np.cos(np.pi * NZ * p[2] / Lz))

    lam = np.where(NX > 0, 0.5, 1.0) * np.where(NY > 0, 0.5, 1.0) * np.where(NZ > 0, 0.5, 1.0)
    coupling = psi(src) * psi(rcv) / lam
    return f, coupling


def modal_ir(n, fs, t_emit0, dims, src, rcv, c, f_max, t60_fn, volume, rng=None, detune_cents=0.0):
    """Discrete-time modal impulse response on the same scale as the image model
    (direct sound amplitude = 1/r).  h[n] = (4*pi*c^2 / (fs*V)) * sum psi psi/Lambda *
    sin(w t)/w * exp(-delta t),  t = emission time."""
    f, coup = shoebox_modes(dims, src, rcv, c, f_max)
    if f.size == 0:
        return np.zeros(n), 0
    if detune_cents and rng is not None:
        f = f * 2 ** (rng.normal(0, detune_cents, f.size) / 1200.0)
    w = 2 * np.pi * f
    delta = 6.9078 / t60_fn(f)
    scale = 4 * np.pi * c * c / (fs * volume)
    amp = scale * coup / w * np.exp(-delta * t_emit0)
    # Evaluate the exact DTFT of each damped sinusoid on the FFT grid (only the
    # low-frequency bins are needed; the result is low-passed afterwards).
    nfft = int(2 ** np.ceil(np.log2(n)))
    fb = np.fft.rfftfreq(nfft, 1.0 / fs)
    kmax = int(np.searchsorted(fb, 1.6 * f_max))
    Om = 2 * np.pi * fb[:kmax] / fs
    e = np.exp(-1j * Om)[None, :]
    X = np.zeros(kmax, dtype=complex)
    for s in range(0, f.size, 256):
        sl = slice(s, s + 256)
        p = np.exp((-delta[sl] + 1j * w[sl]) / fs)[:, None]
        ph = np.exp(1j * w[sl] * t_emit0)[:, None]
        Xm = (ph / (1 - p * e) - np.conj(ph) / (1 - np.conj(p) * e)) / 2j
        X += np.sum(amp[sl, None] * Xm, axis=0)
    taper = np.ones(kmax)
    k0 = int(np.searchsorted(fb, 1.2 * f_max))
    taper[k0:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, kmax - k0)))
    full = np.zeros(fb.size, dtype=complex)
    full[:kmax] = X * taper
    h = np.fft.irfft(full, nfft)[:n]
    return h, f.size
