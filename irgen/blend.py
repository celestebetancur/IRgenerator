"""Blend two multichannel IR folders into a new one.

Channel i of A is blended with channel i of B, so the result has min(N_A, N_B) channels.
Both inputs are resampled to A's rate if needed, aligned at their direct sound, and
scaled to the same mean channel energy, so `amount` (0 = A, 1 = B) is balanced.

mix    weighted sum of the waveforms: you hear both spaces layered. Linear weights around
       the direct sound (it is the same impulse in both, so it adds coherently), equal-power
       weights for the rest (the tails are uncorrelated), with a short crossfade between.
morph  perceptual interpolation per 1/3-octave band and channel. The decay time moves
       geometrically, T = T_A^(1-a)·T_B^a (equal steps in log T are equal perceptual
       steps: the T60 JND is relative). Each band envelope (dB) is first time-warped onto
       that decay (its shape, e.g. a double slope or a slow build-up, is kept), then the
       two are averaged: level and spectrum land in between too. Envelopes that run into
       their noise floor or end early are extrapolated along their fitted decay slope.
       The fine structure is an equal-power mix of both IRs' band signals, each re-shaped
       to the target envelope. a = 0 or 1 gives A or B back unchanged.

An IR folder can be an IR set (renders/sets/<id>, with manifest.json), a GUI render
(renders/gui/<name>, renders/gui/strange/<name>), a previous blend (renders/blends/<name>),
or any folder of WAVs: the WAV with the most channels, or else the mono WAVs sorted by name
(one per channel). Files starting with "_" (previews, auditions) are ignored.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from . import __version__

ROOT = Path(__file__).resolve().parent.parent
RENDERS = ROOT / "renders"
BLENDS = RENDERS / "blends"
MODES = ("morph", "mix")


class BlendError(ValueError):
    pass


# --------------------------------------------------------------------------------------
# folders
# --------------------------------------------------------------------------------------
def resolve_folder(spec):
    """'set/<id>' | 'blend/<name>' | 'gui/<name>' | 'strange/<name>' | a path -> Path."""
    s = str(spec or "").strip()
    if not s:
        raise BlendError("no IR folder given")
    kinds = {"set": RENDERS / "sets", "blend": BLENDS, "gui": RENDERS / "gui", "strange": RENDERS / "gui" / "strange"}
    kind, _, name = s.partition("/")
    if kind in kinds and name and "/" not in name and name not in (".", ".."):
        p = kinds[kind] / name
    elif "/" not in s and (kinds["set"] / s).is_dir():   # a bare set id
        p = kinds["set"] / s
    else:
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = ROOT / p
    if not p.is_dir():
        raise BlendError(f"IR folder '{spec}' not found")
    return p.resolve()


def _wavs(d):
    return sorted(p for p in d.glob("*.wav") if not p.name.startswith("_"))


def load_folder(path):
    """-> (ir (C, n) float64, fs, meta). meta: source, channels, direct_index (if known)."""
    import soundfile as sf
    d = resolve_folder(path) if not isinstance(path, Path) else path
    man = d / "manifest.json"
    meta = {"path": str(d), "name": d.name}
    if man.is_file():
        m = json.loads(man.read_text())
        meta.update(kind="blend" if m.get("blend") else "set", direct_index=m.get("direct_index"))
        if m.get("files"):
            chans, fs = [], None
            for f in sorted(m["files"], key=lambda f: f["channel"]):
                x, fs = sf.read(str(d / f["file"]), always_2d=True)
                chans.append(x[:, 0])
            return _stack(chans), fs, meta
        if m.get("interleaved"):
            x, fs = sf.read(str(d / m["interleaved"]["file"]), always_2d=True)
            return x.T.copy(), fs, meta
    files = _wavs(d)
    if not files and any(d.glob("*_channels")):
        files = _wavs(next(d.glob("*_channels")))
    if not files:
        raise BlendError(f"no WAV files in {d}")
    infos = [(p, sf.info(str(p))) for p in files]
    multi = max(infos, key=lambda t: t[1].channels)
    meta.setdefault("kind", "folder")
    if multi[1].channels > 1:
        x, fs = sf.read(str(multi[0]), always_2d=True)
        meta["file"] = multi[0].name
        return x.T.copy(), fs, meta
    chans, rates = [], set()
    for p, info in infos:
        x, fs = sf.read(str(p), always_2d=True)
        chans.append(x[:, 0])
        rates.add(fs)
    if len(rates) > 1:
        raise BlendError(f"mono WAVs in {d} have different sample rates {sorted(rates)}")
    return _stack(chans), fs, meta


def _stack(chans):
    n = max(c.size for c in chans)
    out = np.zeros((len(chans), n))
    for i, c in enumerate(chans):
        out[i, :c.size] = c
    return out


def describe(path):
    """Cheap summary (no audio decoding) for pickers: channels, sample rate, length."""
    import soundfile as sf
    d = Path(path)
    man = d / "manifest.json"
    if man.is_file():
        m = json.loads(man.read_text())
        return {"channels": m.get("channels"), "sample_rate": m.get("sample_rate"), "length_s": m.get("length_s")}
    files = _wavs(d) or (_wavs(next(d.glob("*_channels"))) if any(d.glob("*_channels")) else [])
    if not files:
        return None
    infos = [sf.info(str(p)) for p in files]
    best = max(infos, key=lambda i: i.channels)
    ch = best.channels if best.channels > 1 else len(infos)
    return {"channels": ch, "sample_rate": best.samplerate, "length_s": round(max(i.duration for i in infos), 3)}


def list_folders():
    """Every IR folder irgen knows about: sets, blends, standard and strange GUI renders."""
    out = []
    for kind, base in (("blend", BLENDS), ("set", RENDERS / "sets"), ("gui", RENDERS / "gui"), ("strange", RENDERS / "gui" / "strange")):
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir() or d.name.startswith(".") or (kind == "gui" and d.name in ("strange", "blends")):
                continue
            try:
                info = describe(d)
            except Exception:
                info = None
            if info:
                label = d.name
                if kind == "set":
                    try:
                        m = json.loads((d / "manifest.json").read_text())
                        label = f"{m.get('name', d.name)} · {d.name}"
                    except (OSError, json.JSONDecodeError):
                        pass
                out.append({"id": f"{kind}/{d.name}", "kind": kind, "label": label, **info,
                            "modified": d.stat().st_mtime})
    return out


# --------------------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------------------
def onset(ir, meta):
    if meta.get("direct_index") is not None:
        return int(meta["direct_index"])
    a = np.max(np.abs(ir), axis=0)
    return int(np.argmax(a >= 0.1 * a.max()))   # first sample within 20 dB of the peak


def _prepare(A, fa, ma, B, fb, mb):
    if fb != fa:
        from math import gcd
        g = gcd(int(fa), int(fb))
        B = signal.resample_poly(B, int(fa) // g, int(fb) // g, axis=-1)
        mb = {**mb, "direct_index": None if mb.get("direct_index") is None else int(round(mb["direct_index"] * fa / fb))}
    C = min(A.shape[0], B.shape[0])
    A, B = A[:C], B[:C]
    oa, ob = onset(A, ma), onset(B, mb)
    lead = max(oa, ob)
    n = max(A.shape[1] - oa, B.shape[1] - ob) + lead
    out = []
    for X, o in ((A, oa), (B, ob)):
        Y = np.zeros((C, n))
        Y[:, lead - o: lead - o + X.shape[1]] = X
        e = float(np.mean(np.sum(Y ** 2, axis=1)))
        out.append(Y / np.sqrt(e) if e > 0 else Y)
    return out[0], out[1], lead


# --------------------------------------------------------------------------------------
# blending
# --------------------------------------------------------------------------------------
def mix(A, B, amount, lead, fs):
    a = float(amount)
    t = (np.arange(A.shape[1]) - lead) / fs
    x = np.clip((t - 0.0025) / 0.0025, 0, 1)            # 2.5 ms coherent window, 2.5 ms fade
    w = 0.5 * (1 - np.cos(np.pi * x))
    wa = (1 - w) * (1 - a) + w * np.sqrt(1 - a)
    wb = (1 - w) * a + w * np.sqrt(a)
    return A * wa + B * wb


def morph(A, B, amount, lead, fs, bands_per_octave=3, hop_ms=1.0, progress=None):
    a = float(amount)
    if a <= 0:
        return A.copy()
    if a >= 1:
        return B.copy()
    # channels are independent: run groups of them in parallel (bounded memory, all cores)
    from concurrent.futures import ThreadPoolExecutor
    C = A.shape[0]
    W = max(1, min(C, os.cpu_count() or 1))
    groups = [g for g in np.array_split(np.arange(C), W) if g.size]
    done = [0]

    def run(g):
        y = _morph_group(A[g], B[g], a, lead, fs, bands_per_octave, hop_ms)
        done[0] += 1
        if progress:
            progress(done[0] / len(groups))
        return g, y

    out = np.zeros_like(A)
    with ThreadPoolExecutor(len(groups)) as ex:
        for g, y in ex.map(run, groups):
            out[g] = y
    return out


def _morph_group(A, B, a, lead, fs, bands_per_octave, hop_ms):
    from scipy import fft as sfft
    from .late import band_centers, band_masks
    C, n = A.shape
    # zero-phase band masks ring (most in the narrow low bands): pad so nothing wraps around
    nfft = sfft.next_fast_len(n + int(0.5 * fs), real=True)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    centers = band_centers(fs, bands_per_octave)
    masks = band_masks(f, centers)
    XA, XB = sfft.rfft(A, nfft, axis=1), sfft.rfft(B, nfft, axis=1)
    H = max(1, int(round(hop_ms * fs / 1000)))
    nf = int(np.ceil(n / H))
    k_lead = lead // H
    # frame-rate gains -> samples: linear between frame centres
    pos = np.clip((np.arange(n) + 0.5) / H - 0.5, 0, nf - 1)
    i0 = np.minimum(pos.astype(int), nf - 2) if nf > 1 else np.zeros(n, int)
    fr = pos - i0 if nf > 1 else np.zeros(n)

    def up(g):
        return g[:, i0] * (1 - fr) + g[:, np.minimum(i0 + 1, nf - 1)] * fr

    out = np.zeros((C, n))
    for i, fc in enumerate(centers):
        bw = fc * (2 ** (0.5 / bands_per_octave) - 2 ** (-0.5 / bands_per_octave))
        W = int(np.clip(8.0 / bw * 1000 / hop_ms, 2, 20))  # ~8 cycles of the band, 2-20 ms
        band = np.flatnonzero(masks[i] > 0)
        if band.size == 0:
            continue
        sl = slice(band[0], band[-1] + 1)                # masks are compact: only touch their bins
        Ya, Yb = np.zeros_like(XA), np.zeros_like(XB)
        Ya[:, sl], Yb[:, sl] = XA[:, sl] * masks[i, sl], XB[:, sl] * masks[i, sl]
        ab = sfft.irfft(Ya, nfft, axis=1)[:, :n]
        bb = sfft.irfft(Yb, nfft, axis=1)[:, :n]
        LA, okA, sA = _envelope_db(ab, H, nf, W, k_lead)
        LB, okB, sB = _envelope_db(bb, H, nf, W, k_lead)
        st = -np.exp((1 - a) * np.log(-sA) + a * np.log(-sB))   # target slope: T = T_A^(1-a) T_B^a
        L = (1 - a) * _warp(LA, sA, st, k_lead) + a * _warp(LB, sB, st, k_lead)   # target band level, dB
        wa, wb = np.sqrt(1 - a) * okA, np.sqrt(a) * okB  # carrier weights, equal power
        norm = np.sqrt(np.maximum(wa ** 2 + wb ** 2, 1e-12))
        wa, wb = wa / norm, wb / norm
        live = (okA + okB) > 0
        ga = np.where(live, wa * 10 ** (np.minimum(L - LA, 30.0) / 20), 0.0)   # re-shape A to the target
        gb = np.where(live, wb * 10 ** (np.minimum(L - LB, 30.0) / 20), 0.0)
        out += ab * up(ga) + bb * up(gb)
    return out


def _warp(L, s, st, k_lead):
    """Time-warp each channel's envelope from its own decay slope s to the slope st
    (dB per frame, both < 0) after the direct sound; beyond the end, continue along s."""
    C, nf = L.shape
    k = np.arange(nf, dtype=float)
    out = L.copy()
    for c in range(C):
        idx = k_lead + (k[k_lead:] - k_lead) * (st[c] / s[c])
        v = np.interp(idx, k, L[c])
        past = idx > nf - 1
        v[past] = L[c, -1] + s[c] * (idx[past] - (nf - 1))
        out[c, k_lead:] = v
    return np.maximum(out, -160.0)


def _envelope_db(x, H, nf, W, k_lead, floor_db=-160.0, reliable_db=70.0):
    """Smoothed frame power (dB) per channel, extrapolated along the fitted decay slope
    once it drops into its noise floor / ends; a 0..1 'reliable' weight; the slope
    (dB per frame, < 0)."""
    C, n = x.shape
    p = np.zeros((C, nf))
    sq = x * x
    full = (n // H) * H
    p[:, :n // H] = sq[:, :full].reshape(C, -1, H).mean(axis=2)
    if full < n:
        p[:, -1] = sq[:, full:].mean(axis=1)
    p = uniform_filter1d(uniform_filter1d(p, W, axis=1, mode="nearest"), W, axis=1, mode="nearest")
    L = 10 * np.log10(np.maximum(p, 10 ** (floor_db / 10)))
    ok = np.zeros((C, nf))
    slopes = np.full(C, -60.0 / max(nf - k_lead, 10))
    k = np.arange(nf)
    for c in range(C):
        pk = float(L[c].max())
        if pk <= floor_db + 1:
            L[c] = floor_db
            continue
        ipk = int(np.argmax(L[c]))
        live = np.flatnonzero(L[c, ipk:] >= pk - 55.0)
        cut = ipk + int(live[-1])
        # decay slope from the band's Schroeder curve (-5..-35 dB, else -5..-25 dB)
        e = np.cumsum(p[c, ::-1])[::-1]
        edc = 10 * np.log10(np.maximum(e / max(e[max(k_lead, 0)], 1e-300), 1e-30))
        slope = 0.0
        for lo in (-35.0, -25.0):
            sel = np.flatnonzero((edc <= -5.0) & (edc >= lo) & (k >= k_lead))
            if sel.size >= 8:
                slope = float(np.polyfit(sel, edc[sel], 1)[0])
                break
        if slope >= 0:
            slope = -60.0 / max(cut - k_lead, 10)
        slopes[c] = slope
        L[c, cut:] = L[c, cut] + slope * (k[cut:] - cut)
        L[c] = np.maximum(L[c], floor_db)
        ok[c] = np.clip((10 * np.log10(np.maximum(p[c], 1e-300)) - (pk - reliable_db)) / 10.0, 0, 1)
        ok[c, :max(k_lead - 2, 0)] = 0.0
    return L, ok, slopes


# --------------------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------------------
def _safe(name):
    import re
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name or "")).strip("_")


def blend_folders(name, a, b, mode="morph", amount=0.5, normalize="energy", layout="separate",
                  fmt="FLOAT", overwrite=False, outdir=None, progress=None, inspect=None):
    """Blend folders a and b (see resolve_folder) into renders/blends/<name>/.
    Returns (manifest, ir). Pass a dict as inspect to also get the aligned, normalised
    inputs (keys "a", "b")."""
    from .export import write_wav
    name = _safe(name)
    if not name:
        raise BlendError("give the blend a name")
    if mode not in MODES:
        raise BlendError(f"mode must be one of {MODES}")
    amount = float(amount)
    if not 0.0 <= amount <= 1.0:
        raise BlendError("amount must be between 0 (A) and 1 (B)")
    if normalize not in ("energy", "peak", "none"):
        raise BlendError("normalize must be energy | peak | none")
    if layout not in ("separate", "interleaved", "both"):
        raise BlendError("layout must be separate | interleaved | both")
    if fmt not in ("FLOAT", "PCM_16", "PCM_24", "PCM_32"):
        raise BlendError("format must be FLOAT | PCM_16 | PCM_24 | PCM_32")
    base = Path(outdir) if outdir else BLENDS
    dest = base / name
    da, db_ = resolve_folder(a), resolve_folder(b)
    if dest.exists() and not overwrite:
        raise BlendError(f"blend '{name}' already exists")
    if dest.resolve() in (da, db_):
        raise BlendError("the output folder can't be one of the inputs")
    t0 = time.time()
    A, fa, ma = load_folder(da)
    B, fb, mb = load_folder(db_)
    na, nb = A.shape[0], B.shape[0]
    A, B, lead = _prepare(A, fa, ma, B, fb, mb)
    fs = int(fa)
    if inspect is not None:
        inspect.update(a=A, b=B)
    y = morph(A, B, amount, lead, fs, progress=progress) if mode == "morph" else mix(A, B, amount, lead, fs)
    C, n = y.shape
    if normalize == "energy":
        g = 1.0 / np.sqrt(max(float(np.mean(np.sum(y ** 2, axis=1))), 1e-24))
    elif normalize == "peak":
        g = 10 ** (-1.0 / 20) / max(float(np.max(np.abs(y))), 1e-12)
    else:
        g = 1.0
    y *= g
    fo = min(int(0.02 * fs), n // 4)                   # 20 ms fade-out
    if fo > 1:
        y[:, -fo:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fo)))
    dt = time.time() - t0
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / f".{name}.tmp-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        files, inter = [], None
        if layout in ("separate", "both"):
            for ch in range(C):
                fn = f"ch{ch + 1:02d}.wav"
                write_wav(str(tmp / fn), y[ch], fs, fmt, True, ch)
                files.append({"channel": ch + 1, "file": fn})
        if layout in ("interleaved", "both"):
            write_wav(str(tmp / "set.wav"), y.T if C > 1 else y[0], fs, fmt, True, 0)
            inter = {"file": "set.wav", "channels": C}
        manifest = {
            "api": 1, "irgen_version": __version__, "set_id": f"blend-{name}", "stage": "full", "name": name,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "blend": {"mode": mode, "amount": amount,
                      "a": {"spec": str(a), "path": str(da), "channels": na, "sample_rate": int(fa)},
                      "b": {"spec": str(b), "path": str(db_), "channels": nb, "sample_rate": int(fb)}},
            "engine": "blend", "sample_rate": fs, "channels": C, "length_samples": int(n), "length_s": round(n / fs, 6),
            "format": fmt, "layout": layout, "normalize": normalize, "gain_db": round(20 * np.log10(g), 2),
            "direct_index": int(lead), "direct_present": True, "files": files, "interleaved": inter,
            "render_seconds": round(dt, 3),
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if dest.exists():
            old = base / f".{name}.old-{uuid.uuid4().hex[:8]}"
            os.replace(dest, old)
            os.replace(tmp, dest)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return with_paths(json.loads((dest / "manifest.json").read_text()), dest), y


def with_paths(m, d):
    for f in m.get("files", []):
        f["path"] = str(d / f["file"])
        f["url"] = f"/v1/blends/{m['name']}/{f['file']}"
    if m.get("interleaved"):
        m["interleaved"]["path"] = str(d / m["interleaved"]["file"])
        m["interleaved"]["url"] = f"/v1/blends/{m['name']}/{m['interleaved']['file']}"
    return m


def find_blend(name):
    d = BLENDS / _safe(name)
    m = d / "manifest.json"
    return with_paths(json.loads(m.read_text()), d) if m.is_file() else None
