"""ISO 3382-style analysis of an impulse response + diagnostic plots."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.special import erfc

from .acoustics import OCTAVE_BANDS


def _onset(ir):
    a = np.abs(ir)
    thr = a.max() * 10 ** (-20 / 20)
    return int(np.argmax(a >= thr))


def _edc_db(x):
    e = np.cumsum(x[::-1] ** 2)[::-1]
    e = e / max(e[0], 1e-300)
    return 10 * np.log10(np.maximum(e, 1e-30))


def _fit_decay(edc, fs, hi, lo):
    idx = np.flatnonzero((edc <= hi) & (edc >= lo))
    if idx.size < 10 or edc.min() > lo:
        return float("nan")
    t = idx / fs
    slope, _ = np.polyfit(t, edc[idx], 1)
    return float(-60.0 / slope) if slope < 0 else float("nan")


def _params(x, fs, onset):
    y = x[onset:]
    e = y ** 2
    tot = e.sum()
    n50, n80 = int(0.05 * fs), int(0.08 * fs)
    edc = _edc_db(y)
    t = np.arange(y.size) / fs
    return {
        "EDT": _fit_decay(edc, fs, 0.0, -10.0),
        "T20": _fit_decay(edc, fs, -5.0, -25.0),
        "T30": _fit_decay(edc, fs, -5.0, -35.0),
        "C50": float(10 * np.log10(e[:n50].sum() / max(e[n50:].sum(), 1e-300))),
        "C80": float(10 * np.log10(e[:n80].sum() / max(e[n80:].sum(), 1e-300))),
        "D50": float(e[:n50].sum() / max(tot, 1e-300)),
        "Ts_ms": float(1000 * np.sum(t * e) / max(tot, 1e-300)),
    }, edc


def echo_density(ir, fs, win_ms=20.0, hop_ms=1.0):
    """Normalised echo density profile (Abel & Huang 2006)."""
    L = int(win_ms * fs / 1000)
    hop = int(hop_ms * fs / 1000)
    w = np.hanning(L)
    w /= w.sum()
    times, ned = [], []
    for s in range(0, ir.size - L, hop):
        seg = ir[s:s + L]
        sd = np.sqrt(np.sum(w * seg ** 2))
        ned.append(np.sum(w * (np.abs(seg) > sd)) / erfc(1 / np.sqrt(2)))
        times.append((s + L / 2) / fs)
    return np.array(times), np.array(ned)


def analyze(ir, fs, bands=None):
    ir = np.asarray(ir, float)
    bands = [b for b in (bands or OCTAVE_BANDS) if b * np.sqrt(2) < fs / 2 * 0.95]
    onset = _onset(ir)
    res = {"onset_s": onset / fs, "bands": {}, "edc": {}}
    for fc in bands:
        sos = signal.butter(3, [fc / np.sqrt(2), fc * np.sqrt(2)], "band", fs=fs, output="sos")
        y = signal.sosfilt(sos, ir)
        p, edc = _params(y, fs, onset)
        res["bands"][float(fc)] = p
        res["edc"][float(fc)] = edc
    p, edc = _params(ir, fs, onset)
    res["broadband"] = p
    res["edc"]["broadband"] = edc
    # direct-to-reverberant ratio (direct = +-2.5 ms around the peak)
    pk = int(np.argmax(np.abs(ir)))
    d = int(0.0025 * fs)
    e = ir ** 2
    ed = e[max(pk - d, 0):pk + d].sum()
    er = e.sum() - ed
    res["DRR_dB"] = float(10 * np.log10(ed / max(er, 1e-300)))
    t, ned = echo_density(ir[onset:], fs)
    res["echo_density"] = (t, ned)
    hit = np.flatnonzero(ned >= 0.95)
    res["ned_mixing_ms"] = float(t[hit[0]] * 1000) if hit.size else float("nan")
    # bass ratio / brilliance on T30 (fallback T20)
    def T(fc):
        b = res["bands"].get(float(fc))
        if not b:
            return np.nan
        return b["T30"] if np.isfinite(b["T30"]) else b["T20"]
    res["BR"] = float((T(125) + T(250)) / (T(500) + T(1000)))
    res["Br"] = float((T(2000) + T(4000)) / (T(500) + T(1000)))
    return res


def summary_table(res):
    lines = []
    hdr = f"{'band':>9} {'EDT':>6} {'T20':>6} {'T30':>6} {'C50':>6} {'C80':>6} {'D50':>5} {'Ts':>6}"
    lines.append(hdr)
    rows = list(res["bands"].items()) + [("broad", res["broadband"])]
    for fc, p in rows:
        name = f"{fc:>7.0f}Hz" if isinstance(fc, float) else f"{fc:>9}"
        lines.append(f"{name} {p['EDT']:6.2f} {p['T20']:6.2f} {p['T30']:6.2f} {p['C50']:6.1f} {p['C80']:6.1f} {p['D50']:5.2f} {p['Ts_ms']:6.0f}")
    lines.append(f"DRR {res['DRR_dB']:.1f} dB | BR {res['BR']:.2f} | Br {res['Br']:.2f} | echo-density mixing {res['ned_mixing_ms']:.0f} ms")
    return "\n".join(lines)


def to_json(res):
    return {
        "onset_s": res["onset_s"],
        "bands": {str(int(k)): v for k, v in res["bands"].items()},
        "broadband": res["broadband"],
        "DRR_dB": res["DRR_dB"], "BR": res["BR"], "Br": res["Br"],
        "ned_mixing_ms": res["ned_mixing_ms"],
    }


def plot_report(ir, fs, res, path, title="", target_t60=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ir = np.asarray(ir, float)
    t = np.arange(ir.size) / fs
    on = res["onset_s"]
    fig, ax = plt.subplots(4, 2, figsize=(14, 15))
    fig.suptitle(title, fontsize=14)

    # 1 early part
    a = ax[0, 0]
    m = (t >= on - 0.002) & (t <= on + 0.15)
    a.plot((t[m] - on) * 1000, ir[m], lw=0.6)
    a.set(title="Early part (0-150 ms after direct)", xlabel="ms", ylabel="amplitude")
    a.grid(alpha=0.3)

    # 2 envelope in dB
    a = ax[0, 1]
    db = 20 * np.log10(np.abs(ir) + 1e-12)
    a.plot(t, db, lw=0.3, color="0.55")
    sm = int(0.005 * fs)
    env = 10 * np.log10(np.convolve(ir ** 2, np.ones(sm) / sm, "same") + 1e-24)
    a.plot(t, env, lw=1.0, color="C0", label="5 ms RMS")
    a.set(title="Envelope", xlabel="s", ylabel="dBFS", ylim=(max(db.max() - 120, -200), db.max() + 3))
    a.legend(); a.grid(alpha=0.3)

    # 3 spectrogram
    a = ax[1, 0]
    nper = int(2 ** np.round(np.log2(0.02 * fs)))
    f, tt, S = signal.spectrogram(ir, fs, nperseg=nper, noverlap=nper * 3 // 4, window="hann")
    SdB = 10 * np.log10(S + 1e-30)
    vmax = SdB.max()
    a.pcolormesh(tt, f, SdB, shading="auto", vmin=vmax - 100, vmax=vmax, cmap="magma")
    a.set_yscale("log"); a.set_ylim(30, fs / 2)
    a.set(title="Spectrogram (dB)", xlabel="s", ylabel="Hz")

    # 4 EDCs
    a = ax[1, 1]
    for i, (fc, edc) in enumerate(res["edc"].items()):
        tt2 = np.arange(edc.size) / fs
        lab = f"{fc:.0f} Hz" if isinstance(fc, float) else "broadband"
        a.plot(tt2, edc, lw=1.0 if lab != "broadband" else 2.0, color=None if lab != "broadband" else "k", label=lab)
    a.set(title="Energy decay curves (Schroeder)", xlabel="s", ylabel="dB", ylim=(-80, 2))
    a.legend(fontsize=7, ncol=2); a.grid(alpha=0.3)

    # 5 reverberation times
    a = ax[2, 0]
    fcs = np.array(list(res["bands"].keys()))
    for key, mk in (("EDT", "o-"), ("T20", "s-"), ("T30", "^-")):
        a.plot(fcs, [res["bands"][f][key] for f in fcs], mk, label=key)
    if target_t60 is not None:
        tf, tv = target_t60
        a.plot(tf, tv, "k--", lw=2, label="target T60")
    a.set_xscale("log"); a.set_xticks(fcs); a.set_xticklabels([f"{f:g}" for f in fcs])
    a.set(title="Reverberation time per octave", xlabel="Hz", ylabel="s")
    a.legend(); a.grid(alpha=0.3, which="both")

    # 6 magnitude response
    a = ax[2, 1]
    nfft = int(2 ** np.ceil(np.log2(ir.size)))
    H = np.abs(np.fft.rfft(ir, nfft))
    fr = np.fft.rfftfreq(nfft, 1 / fs)
    # 1/6-octave smoothing in power
    P = H ** 2
    cs = np.concatenate([[0], np.cumsum(P)])
    lo = np.searchsorted(fr, fr * 2 ** (-1 / 12))
    hi = np.maximum(np.searchsorted(fr, fr * 2 ** (1 / 12)), lo + 1)
    Ps = (cs[hi] - cs[lo]) / (hi - lo)
    a.semilogx(fr[1:], 10 * np.log10(Ps[1:] + 1e-30), lw=1.2)
    a.set_xlim(20, fs / 2)
    a.set(title="Magnitude response (1/6-oct smoothed)", xlabel="Hz", ylabel="dB")
    a.grid(alpha=0.3, which="both")

    # 7 echo density
    a = ax[3, 0]
    te, ned = res["echo_density"]
    a.plot(te * 1000, ned)
    a.axhline(1.0, color="k", ls="--", lw=0.8)
    a.set(title="Normalised echo density (Abel & Huang)", xlabel="ms after direct", ylabel="NED",
          xlim=(0, min(500, te.max() * 1000 if te.size else 500)))
    a.grid(alpha=0.3)

    # 8 table
    a = ax[3, 1]
    a.axis("off")
    a.text(0.0, 1.0, summary_table(res), family="monospace", fontsize=8.5, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    plt.close(fig)
