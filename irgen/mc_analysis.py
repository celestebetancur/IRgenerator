"""Evaluation of a multichannel IR set: decorrelation (vs target coherence) and
perceptual similarity (per-channel ISO 3382 parameters vs just-noticeable differences)."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .acoustics import OCTAVE_BANDS
from .analysis import _onset, _params

# Just-noticeable differences (ISO 3382-1 Annex A / common literature)
JND = {"T30_rel": 0.05, "EDT_rel": 0.05, "C80_db": 1.0, "level_db": 1.0, "spectrum_db": 1.0}


def _band(x, fs, fc):
    sos = signal.butter(3, [fc / np.sqrt(2), min(fc * np.sqrt(2), fs / 2 * 0.99)], "band", fs=fs, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


def _corr_matrix(X):
    X = X - X.mean(axis=1, keepdims=True)
    nrm = np.sqrt(np.sum(X ** 2, axis=1))
    return (X @ X.T) / np.maximum(np.outer(nrm, nrm), 1e-300)


def max_lag_xcorr(X, fs, max_lag_ms=1.0):
    """Max |normalised cross-correlation| over +-max_lag for every pair (FFT based)."""
    N, n = X.shape
    L = int(max_lag_ms * fs / 1000)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(X, nfft, axis=1)
    nrm = np.sqrt(np.sum(X ** 2, axis=1))
    R = np.zeros((N, N))
    for i in range(N):
        cc = np.fft.irfft(F[i][None, :] * np.conj(F[i:]), nfft, axis=1)
        cc = np.concatenate([cc[:, -L:], cc[:, :L + 1]], axis=1)
        v = np.max(np.abs(cc), axis=1) / np.maximum(nrm[i] * nrm[i:], 1e-300)
        R[i, i:] = v
        R[i:, i] = v
    return R


def analyze_multichannel(irs, fs, late_start_s, target_fn=None, bands=None):
    irs = np.asarray(irs, float)
    N, n = irs.shape
    bands = [b for b in (bands or OCTAVE_BANDS) if b * np.sqrt(2) < fs / 2 * 0.95]
    onset = _onset(irs[0])
    ls = int(np.clip(late_start_s * fs, 0, n - fs // 10))
    late = irs[:, ls:]
    out = {"channels": N, "late_start_ms": 1000 * ls / fs, "bands": [float(b) for b in bands]}

    # ---- decorrelation ----
    R_full = max_lag_xcorr(irs, fs)
    R_late = _corr_matrix(late)
    iu = np.triu_indices(N, 1)
    coh_meas, coh_tgt = [], []
    for fc in bands:
        C = _corr_matrix(_band(late, fs, fc))
        coh_meas.append(C[iu])
        if target_fn is not None:
            coh_tgt.append(target_fn(fc)[iu])
    out["xcorr_full_maxlag"] = R_full
    out["corr_late"] = R_late
    out["band_coherence_measured"] = np.array(coh_meas)          # (bands, pairs)
    out["band_coherence_target"] = np.array(coh_tgt) if coh_tgt else None
    if N > 1:
        out["summary_decorrelation"] = {
            "full_ir_max_abs_xcorr_mean": float(np.mean(R_full[iu])),
            "full_ir_max_abs_xcorr_max": float(np.max(R_full[iu])),
            "late_abs_corr_mean": float(np.mean(np.abs(R_late[iu]))),
            "late_abs_corr_max": float(np.max(np.abs(R_late[iu]))),
        }
        if coh_tgt:
            err = np.abs(np.array(coh_meas) - np.array(coh_tgt))
            out["summary_decorrelation"]["coherence_abs_error_mean"] = float(err.mean())
            out["summary_decorrelation"]["coherence_abs_error_p95"] = float(np.percentile(err, 95))
            # statistical floor: the estimator has std ~ 1/sqrt(BT) for the band/late window
            out["summary_decorrelation"]["estimator_noise_1k"] = float(1 / np.sqrt(max(1000 * 0.7 * (n - ls) / fs, 1)))

    # ---- similarity ----
    per = {k: np.full((N, len(bands)), np.nan) for k in ("T30", "EDT", "C80")}
    for bi, fc in enumerate(bands):
        Y = _band(irs, fs, fc)
        for ch in range(N):
            p, _ = _params(Y[ch], fs, onset)
            per["T30"][ch, bi] = p["T30"] if np.isfinite(p["T30"]) else p["T20"]
            per["EDT"][ch, bi] = p["EDT"]
            per["C80"][ch, bi] = p["C80"]
    level = 10 * np.log10(np.sum(irs ** 2, axis=1))
    # 1/3-octave spectra of the late part (dB)
    nfft = int(2 ** np.ceil(np.log2(late.shape[1])))
    P = np.abs(np.fft.rfft(late, nfft, axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    third = 1000 * 2 ** (np.arange(-15, 14) / 3)
    third = third[(third <= 16000) & (third * 2 ** (1 / 6) < fs / 2 * 0.95) & (third >= 40)]
    spec = np.stack([10 * np.log10(np.mean(P[:, (f >= fc * 2 ** (-1 / 6)) & (f < fc * 2 ** (1 / 6))], axis=1) + 1e-300)
                     for fc in third], axis=1)
    keep = spec.mean(axis=0) > spec.mean(axis=0).max() - 50  # ignore bands with (almost) no energy
    third, spec = third[keep], spec[:, keep]
    spec_dev = spec - spec.mean(axis=0, keepdims=True)
    out.update({"per_channel": per, "level_db": level, "third_oct": third, "spectrum_dev_db": spec_dev})
    medT = np.nanmedian(per["T30"], axis=0)
    medE = np.nanmedian(per["EDT"], axis=0)
    medC = np.nanmedian(per["C80"], axis=0)
    # ISO 3382 single-number values: mean of the 500 Hz and 1 kHz octaves
    mid = [i for i, b in enumerate(bands) if b in (500.0, 1000.0)]
    Tm, Em, Cm = (np.nanmean(per[k][:, mid], axis=1) for k in ("T30", "EDT", "C80"))
    sim = {
        "T30_mid_max_dev_pct": float(np.max(np.abs(Tm / np.median(Tm) - 1)) * 100),
        "EDT_mid_max_dev_pct": float(np.max(np.abs(Em / np.median(Em) - 1)) * 100),
        "C80_mid_max_dev_db": float(np.max(np.abs(Cm - np.median(Cm)))),
        "level_spread_db": float(level.max() - level.min()),
        "spectrum_rms_dev_db": float(np.sqrt(np.mean(spec_dev ** 2))),
        "worst_band_T30_dev_pct": float(np.nanmax(np.abs(per["T30"] / medT - 1)) * 100),
        "worst_band_EDT_dev_pct": float(np.nanmax(np.abs(per["EDT"] / medE - 1)) * 100),
        "worst_band_C80_dev_db": float(np.nanmax(np.abs(per["C80"] - medC))),
        "spectrum_max_dev_db": float(np.max(np.abs(spec_dev))),
    }
    sim["within_jnd"] = {
        "T30": sim["T30_mid_max_dev_pct"] <= JND["T30_rel"] * 100,
        "EDT": sim["EDT_mid_max_dev_pct"] <= JND["EDT_rel"] * 100,
        "C80": sim["C80_mid_max_dev_db"] <= JND["C80_db"],
        "level": sim["level_spread_db"] <= JND["level_db"],
        "spectrum": sim["spectrum_rms_dev_db"] <= JND["spectrum_db"],
    }
    out["summary_similarity"] = sim
    return out


def mc_to_json(res):
    return {
        "channels": res["channels"],
        "late_start_ms": res["late_start_ms"],
        "decorrelation": res.get("summary_decorrelation", {}),
        "similarity": res["summary_similarity"],
        "band_coherence_mean_measured": dict(zip([str(int(b)) for b in res["bands"]],
                                                 np.mean(res["band_coherence_measured"], axis=1).round(4).tolist()))
        if res["channels"] > 1 else {},
        "band_coherence_mean_target": dict(zip([str(int(b)) for b in res["bands"]],
                                               np.mean(res["band_coherence_target"], axis=1).round(4).tolist()))
        if res.get("band_coherence_target") is not None and res["channels"] > 1 else {},
        "level_db_per_channel": np.round(res["level_db"], 2).tolist(),
        "T30_per_channel": {str(int(b)): np.round(res["per_channel"]["T30"][:, i], 3).tolist() for i, b in enumerate(res["bands"])},
    }


def plot_mc_report(irs, fs, res, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = res["channels"]
    bands = np.array(res["bands"])
    fig, ax = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle(title, fontsize=14)

    a = ax[0, 0]
    im = a.imshow(np.abs(res["corr_late"]), vmin=0, vmax=1, cmap="viridis")
    step = max(1, int(np.ceil(N / 16)))
    tk = np.arange(0, N, step)
    a.set_xticks(tk); a.set_xticklabels(tk + 1); a.set_yticks(tk); a.set_yticklabels(tk + 1)
    a.set(title="|correlation| of late part (pairwise)", xlabel="channel", ylabel="channel")
    fig.colorbar(im, ax=a, fraction=0.046)

    a = ax[0, 1]
    cm = res["band_coherence_measured"]
    if N > 1:
        a.fill_between(bands, cm.min(axis=1), cm.max(axis=1), alpha=0.2, label="measured range (all pairs)")
        a.plot(bands, cm.mean(axis=1), "o-", label="measured mean")
        ct = res.get("band_coherence_target")
        if ct is not None:
            a.plot(bands, ct.mean(axis=1), "k--", lw=2, label="target mean")
            if ct.shape[1] > 1:
                a.fill_between(bands, ct.min(axis=1), ct.max(axis=1), color="k", alpha=0.07, label="target range")
    a.set_xscale("log"); a.set_xticks(bands); a.set_xticklabels([f"{b:g}" for b in bands])
    a.set(title="Band coherence of late part", xlabel="Hz", ylabel="correlation coefficient", ylim=(-0.3, 1.05))
    a.axhline(0, color="0.5", lw=0.8); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 0]
    T = res["per_channel"]["T30"]
    for ch in range(N):
        a.plot(bands, T[ch], color="C0", alpha=max(0.08, 1.0 / N ** 0.5), lw=1)
    med = np.nanmedian(T, axis=0)
    a.plot(bands, med, "k", lw=2, label="median")
    a.fill_between(bands, med * 0.95, med * 1.05, color="g", alpha=0.15, label="±5 % (JND)")
    a.set_xscale("log"); a.set_xticks(bands); a.set_xticklabels([f"{b:g}" for b in bands])
    a.set(title="T30 per channel", xlabel="Hz", ylabel="s"); a.legend(); a.grid(alpha=0.3)

    a = ax[1, 1]
    th = res["third_oct"]
    for ch in range(N):
        a.semilogx(th, res["spectrum_dev_db"][ch], color="C1", alpha=max(0.08, 1.0 / N ** 0.5), lw=1)
    a.axhspan(-1, 1, color="g", alpha=0.15, label="±1 dB")
    a.set(title="Late-part 1/3-octave spectrum, deviation from channel mean", xlabel="Hz", ylabel="dB", ylim=(-6, 6))
    a.legend(); a.grid(alpha=0.3, which="both")

    a = ax[2, 0]
    k = min(N, 6)
    t = np.arange(irs.shape[1]) / fs
    m = t < res["late_start_ms"] / 1000 + 0.05
    sc = np.max(np.abs(irs[:k, m])) * 1.2
    for ch in range(k):
        a.plot(t[m] * 1000, irs[ch, m] / sc - ch, lw=0.5)
    a.set(title=f"Early part, first {k} channels (offset)", xlabel="ms", yticks=[])
    a.axvline(res["late_start_ms"], color="k", ls="--", lw=0.8)

    a = ax[2, 1]
    a.axis("off")
    lines = [f"channels: {N}", ""]
    for k2, v in res.get("summary_decorrelation", {}).items():
        lines.append(f"{k2:34s} {v:7.3f}")
    lines.append("")
    s = res["summary_similarity"]
    for k2, v in s.items():
        if k2 != "within_jnd":
            lines.append(f"{k2:34s} {v:7.2f}")
    lines.append("")
    lines.append("within JND: " + ", ".join(f"{k2} {'yes' if v else 'NO'}" for k2, v in s["within_jnd"].items()))
    a.text(0, 1, "\n".join(lines), family="monospace", fontsize=9, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=100)
    plt.close(fig)
