"""IR synthesis engine: direct sound + early reflections + late tail (+ low-frequency modes)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import acoustics as ac
from .config import load_preset
from .eq import apply_eq, dc_block, lr4
from .late import band_centers, gaussian, modal_ir, shape_noise, shape_noise_multi, velvet_growing
from .multichannel import array_offsets, coherence_matrix, mixing_matrix
from .reflections import ReflectionRenderer, list_early, parametric_early, shoebox_images
from scipy import signal


def _auto(v):
    return v is None or v == "auto"


def _surface_matrix(spec, name, allow_material):
    """Resolve {default, west, east, ...} | scalar | list into a (6, 9) array."""
    if isinstance(spec, dict) and not ({"freqs", "material", "mid"} & set(spec)):
        default = spec.get("default", "plaster" if allow_material else 0.1)
        rows = [ac.resolve_band_values(spec.get(s, default), f"{name}.{s}", allow_material) for s in ac.SURFACES]
    else:
        row = ac.resolve_band_values(spec, name, allow_material)
        rows = [row] * 6
    return np.array(rows, dtype=float)


class Room:
    def __init__(self, rc, rcv=None, rcv_orient=None):
        self.dims = np.asarray(rc["dimensions"], float)
        Lx, Ly, Lz = self.dims
        self.src = np.asarray(rc["source"]["position"], float)
        self.rcv = np.asarray(rc["receiver"]["position"] if rcv is None else rcv, float)
        for nm, p in (("source", self.src), ("receiver", self.rcv)):
            if np.any(p <= 0) or np.any(p >= self.dims):
                raise ValueError(f"room.{nm}.position {p.tolist()} lies outside the room {self.dims.tolist()}")
        self.air = rc.get("air", {})
        self.c = ac.speed_of_sound(self.air.get("temperature_c", 20.0))
        self.areas = np.array([Ly * Lz, Ly * Lz, Lx * Lz, Lx * Lz, Lx * Ly, Lx * Ly])
        self.V = Lx * Ly * Lz
        self.S = self.areas.sum()
        self.alpha = np.clip(_surface_matrix(rc["materials"], "room.materials", True), 1e-4, 0.999)
        self.scat = np.clip(_surface_matrix(rc.get("scattering", 0.0), "room.scattering", False), 0.0, 0.999)
        self.alpha_mean = self.areas @ self.alpha / self.S
        self.scat_mean = self.areas @ self.scat / self.S
        air_db = ac.air_absorption_db_per_m(ac.OCTAVE_BANDS, self.air.get("temperature_c", 20.0),
                                            self.air.get("humidity_pct", 50.0), self.air.get("pressure_kpa", 101.325))
        self.m_air = air_db / (10 * np.log10(np.e))
        self.T = ac.reverberation_time(self.V, self.S, self.alpha_mean, self.m_air, rc.get("reverb_formula", "eyring"), self.c)
        self.t_mix = ac.mixing_time_s(self.V, self.S)
        self.r = float(np.linalg.norm(self.rcv - self.src))
        self.t_direct = self.r / self.c
        sd, rd = rc["source"], rc["receiver"]
        self.src_dir = (ac.pattern_coeff(sd.get("directivity", "omni")), ac.orientation_vector(sd.get("orientation", [0, 0])))
        axis = ac.orientation_vector(rd.get("orientation", [0, 0])) if rcv_orient is None else np.asarray(rcv_orient, float)
        self.rcv_dir = (ac.pattern_coeff(rd.get("directivity", "omni")), axis)
        u = (self.src - self.rcv) / self.r  # direction the direct sound arrives from
        self.a_direct = (1.0 / self.r) * ac.directivity_gain(self.rcv_dir[0], self.rcv_dir[1], u[None, :])[0] \
            * ac.directivity_gain(self.src_dir[0], self.src_dir[1], -u[None, :])[0]
        self.diffuse_gain = ac.diffuse_power(self.src_dir[0]) * ac.diffuse_power(self.rcv_dir[0])
        t_mid = float(np.interp(np.log2(1000), np.log2(ac.OCTAVE_BANDS), self.T))
        self.schroeder = 2000.0 * np.sqrt(t_mid / self.V)
        # first-order reflection time
        firsts = []
        for ax in range(3):
            for wall in (0.0, self.dims[ax]):
                img = self.src.copy()
                img[ax] = 2 * wall - img[ax]
                firsts.append(np.linalg.norm(img - self.rcv))
        self.t_first = min(firsts) / self.c


# T30 calibration factors, cached per exact calibration setup (renders/cache/calibration).
# Calibration is deterministic, so a hit reproduces the very same IR without the extra
# render + analysis passes. IRGEN_CALIBRATION_CACHE=0 turns the cache off.
CAL_CACHE = Path(__file__).resolve().parent.parent / "renders" / "cache" / "calibration"
_cal_mem = {}


def _cal_key(cfg, iters):
    """Everything the mono calibration render depends on (not naming, output format or
    normalisation: T30 is level-independent; not multichannel: calibration is mono)."""
    from . import __version__
    c = copy.deepcopy(cfg)
    c.pop("name", None)
    c.pop("multichannel", None)
    for k in ("format", "dither", "write_report", "normalize", "peak_dbfs"):
        c["output"].pop(k, None)
    c["late"]["calibrate"] = iters
    blob = json.dumps({"irgen": __version__, "cfg": c}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _cal_get(key):
    if os.environ.get("IRGEN_CALIBRATION_CACHE", "1") == "0":
        return None
    if key in _cal_mem:
        return _cal_mem[key]
    p = CAL_CACHE / f"{key}.json"
    try:
        hit = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    _cal_mem[key] = hit
    return hit


def _cal_put(key, adj, hist):
    if os.environ.get("IRGEN_CALIBRATION_CACHE", "1") == "0":
        return
    hit = {"adj": [float(a) for a in adj], "hist": [float(h) for h in hist]}
    _cal_mem[key] = hit
    try:
        CAL_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CAL_CACHE / f".{key}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(hit))
        os.replace(tmp, CAL_CACHE / f"{key}.json")
    except OSError:
        pass   # read-only location: the in-memory cache still works


def generate(preset, verbose=False):
    """Render an IR from a preset (path or dict). Returns (ir, fs, info, resolved_cfg).

    If late.calibrate > 0, the internal decay curve is iteratively corrected so that the
    octave-band T30 measured on the final IR matches the requested / theoretical T60."""
    from .analysis import analyze
    cfg = load_preset(preset)
    iters = int(cfg["late"].get("calibrate", 0) or 0)
    if cfg["late"].get("double_slope", {}).get("enabled") or cfg["late"]["source"] == "none":
        iters = 0
    adj = np.ones(len(ac.OCTAVE_BANDS))
    multi = int((cfg.get("multichannel") or {}).get("channels", 1)) > 1
    # calibration runs on a mono render (fast); the multichannel set is rendered once at the end
    cal_cfg = cfg
    if multi and iters:
        cal_cfg = copy.deepcopy(cfg)
        cal_cfg["multichannel"]["channels"] = 1
    key = _cal_key(cfg, iters) if iters else None
    hit = _cal_get(key) if key else None
    if hit is not None:   # same calibration setup as before: reuse its factors, render once
        adj, hist = np.array(hit["adj"]), list(hit["hist"])
        ir, fs, info = _render(cfg, adj)
        info["t60_calibration_factors"] = dict(zip(ac.OCTAVE_BANDS.tolist(), adj.round(3).tolist()))
        info["t60_calibration_max_error_pct"] = [round((2 ** h - 1) * 100, 1) for h in hist]
        info["t60_calibration_cached"] = True
        return ir, fs, info, cfg
    ir, fs, info = _render(cal_cfg, adj)
    target = np.array(list(info["t60_bands"].values()))
    hist = []
    for it in range(iters):
        res = analyze(ir if ir.ndim == 1 else ir[0], fs)
        meas = np.full(adj.size, np.nan)
        for i, fc in enumerate(ac.OCTAVE_BANDS):
            b = res["bands"].get(float(fc))
            if b:
                meas[i] = b["T30"] if np.isfinite(b["T30"]) else b["T20"]
        err = target / meas
        ok = np.isfinite(err)
        hist.append(float(np.nanmax(np.abs(np.log2(err[ok])))) if ok.any() else 0.0)
        if not ok.any() or hist[-1] < np.log2(1.02):
            break
        adj[ok] *= np.clip(err[ok], 0.5, 2.0)
        ir, fs, info = _render(cal_cfg, adj)
    if cal_cfg is not cfg:   # calibrated on mono: render the multichannel set once, with the factors
        ir, fs, info = _render(cfg, adj)
    if iters:
        _cal_put(key, adj, hist)
        info["t60_calibration_factors"] = dict(zip(ac.OCTAVE_BANDS.tolist(), adj.round(3).tolist()))
        info["t60_calibration_max_error_pct"] = [round((2 ** h - 1) * 100, 1) for h in hist]
    return ir, fs, info, cfg


def _render(cfg, t_adjust):
    oc, ec, lc, dc = cfg["output"], cfg["early"], cfg["late"], cfg["direct"]
    mc = cfg.get("multichannel", {}) or {}
    fs = int(oc["sample_rate"])
    N = int(np.clip(int(mc.get("channels", 1)), 1, 64))
    seed = int(cfg.get("seed", 1))
    seeds = np.random.SeedSequence(seed).spawn(4)
    rng_img, rng_er, rng_late, rng_mod = [np.random.default_rng(s) for s in seeds]
    ch_seeds = np.random.SeedSequence([seed, 7919]).spawn(3 * 64)  # per-channel streams (ch >= 1)

    def ch_rng(kind, ch):
        base = {"img": rng_img, "er": rng_er, "late": rng_late}[kind]
        if ch == 0:
            return base
        return np.random.default_rng(ch_seeds[{"img": 0, "er": 1, "late": 2}[kind] * 64 + ch])

    info = {"name": cfg.get("name"), "sample_rate": fs, "channels": N}

    uses_room = ec["source"] == "room" or lc["source"] == "room"
    room = Room(cfg["room"]) if uses_room else None
    air_cfg = cfg["room"].get("air", {})
    c = room.c if room else ac.speed_of_sound(air_cfg.get("temperature_c", 20.0))

    # ---------------- channel layout ----------------
    acfg = mc.get("array", {}) or {}
    offsets, orients = array_offsets(acfg, N)
    has_array = N > 1 and acfg.get("type", "none") != "none"
    er_mode = mc.get("early", "jitter") if N > 1 else "same"
    dir_mode = mc.get("direct", "auto") if N > 1 else "same"
    if dir_mode == "auto":
        dir_mode = "per_channel" if has_array else "same"
    pos_offsets = offsets.copy()
    if N > 1 and not has_array and er_mode in ("jitter", "independent"):
        rj = float(mc.get("receiver_jitter_m", 0.15))
        jr = np.random.default_rng([seed, 31337])
        pos_offsets = jr.normal(0.0, rj, (N, 3))
        pos_offsets[0] = 0.0
    ccfg = mc.get("coherence", {"model": "independent"}) or {}
    if N > 1 and ccfg.get("model", "independent") in ("diffuse", "diffuse_3d", "diffuse_2d") and not has_array:
        raise ValueError("coherence model 'diffuse' needs multichannel.array (mic spacing defines coherence)")
    rooms = [room] * N
    if room is not None and N > 1 and (er_mode != "same" or dir_mode == "per_channel"):
        rooms = []
        for ch in range(N):
            p = np.clip(room.rcv + pos_offsets[ch], 0.02, room.dims - 0.02)
            rooms.append(Room(cfg["room"], rcv=p, rcv_orient=None if orients is None else orients[ch]))

    # ---------------- reverberation time curve ----------------
    if lc.get("t60") is not None:
        T_bands = ac.curve_from_spec(lc["t60"], ac.OCTAVE_BANDS)
    elif room is not None:
        T_bands = room.T.copy()
    else:
        T_bands = ac.curve_from_spec({"mid": 1.8, "low_ratio": 1.25, "high_ratio": 0.6}, ac.OCTAVE_BANDS)
    T_bands = np.maximum(T_bands * float(lc.get("t60_scale", 1.0)), 0.02)
    T_target = T_bands.copy()
    T_bands = T_bands * t_adjust
    logT = np.log(T_bands)
    lgb = np.log2(ac.OCTAVE_BANDS)

    def t60_fn(f):
        return np.exp(np.interp(np.log2(np.maximum(f, 1e-3)), lgb, logT))

    ds = lc.get("double_slope", {}) or {}
    if ds.get("enabled"):
        if ds.get("t60") is not None:
            T2_bands = ac.curve_from_spec(ds["t60"], ac.OCTAVE_BANDS)
        else:
            T2_bands = T_bands * float(ds.get("t60_ratio", 2.5))
        logT2 = np.log(np.maximum(T2_bands, 0.02))

        def t60b_fn(f):
            return np.exp(np.interp(np.log2(np.maximum(f, 1e-3)), lgb, logT2))
    else:
        T2_bands = None

    # ---------------- timeline ----------------
    t_direct = room.t_direct if room else 0.0
    lead = int(round(oc["lead_ms"] * fs / 1000.0))
    if room and N > 1 and dir_mode == "per_channel":
        # keep every channel's direct sound after the lead-in
        lead += int(np.ceil(max(0.0, t_direct - min(r.t_direct for r in rooms)) * fs))
    shift = (t_direct if (room and oc.get("align_to_direct", True)) else 0.0) - lead / fs  # t_emit = shift + n/fs

    def to_samples(t_emit):
        return (np.asarray(t_emit) - shift) * fs

    trim = float(oc.get("trim_db", 90.0))
    audible = ac.OCTAVE_BANDS <= min(16000, fs / 2)
    t_end = t_direct + float(np.max(np.maximum(T_bands, T_target)[audible])) * trim / 60.0
    if T2_bands is not None:
        t_end = max(t_end, t_direct + float(np.max(T2_bands[audible])) * (trim + ds.get("level_db", -18.0)) / 60.0)
    if _auto(oc.get("length_s")):
        length = min(t_end - shift + 0.05, float(oc.get("max_length_s", 30.0)))
    else:
        length = float(oc["length_s"])
    n = int(np.ceil(length * fs))
    t_emit = shift + np.arange(n) / fs
    info.update({"length_s": n / fs, "t60_bands": dict(zip(ac.OCTAVE_BANDS.tolist(), T_target.round(3).tolist()))})

    renderer = ReflectionRenderer(fs, air=air_cfg)
    a_d = room.a_direct if room else 1.0

    def tone_grid(spec):
        return None if spec is None else ac.curve_from_spec(spec, renderer.f)

    # ---------------- direct sound ----------------
    ir_direct = np.zeros((N, n))
    if dc.get("enabled", True):
        dgain = 10 ** (dc.get("gain_db", 0.0) / 20.0)
        dtone = tone_grid(dc.get("tone"))
        for ch in range(N):
            if ch > 0 and dir_mode == "same":
                ir_direct[ch] = ir_direct[0]
                continue
            r_ = rooms[ch]
            renderer.render(ir_direct[ch], [to_samples(r_.t_direct if r_ else t_direct)],
                            [(r_.a_direct if r_ else 1.0) * dgain], air_dist=[r_.r] if r_ else None, tone_db=dtone)

    # ---------------- early reflections ----------------
    ir_early = np.zeros((N, n))
    eg = 10 ** (ec.get("gain_db", 0.0) / 20.0)
    etone = tone_grid(ec.get("tone"))
    xover = None
    if room is not None:
        t_x = (room.t_mix if _auto(ec.get("crossover_ms")) else ec["crossover_ms"] / 1000.0)
        width = (max(0.5 * t_x, 0.005) if _auto(ec.get("crossover_width_ms")) else ec["crossover_width_ms"] / 1000.0)
        xover = (t_direct + t_x, width)
        info.update({"room_volume_m3": round(room.V, 2), "room_surface_m2": round(room.S, 2),
                     "t60_room_bands": dict(zip(ac.OCTAVE_BANDS.tolist(), room.T.round(3).tolist())),
                     "mixing_time_ms": round(room.t_mix * 1000, 1), "crossover_ms": round(t_x * 1000, 1),
                     "crossover_width_ms": round(width * 1000, 1), "schroeder_hz": round(room.schroeder, 1),
                     "source_receiver_distance_m": round(room.r, 3), "speed_of_sound": round(room.c, 2),
                     "first_reflection_ms": round((room.t_first - room.t_direct) * 1000, 2)})

    def xwin(t):
        """ISM fade-out window (1 -> 0 around the crossover)."""
        tc, w = xover
        x = np.clip((t - (tc - w / 2)) / w, 0.0, 1.0)
        return np.cos(0.5 * np.pi * x)  # amplitude window; energy window = cos^2

    src_kind = ec["source"]
    rain = None
    if src_kind == "room":
        beta_db = 10 * np.log10(1 - room.alpha) + 10 * np.log10(1 - room.scat)
        t_max = xover[0] + xover[1] / 2
        mo = None if _auto(cfg["room"].get("max_order")) else int(cfg["room"]["max_order"])
        jit = float(cfg["room"].get("image_jitter_m", 0.0))
        for ch in range(N):
            if ch > 0 and er_mode == "same":
                ir_early[ch] = ir_early[0]
                continue
            r_ = rooms[ch]
            # 'jitter': same image-position randomisation for all channels (only the mic moves)
            irng = ch_rng("img", ch) if er_mode == "independent" else np.random.default_rng(seeds[0])
            imgs, Nord = shoebox_images(r_.dims, r_.src, r_.rcv, beta_db, r_.c, t_max, mo,
                                        r_.src_dir, r_.rcv_dir, jit, irng)
            sel = imgs["order"] > 0
            amps = imgs["amp"][sel] * xwin(imgs["time"][sel]) * eg
            nimg = renderer.render(ir_early[ch], to_samples(imgs["time"][sel]), amps, imgs["band_db"][sel],
                                   imgs["dist"][sel], etone)
            if ch == 0:
                info.update({"image_sources": int(nimg), "ism_max_order": int(Nord)})
                # "Diffuse rain": energy removed from the specular images by scattering is
                # re-injected into the tail, smeared causally over ~half a mean free time.
                air_oct = ac.air_absorption_db_per_m(ac.OCTAVE_BANDS, air_cfg.get("temperature_c", 20.0),
                                                     air_cfg.get("humidity_pct", 50.0), air_cfg.get("pressure_kpa", 101.325))
                cnt, dist0, amp0 = imgs["counts"][sel], imgs["dist"][sel], imgs["amp"][sel]
                a_db = cnt @ (10 * np.log10(1 - room.alpha)) - dist0[:, None] * air_oct[None, :]
                s_db = cnt @ beta_db - dist0[:, None] * air_oct[None, :]
                e_sc = (amp0 ** 2)[:, None] * (10 ** (a_db / 10) - 10 ** (s_db / 10))
                dtb = 0.0005
                nb = int(np.ceil(t_max / dtb)) + 2
                bi = np.minimum((imgs["time"][sel] / dtb).astype(int), nb - 1)
                H = np.stack([np.bincount(bi, weights=e_sc[:, k], minlength=nb) for k in range(e_sc.shape[1])]) / dtb
                tau = float(np.clip(0.5 * 4 * room.V / (room.c * room.S), 0.002, 0.03))
                aa = np.exp(-dtb / tau)
                H = signal.lfilter([1 - aa], [1, -aa], H, axis=1)
                rain = (np.arange(nb) * dtb, H / fs)  # per-sample energy
                info["diffuse_rain_tau_ms"] = round(tau * 1000, 1)
    elif src_kind in ("parametric", "list"):
        tj = float(mc.get("time_jitter_ms", 0.4)) / 1000.0
        gj = float(mc.get("gain_jitter_db", 1.0))
        base = None
        for ch in range(N):
            if ch > 0 and er_mode == "same":
                ir_early[ch] = ir_early[0]
                continue
            if src_kind == "parametric" and (ch == 0 or er_mode == "independent"):
                refl = parametric_early(ec.get("parametric", {}), T_target, ch_rng("er", ch), c)
            elif src_kind == "list" and ch == 0:
                refl = list_early(ec.get("list", []), c)
            else:
                refl = dict(base)
                jr = ch_rng("er", ch)
                refl["time"] = np.maximum(base["time"] + jr.normal(0, tj, base["time"].size), 1e-4)
                refl["gain"] = base["gain"] * 10 ** (jr.normal(0, gj, base["gain"].size) / 20.0)
            if ch == 0:
                base = refl
            if refl is not None:
                nref = renderer.render(ir_early[ch], to_samples(t_direct + refl["time"]), a_d * refl["gain"] * eg,
                                       refl["band_db"], refl["air_dist"], etone)
                if ch == 0:
                    info["early_reflections"] = int(nref)
    elif src_kind != "none":
        raise ValueError("early.source must be room | parametric | list | none")

    # ---------------- late tail ----------------
    ir_late = np.zeros((N, n))
    lkind = lc["source"]
    if lkind != "none":
        noise_kind = lc.get("noise", "auto")
        if noise_kind == "auto":
            noise_kind = "gaussian" if lkind == "room" else "velvet"
        if lkind == "room":
            onset_emit = room.t_first - 0.0005
        else:
            if _auto(lc.get("onset_ms")):
                if src_kind == "parametric":
                    onset_rel = ec["parametric"].get("start_ms", 6.0) / 1000.0
                elif src_kind == "list" and ec.get("list"):
                    onset_rel = min(e["time_ms"] for e in ec["list"]) / 1000.0
                else:
                    onset_rel = 0.002
            else:
                onset_rel = lc["onset_ms"] / 1000.0
            onset_emit = t_direct + onset_rel
        onset_idx = int(np.clip(np.floor(to_samples(onset_emit)), 0, n - 1))
        dens = lc.get("density", {})
        exc = np.zeros((N, n))
        for ch in range(N):
            r_ = ch_rng("late", ch)
            if noise_kind == "velvet":
                full = dens.get("full_ms", "auto")
                full_s = (room.t_mix if room else 0.08) if _auto(full) else full / 1000.0
                dmax = dens.get("max_per_s", "auto")
                dmax = 0.25 * fs if _auto(dmax) else float(dmax)
                exc[ch] = velvet_growing(n, fs, onset_idx, r_, dens.get("initial_per_s", 300.0),
                                         dmax, full_s, dens.get("exponent", 2.0), xfade_s=max(0.01, 0.3 * full_s),
                                         amp=dens.get("pulse_amplitude", "gaussian"))
            elif noise_kind == "gaussian":
                exc[ch] = gaussian(n, onset_idx, r_)
            else:
                raise ValueError("late.noise must be gaussian | velvet | auto")

        centers = band_centers(fs, int(lc.get("bands_per_octave", 3)))
        tone_c = ac.curve_from_spec(lc.get("tone"), centers, default=0.0)
        lg = 10 ** (lc.get("gain_db", 0.0) / 20.0)
        attack = max(float(lc.get("attack_ms", 15.0)) / 1000.0, 1e-4)
        ashape = float(lc.get("attack_shape", 1.0))
        t_rel = t_emit - onset_emit
        if lkind == "room":
            ramp = np.clip(t_rel / 0.001, 0, 1)  # room: natural build-up comes from the scattering model
        else:
            x = np.clip(t_rel / attack, 0.0, 1.0) ** ashape
            ramp = np.sin(0.5 * np.pi * x)
        ramp[t_rel < 0] = 0.0
        te = np.maximum(t_emit, 0.0)
        tsd = np.maximum(t_emit - t_direct, 0.0)
        if lkind == "room":
            w2 = xwin(t_emit) ** 2 if xover else np.zeros(n)
            base_e = 4 * np.pi * room.c / room.V * room.diffuse_gain / fs
            nr = int(np.flatnonzero(w2 > 0)[-1]) + 1 if np.any(w2 > 0) else 0
            lgo = np.log2(ac.OCTAVE_BANDS)

        def env_fn(i, fc):
            T = float(t60_fn(fc))
            if lkind == "room":
                E = base_e * np.exp(-13.8155 * te / T)
                if T2_bands is not None:
                    E = E + base_e * 10 ** (ds.get("level_db", -18.0) / 10) * np.exp(-13.8155 * te / float(t60b_fn(fc)))
                E = E * (1.0 - w2)
                if rain is not None and nr > 0:
                    # early region: actual scattered energy; late region: statistical decay
                    pos = np.clip(np.interp(np.log2(fc), lgo, np.arange(lgo.size)), 0, lgo.size - 1)
                    j = int(min(np.floor(pos), lgo.size - 2))
                    fr = pos - j
                    Hk = (1 - fr) * rain[1][j] + fr * rain[1][j + 1]
                    E[:nr] += w2[:nr] * np.interp(t_emit[:nr], rain[0], Hk, left=0.0, right=0.0)
            else:
                E = np.exp(-13.8155 * tsd / T)
                if T2_bands is not None:
                    E = E + 10 ** (ds.get("level_db", -18.0) / 10) * np.exp(-13.8155 * tsd / float(t60b_fn(fc)))
            return np.sqrt(np.maximum(E, 0.0)) * ramp * 10 ** (tone_c[i] / 20.0)

        fl = lc.get("envelope_flatten", True)
        flat = (int(lc.get("bands_per_octave", 3)), onset_idx, float(lc.get("flatten_cycles", 8.0))) if fl else None
        if N == 1:
            ir_late = shape_noise(exc[0], fs, centers, env_fn, flat)[None, :]
        else:
            model = ccfg.get("model", "independent")
            mix_fn = None
            if model != "independent" or ccfg.get("coherent_blend") is not None:
                def mix_fn(fc):
                    return mixing_matrix(coherence_matrix(ccfg, fc, N, offsets, c))
            ir_late = shape_noise_multi(exc, fs, centers, env_fn, mix_fn, flat)
            info["coherence_model"] = model
        del exc
        ir_late = apply_eq(ir_late, fs, cfg.get("eq_late"))
        drr = lc.get("drr_db")
        drr = None if _auto(drr) else drr
        if lkind == "parametric" and drr is None:
            drr = 0.0
        if drr is not None:
            e_tail = np.mean(np.sum(ir_late ** 2, axis=1))
            if e_tail > 0:
                ir_late *= np.sqrt(a_d ** 2 / 10 ** (drr / 10.0) / e_tail)
        ir_late *= lg
        info["late_noise"] = noise_kind
        info["late_bands"] = int(centers.size)

    # theory-based DRR (energy), channel 0, before the modal split
    e_dir = np.sum(ir_direct[0] ** 2)
    e_rev = np.sum(ir_early[0] ** 2) + np.sum(ir_late[0] ** 2)
    ir = ir_direct + ir_early + ir_late
    del ir_direct, ir_early, ir_late

    # ---------------- modal low end ----------------
    mcfg = cfg["room"].get("modal", {})
    modal_on = False
    if room is not None and src_kind == "room" and lkind == "room" and dc.get("enabled", True):
        en = mcfg.get("enabled", "auto")
        modal_on = (room.schroeder > 60.0) if en == "auto" else bool(en)
    if modal_on:
        fc = float(np.clip(room.schroeder, 50, 400)) if _auto(mcfg.get("crossover_hz")) else float(mcfg["crossover_hz"])
        lo = np.zeros((N, n))
        nmodes = 0
        for ch in range(N):
            if ch > 0 and rooms[ch] is rooms[0]:
                lo[ch] = lo[0]
                continue
            r_ = rooms[ch]
            hm, nmodes = modal_ir(n, fs, shift, r_.dims, r_.src, r_.rcv, r_.c, 2.5 * fc, t60_fn, r_.V,
                                  rng_mod, float(mcfg.get("detune_cents", 0.0)))
            g0 = r_.t_direct - min(0.001, 0.25 / fc)  # causal gate before the direct sound
            gate = np.clip((t_emit - g0) / (r_.t_direct - g0), 0, 1)
            lo[ch] = hm * np.sin(0.5 * np.pi * gate) ** 2 * 10 ** (mcfg.get("gain_db", 0.0) / 20.0)
        # Early (correlated) part: Linkwitz-Riley (amplitude complementary).
        # Late (uncorrelated) part: Butterworth (power complementary) -> no -3 dB hole at fc.
        lp_lr, hp_lr = lr4(fs, fc)
        lp_bw = signal.butter(4, fc, "low", fs=fs, output="sos")
        hp_bw = signal.butter(4, fc, "high", fs=fs, output="sos")
        tc, wd = xover
        W = np.cos(0.5 * np.pi * np.clip((t_emit - (tc - wd / 2)) / wd, 0, 1)) ** 2
        ir = (signal.sosfilt(hp_lr, ir * W) + signal.sosfilt(hp_bw, ir * (1 - W))
              + signal.sosfilt(lp_lr, lo * W) + signal.sosfilt(lp_bw, lo * (1 - W)))
        e_rev += np.sum(lo[0] ** 2)
        del lo
        info.update({"modal_crossover_hz": round(fc, 1), "modes": int(nmodes)})

    ir = apply_eq(ir, fs, cfg.get("eq"))
    ir = dc_block(ir, fs, oc.get("dc_highpass_hz", 0.0))
    fo = int(oc.get("fade_out_ms", 30.0) * fs / 1000.0)
    if fo > 0:
        ir[:, -fo:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fo)))
    if e_dir > 0 and e_rev > 0:
        info["drr_components_db"] = round(10 * np.log10(e_dir / e_rev), 2)

    norm = oc.get("normalize", "peak")   # one common gain for all channels
    if norm == "peak":
        g = 10 ** (oc.get("peak_dbfs", -1.0) / 20.0) / max(np.max(np.abs(ir)), 1e-12)
    elif norm == "energy":
        g = 1.0 / np.sqrt(max(np.mean(np.sum(ir ** 2, axis=1)), 1e-24))
    else:
        g = 1.0
    ir *= g
    info["normalize_gain_db"] = round(20 * np.log10(g), 2)
    info["direct_index"] = lead if dc.get("enabled", True) else None
    if N > 1:
        info["late_start_s"] = float((xover[0] + xover[1] / 2 - shift) if xover else
                                     (t_direct - shift + max(0.08, float(ec.get("parametric", {}).get("end_ms", 70)) / 1000.0)))
        info["receiver_offsets_m"] = np.round(pos_offsets if not has_array else offsets, 4).tolist()
        info["early_mode"], info["direct_mode"] = er_mode, dir_mode
    return (ir[0] if N == 1 else ir), fs, info
