"""Command line interface.

    python -m irgen render presets/concert_hall.json -o renders
    python -m irgen render presets/concert_hall.json --set late.t60_scale=1.3 --set seed=7
    python -m irgen analyze some_ir.wav
    python -m irgen materials
    python -m irgen gui
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _set_path(d, path, value):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass
    d[keys[-1]] = value


def render_preset(preset, outdir, name=None, quiet=False, progress=None):
    """Render a preset dict, write wav(s), json and report(s). Returns a summary dict."""
    from .analysis import analyze, plot_report, summary_table, to_json
    from .engine import generate
    from .export import write_json, write_wav
    from .acoustics import OCTAVE_BANDS, speed_of_sound

    t0 = time.time()
    ir, fs, info, cfg = generate(preset)
    dt = time.time() - t0
    base = name or cfg.get("name") or "ir"
    os.makedirs(outdir, exist_ok=True)
    oc = cfg["output"]
    fmt, dith, seed = oc.get("format", "PCM_24"), oc.get("dither", True), cfg.get("seed", 1)
    files = {"wav": [], "report": None, "mc_report": None}
    multi = ir.ndim == 2
    if not multi:
        wav = os.path.join(outdir, f"{base}.wav")
        write_wav(wav, ir, fs, fmt, dith, seed)
        files["wav"].append(wav)
    else:
        mode = cfg["multichannel"].get("export", "interleaved")
        if mode in ("interleaved", "both"):
            wav = os.path.join(outdir, f"{base}_{ir.shape[0]}ch.wav")
            write_wav(wav, ir.T, fs, fmt, dith, seed)
            files["wav"].append(wav)
        if mode in ("separate", "both"):
            sub = os.path.join(outdir, f"{base}_channels")
            os.makedirs(sub, exist_ok=True)
            for ch in range(ir.shape[0]):
                w = os.path.join(sub, f"{base}_ch{ch + 1:02d}.wav")
                write_wav(w, ir[ch], fs, fmt, dith, seed + ch)
                files["wav"].append(w)
        files["channel_preview"] = []
    info["render_seconds"] = round(dt, 2)
    out = {"info": info, "resolved_preset": cfg}
    table = ""
    if oc.get("write_report", True):
        ref = ir if not multi else ir[0]
        res = analyze(ref, fs)
        out["analysis"] = to_json(res)
        tgt = (OCTAVE_BANDS, list(info["t60_bands"].values()))
        rep = os.path.join(outdir, f"{base}_report.png")
        plot_report(ref, fs, res, rep, title=f"{base}{' (channel 1)' if multi else ''}  ({fs} Hz, {ref.size / fs:.2f} s)",
                    target_t60=tgt)
        files["report"] = rep
        table = summary_table(res)
        if multi:
            from .mc_analysis import analyze_multichannel, mc_to_json, plot_mc_report
            from .multichannel import array_offsets, coherence_matrix, mixing_matrix
            mc = cfg["multichannel"]
            Nn = ir.shape[0]
            offs, _ = array_offsets(mc.get("array"), Nn)
            cc = mc.get("coherence", {"model": "independent"})
            c = speed_of_sound(cfg["room"].get("air", {}).get("temperature_c", 20.0))

            def target_fn(fc):
                M = mixing_matrix(coherence_matrix(cc, fc, Nn, offs, c))
                return M @ M.T
            mres = analyze_multichannel(ir, fs, info["late_start_s"], target_fn)
            out["multichannel_analysis"] = mc_to_json(mres)
            mrep = os.path.join(outdir, f"{base}_multichannel_report.png")
            plot_mc_report(ir, fs, mres, mrep, title=f"{base}: {Nn} channels, coherence model '{cc.get('model')}'")
            files["mc_report"] = mrep
    jpath = os.path.join(outdir, f"{base}.json")
    write_json(jpath, out)
    files["json"] = jpath
    if not quiet:
        if table:
            print(table)
        if "multichannel_analysis" in out:
            m = out["multichannel_analysis"]
            print("decorrelation:", {k: round(v, 3) for k, v in m["decorrelation"].items()})
            print("similarity:", {k: (round(v, 2) if not isinstance(v, dict) else v) for k, v in m["similarity"].items()})
        print(f"-> {files['wav'][0]}  ({ir.shape[-1] / fs:.2f} s, rendered in {dt:.1f} s)")
    return {"files": files, "info": info, "analysis": out.get("analysis"),
            "multichannel_analysis": out.get("multichannel_analysis"), "table": table, "ir": ir, "fs": fs}


def render(preset_path, outdir, overrides=(), name=None, quiet=False):
    with open(preset_path) as fh:
        preset = json.load(fh)
    for ov in overrides:
        k, v = ov.split("=", 1)
        _set_path(preset, k, v)
    if name is None:
        name = preset.get("name") or os.path.splitext(os.path.basename(preset_path))[0]
    return render_preset(preset, outdir, name, quiet)["files"]["wav"][0]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="irgen", description="Parametric impulse response generator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render one or more presets")
    r.add_argument("presets", nargs="+")
    r.add_argument("-o", "--outdir", default="renders")
    r.add_argument("--set", action="append", default=[], metavar="key.path=value",
                   help="override any preset value, e.g. --set late.t60_scale=1.4")
    r.add_argument("--name", default=None)
    r.add_argument("-q", "--quiet", action="store_true")
    a = sub.add_parser("analyze", help="analyse an existing IR wav")
    a.add_argument("wav")
    a.add_argument("--plot", default=None)
    sub.add_parser("materials", help="list the material database")
    g = sub.add_parser("gui", help="open the browser GUI")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "render":
        for p in args.presets:
            render(p, args.outdir, args.set, args.name if len(args.presets) == 1 else None, args.quiet)
    elif args.cmd == "analyze":
        import soundfile as sf
        from .analysis import analyze, plot_report, summary_table
        x, fs = sf.read(args.wav, always_2d=True)
        res = analyze(x[:, 0], fs)
        print(summary_table(res))
        if args.plot:
            plot_report(x[:, 0], fs, res, args.plot, title=os.path.basename(args.wav))
    elif args.cmd == "gui":
        from .gui import serve
        serve(port=args.port, open_browser=not args.no_browser)
    elif args.cmd == "materials":
        from .acoustics import MATERIALS, OCTAVE_BANDS
        print(f"{'material':20s} " + " ".join(f"{int(b):>6d}" for b in OCTAVE_BANDS))
        for k, v in MATERIALS.items():
            print(f"{k:20s} " + " ".join(f"{x:6.2f}" for x in v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
