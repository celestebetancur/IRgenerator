"""Command line interface.

    python -m irgen render presets/concert_hall.json -o renders
    python -m irgen render presets/concert_hall.json --set late.t60_scale=1.3 --set seed=7
    python -m irgen analyze some_ir.wav
    python -m irgen materials
    python -m irgen gui
    python -m irgen strange presets/strange/inside_a_basketball.json --set space.scale=300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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


def render_preset(preset, outdir, name=None, quiet=False, progress=None, analysis=None):
    """Render a preset dict, write wav(s) and json; with analysis (default: output.write_report)
    also the analysis report(s). Returns a summary dict."""
    from .engine import generate
    from .export import write_json, write_wav

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
    jpath = os.path.join(outdir, f"{base}.json")
    write_json(jpath, {"info": info, "resolved_preset": cfg})
    files["json"] = jpath
    res = {"files": files, "info": info, "analysis": None, "multichannel_analysis": None, "table": "",
           "ir": ir, "fs": fs, "cfg": cfg, "outdir": outdir, "base": base}
    if oc.get("write_report", True) if analysis is None else analysis:
        res.update(analyze_render(ir, fs, info, cfg, outdir, base))
        files.update(report=res["report"], mc_report=res["mc_report"])
    if not quiet:
        if res["table"]:
            print(res["table"])
        if res["multichannel_analysis"]:
            m = res["multichannel_analysis"]
            print("decorrelation:", {k: round(v, 3) for k, v in m["decorrelation"].items()})
            print("similarity:", {k: (round(v, 2) if not isinstance(v, dict) else v) for k, v in m["similarity"].items()})
        print(f"-> {files['wav'][0]}  ({ir.shape[-1] / fs:.2f} s, rendered in {dt:.1f} s)")
    return res


def analyze_render(ir, fs, info, cfg, outdir, base):
    """ISO 3382 analysis + report PNG (and the multichannel report) of a rendered IR;
    adds the results to <base>.json. Separate from rendering so it can run on request."""
    from .analysis import analyze, plot_report, summary_table, to_json
    from .export import write_json
    from .acoustics import OCTAVE_BANDS, speed_of_sound

    multi = ir.ndim == 2
    ref = ir if not multi else ir[0]
    res = analyze(ref, fs)
    out = {"info": info, "resolved_preset": cfg, "analysis": to_json(res)}
    rep = os.path.join(outdir, f"{base}_report.png")
    plot_report(ref, fs, res, rep, title=f"{base}{' (channel 1)' if multi else ''}  ({fs} Hz, {ref.size / fs:.2f} s)",
                target_t60=(OCTAVE_BANDS, list(info["t60_bands"].values())))
    mrep = None
    if multi and "multichannel" in cfg:
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
    write_json(os.path.join(outdir, f"{base}.json"), out)
    return {"analysis": out["analysis"], "multichannel_analysis": out.get("multichannel_analysis"),
            "table": summary_table(res), "report": rep, "mc_report": mrep}


def render(preset_path, outdir, overrides=(), name=None, quiet=False, analysis=None):
    with open(preset_path) as fh:
        preset = json.load(fh)
    for ov in overrides:
        k, v = ov.split("=", 1)
        _set_path(preset, k, v)
    if name is None:
        name = preset.get("name") or os.path.splitext(os.path.basename(preset_path))[0]
    return render_preset(preset, outdir, name, quiet, analysis=analysis)["files"]["wav"][0]


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
    r.add_argument("--no-analysis", action="store_true", help="skip the analysis report (faster)")
    a = sub.add_parser("analyze", help="analyse an existing IR wav")
    a.add_argument("wav")
    a.add_argument("--plot", default=None)
    sg = sub.add_parser("strange", help="render strange-space presets (bizarre / tonal / gigantified spaces)")
    sg.add_argument("presets", nargs="+")
    sg.add_argument("-o", "--outdir", default="renders/strange")
    sg.add_argument("--set", action="append", default=[], metavar="key.path=value")
    sg.add_argument("--import-ir", default=None, metavar="WAV", help="IR for the imported layer")
    sg.add_argument("--no-analysis", action="store_true", help="skip the analysis report (faster)")
    sv = sub.add_parser("serve", help="headless IR-set API for plugins / hosts (http://127.0.0.1:PORT/v1)")
    sv.add_argument("--port", type=int, default=8766)
    st = sub.add_parser("set", help="render a set of N decorrelated IRs from a preset id (prints the manifest)")
    st.add_argument("preset", help="<engine>/<name>, e.g. strange/dorian_cistern or standard/concert_hall")
    st.add_argument("-n", "--channels", type=int, default=None)
    st.add_argument("--set", action="append", default=[], metavar="key.path=value")
    st.add_argument("--sample-rate", type=int, default=None)
    st.add_argument("--seed", type=int, default=None)
    st.add_argument("--normalize", default="energy", choices=["energy", "peak", "none"])
    st.add_argument("--layout", default="separate", choices=["separate", "interleaved", "both"])
    st.add_argument("--preview", action="store_true", help="fast preview variant")
    st.add_argument("--import-ir", default=None, metavar="WAV")
    bl = sub.add_parser("blend", help="blend two multichannel IR folders into a new one (min(A, B) channels)")
    bl.add_argument("name", help="name of the new folder (renders/blends/<name>)")
    bl.add_argument("a", help="IR folder A: a path, a set id, set/<id>, blend/<name>, gui/<name> or strange/<name>")
    bl.add_argument("b", help="IR folder B (same forms)")
    bl.add_argument("--mode", default="morph", choices=["morph", "mix"])
    bl.add_argument("--amount", type=float, default=0.5, help="0 = A, 1 = B")
    bl.add_argument("--normalize", default="energy", choices=["energy", "peak", "none"])
    bl.add_argument("--layout", default="separate", choices=["separate", "interleaved", "both"])
    bl.add_argument("--format", default="FLOAT", choices=["FLOAT", "PCM_16", "PCM_24", "PCM_32"])
    bl.add_argument("--force", action="store_true", help="overwrite an existing blend")
    bl.add_argument("-o", "--outdir", default=None, help="parent folder (default renders/blends)")
    sub.add_parser("materials", help="list the material database")
    g = sub.add_parser("gui", help="open the browser GUI")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "render":
        for p in args.presets:
            render(p, args.outdir, args.set, args.name if len(args.presets) == 1 else None, args.quiet,
                   analysis=False if args.no_analysis else None)
    elif args.cmd == "analyze":
        import soundfile as sf
        from .analysis import analyze, plot_report, summary_table
        x, fs = sf.read(args.wav, always_2d=True)
        res = analyze(x[:, 0], fs)
        print(summary_table(res))
        if args.plot:
            plot_report(x[:, 0], fs, res, args.plot, title=os.path.basename(args.wav))
    elif args.cmd == "strange":
        from .strange import render_strange
        imported = None
        if args.import_ir:
            import soundfile as sf
            x, ifs = sf.read(args.import_ir, always_2d=True)
            imported = (x.T, ifs)
        for p in args.presets:
            with open(p) as fh:
                preset = json.load(fh)
            for ov in args.set:
                k, v = ov.split("=", 1)
                _set_path(preset, k, v)
            name = preset.get("name") or os.path.splitext(os.path.basename(p))[0]
            render_strange(preset, args.outdir, name, imported, quiet=False, analysis=not args.no_analysis)
    elif args.cmd == "serve":
        from .service import serve
        serve(port=args.port)
    elif args.cmd == "set":
        from .service import render_set, resolve_request
        ov = {}
        for o in args.set:
            k, v = o.split("=", 1)
            try:
                ov[k] = json.loads(v)
            except json.JSONDecodeError:
                ov[k] = v
        canon = resolve_request({"preset": args.preset, "overrides": ov, "channels": args.channels,
                                 "sample_rate": args.sample_rate, "seed": args.seed, "normalize": args.normalize,
                                 "layout": args.layout, "import_ir": args.import_ir})
        m = render_set(canon, "preview" if args.preview else "full")
        print(json.dumps({k: m[k] for k in ("set_id", "stage", "channels", "sample_rate", "length_s", "direct_index",
                                            "render_seconds", "cached") if k in m}, indent=1))
        print(f"-> {Path(m['files'][0]['path']).parent if m['files'] else m['interleaved']['path']}")
    elif args.cmd == "blend":
        from .blend import BlendError, blend_folders
        try:
            m, _ = blend_folders(args.name, args.a, args.b, args.mode, args.amount, args.normalize, args.layout,
                                 args.format, args.force, args.outdir)
        except BlendError as e:
            print(f"irgen blend: {e}{' (use --force)' if 'exists' in str(e) else ''}", file=sys.stderr)
            return 2
        print(f"{args.mode} {args.amount:g}: {m['blend']['a']['channels']} + {m['blend']['b']['channels']} ch -> "
              f"{m['channels']} ch, {m['length_s']:.2f} s ({m['render_seconds']:.1f} s)")
        print(f"-> {Path(m['files'][0]['path']).parent if m['files'] else m['interleaved']['path']}")
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
