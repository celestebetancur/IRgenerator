"""Local browser GUI:  python -m irgen gui   ->  http://127.0.0.1:8765

Pure standard-library HTTP server; the page talks to the same engine as the CLI."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRESETS = ROOT / "presets"
OUT = ROOT / "renders" / "gui"
STATIC = Path(__file__).resolve().parent / "gui_static"

_lock = threading.Lock()
_last = {"ir": None, "fs": None, "name": None}


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name or "untitled").strip("_") or "untitled"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ---------------------------------------------------------------- dry test signals
def dry_signal(kind, fs, seconds=6.0):
    rng = np.random.default_rng(0)
    n = int(seconds * fs)
    x = np.zeros(n)
    if kind == "clicks":
        for t in np.arange(0.2, seconds - 1.5, 1.2):
            x[int(t * fs)] = 1.0
    elif kind == "drums":
        t = np.arange(int(0.25 * fs)) / fs
        kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-t * 30)) * t) * np.exp(-t * 18)
        snare = (rng.standard_normal(t.size) * np.exp(-t * 28) + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 35))
        hat = rng.standard_normal(int(0.05 * fs)) * np.exp(-np.arange(int(0.05 * fs)) / fs * 90)
        hat = np.diff(hat, prepend=0)
        beat = 0.42
        for i in range(10):
            p = int((0.2 + i * beat) * fs)
            seg = kick if i % 2 == 0 else snare * 0.6
            x[p:p + seg.size] += seg[: max(0, min(seg.size, n - p))]
            q = int((0.2 + i * beat + beat / 2) * fs)
            x[q:q + hat.size] += 0.3 * hat[: max(0, min(hat.size, n - q))]
    elif kind == "plucks":
        notes = [0, 4, 7, 12, 7, 4, 0, -5]
        for i, semi in enumerate(notes):
            f0 = 220 * 2 ** (semi / 12)
            L = int(fs / f0)
            buf = rng.uniform(-1, 1, L)
            m = int(0.6 * fs)
            y = np.zeros(m)
            for k in range(m):
                y[k] = buf[k % L]
                buf[k % L] = 0.996 * 0.5 * (buf[k % L] + buf[(k + 1) % L])
            p = int((0.2 + i * 0.45) * fs)
            x[p:p + m] += y[: max(0, min(m, n - p))]
    else:  # "voice-like": pulse train through formants, syllables
        from scipy import signal
        t = np.arange(n) / fs
        f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
        ph = np.cumsum(f0 / fs)
        src = (np.diff(np.floor(ph), prepend=0) > 0).astype(float)
        env = (np.sin(2 * np.pi * 2.2 * t) > 0.1) * (t < seconds - 1.5)
        y = np.zeros(n)
        for fmt, bw, g in ((700, 110, 1.0), (1220, 120, 0.5), (2600, 160, 0.25)):
            b, a = signal.iirpeak(fmt, fmt / bw, fs=fs)
            y += g * signal.lfilter(b, a, src)
        x = y * env
    return x / max(np.max(np.abs(x)), 1e-9) * 0.5


def _wav_bytes(x, fs):
    import soundfile as sf
    bio = io.BytesIO()
    sf.write(bio, x, fs, subtype="PCM_16", format="WAV")
    return bio.getvalue()


# ---------------------------------------------------------------- handler
class Handler(BaseHTTPRequestHandler):
    server_version = "irgen"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, body, ctype="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(_jsonable(body)).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    # ---- GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/meta":
                from .acoustics import MATERIALS, OCTAVE_BANDS, PATTERNS
                from .config import DEFAULTS
                presets = sorted(p.stem for p in PRESETS.glob("*.json"))
                return self._send(200, {"defaults": DEFAULTS, "materials": MATERIALS, "bands": OCTAVE_BANDS,
                                        "patterns": list(PATTERNS), "presets": presets})
            if u.path == "/api/preset":
                name = _safe(q.get("name", [""])[0])
                return self._send(200, json.loads((PRESETS / f"{name}.json").read_text()))
            if u.path.startswith("/files/"):
                rel = unquote(u.path[len("/files/"):])
                p = (OUT / rel).resolve()
                if not str(p).startswith(str(OUT.resolve())) or not p.is_file():
                    return self._send(404, {"error": "not found"})
                ctype = {".png": "image/png", ".wav": "audio/wav", ".json": "application/json"}.get(p.suffix, "application/octet-stream")
                return self._send(200, p.read_bytes(), ctype)
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": str(e)})

    # ---- POST
    def do_POST(self):
        u = urlparse(self.path)
        try:
            data = self._body()
            if u.path == "/api/render":
                return self._render(data)
            if u.path == "/api/save_preset":
                name = _safe(data.get("name"))
                path = PRESETS / f"{name}.json"
                if path.exists() and not data.get("overwrite"):
                    return self._send(409, {"error": f"preset '{name}' exists", "exists": True})
                preset = dict(data["preset"])
                preset["name"] = name
                path.write_text(json.dumps(preset, indent=2))
                return self._send(200, {"ok": True, "name": name})
            if u.path == "/api/audition":
                return self._audition(data)
            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def _render(self, data):
        from .cli import render_preset
        preset = data["preset"]
        name = _safe(data.get("name") or preset.get("name"))
        outdir = OUT / name
        if not _lock.acquire(blocking=False):
            return self._send(429, {"error": "a render is already running"})
        try:
            t0 = time.time()
            res = render_preset(preset, str(outdir), name, quiet=True)
        finally:
            _lock.release()
        ir, fs = res["ir"], res["fs"]
        _last.update(ir=ir, fs=fs, name=name)
        stamp = str(int(time.time() * 1000))

        def url(p):
            return None if not p else f"/files/{Path(p).resolve().relative_to(OUT.resolve()).as_posix()}?t={stamp}"
        # browser preview: mono, or channel 1/2 as stereo
        prev = ir if ir.ndim == 1 else ir[:2].T
        (outdir / "_preview.wav").write_bytes(_wav_bytes(prev / max(np.max(np.abs(prev)), 1e-9) * 0.89, fs))
        files = res["files"]
        # envelope thumbnail data (dB, 2 ms RMS) for the inline plot
        x = ir if ir.ndim == 1 else ir[0]
        hop = max(1, int(0.002 * fs))
        m = (x[: x.size // hop * hop].reshape(-1, hop) ** 2).mean(axis=1)
        env = 10 * np.log10(m / max(m.max(), 1e-30) + 1e-14)
        return self._send(200, {
            "name": name, "seconds": round(time.time() - t0, 2),
            "info": res["info"], "analysis": res["analysis"], "multichannel": res["multichannel_analysis"],
            "table": res["table"],
            "files": {"wav": [url(w) for w in files["wav"][:3]], "n_wav": len(files["wav"]),
                      "wav_dir": str(Path(files["wav"][0]).parent),
                      "report": url(files.get("report")), "mc_report": url(files.get("mc_report")),
                      "preview": url(outdir / "_preview.wav"), "json": url(files.get("json"))},
            "envelope": {"hop_s": hop / fs, "db": np.round(np.maximum(env, -120), 1).tolist()},
        })

    def _audition(self, data):
        from scipy.signal import fftconvolve
        import soundfile as sf
        if _last["ir"] is None:
            return self._send(400, {"error": "render an IR first"})
        ir, fs = _last["ir"], _last["fs"]
        if data.get("dry_b64"):
            raw = base64.b64decode(data["dry_b64"].split(",")[-1])
            dry, dfs = sf.read(io.BytesIO(raw), always_2d=True)
            dry = dry.mean(axis=1)
            if dfs != fs:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(int(fs), int(dfs))
                dry = resample_poly(dry, int(fs) // g, int(dfs) // g)
            dry = dry[: int(30 * fs)]
        else:
            dry = dry_signal(data.get("dry", "drums"), fs)
        mix = float(data.get("wet", 0.5))
        if ir.ndim == 1:
            chans = [ir]
        else:
            l, r = int(data.get("left", 0)), int(data.get("right", 1))
            chans = [ir[min(l, ir.shape[0] - 1)], ir[min(r, ir.shape[0] - 1)]]
        wet = np.stack([fftconvolve(dry, h) for h in chans], axis=1)
        wet /= max(np.sqrt(np.mean(wet ** 2)), 1e-12)
        d = np.zeros_like(wet)
        d[: dry.size] = dry[:, None] / max(np.sqrt(np.mean(dry ** 2)), 1e-12)
        y = np.sqrt(1 - mix) * d + np.sqrt(mix) * wet
        y = y / max(np.max(np.abs(y)), 1e-9) * 0.89
        path = OUT / _last["name"] / "_audition.wav"
        path.write_bytes(_wav_bytes(y if y.shape[1] > 1 else y[:, 0], fs))
        return self._send(200, {"url": f"/files/{_last['name']}/_audition.wav?t={int(time.time() * 1000)}"})


def serve(port=8765, open_browser=True):
    OUT.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"irgen GUI running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
