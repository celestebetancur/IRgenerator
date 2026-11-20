"""IR-set service: sets of N decorrelated IRs rendered from irgen presets, for hosts that
load and hot-swap them (e.g. a convolution / auralization plugin).

Versioned JSON API (v1), served on its own by `python -m irgen serve` and also mounted
under /v1 in the GUI server. Everything binds to 127.0.0.1.

    GET    /v1                         api / version / endpoints
    GET    /v1/presets                 [{id, engine, name, about, channels, sample_rate}]
    GET    /v1/presets/<engine>/<name> preset JSON
    POST   /v1/sets                    request -> 202 {job} (or 200 manifest with "wait": true)
    GET    /v1/jobs | /v1/jobs/<id>    job status; manifest(s) when ready
    DELETE /v1/jobs/<id>               cancel a queued job / forget a finished one
    GET    /v1/sets | /v1/sets/<id>    cached sets / one manifest
    GET    /v1/sets/<id>/<file>        a WAV of the set
    DELETE /v1/sets/<id>               remove a cached set
    POST   /v1/blends                  {name, a, b, mode, amount} -> 202 {job} (or 200 manifest with "wait": true)
    GET    /v1/blends | /v1/blends/<name>          blends / one manifest
    GET    /v1/blends/<name>/<file>    a WAV of the blend
    DELETE /v1/blends/<name>           remove a blend
    GET    /v1/folders                 every IR folder that can be blended (sets, blends, GUI renders)

A set is content-addressed: the same request (preset + overrides + set options + irgen
version) gives the same set id, is rendered once and served from renders/sets/<id>/.
Sets are written to a temporary folder and renamed into place, so a reader never sees a
half-written set. Requests may ask for a fast preview first (fewer modes, no T30
calibration): the job then reports a "preview" manifest before the "full" one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__

API_VERSION = 1
ROOT = Path(__file__).resolve().parent.parent
PRESET_DIRS = {"standard": ROOT / "presets", "strange": ROOT / "presets" / "strange"}
SETS = ROOT / "renders" / "sets"
MAX_CHANNELS = 64
MAX_JOBS_KEPT = 200

# set by the GUI: () -> (ir, fs) of its last standard render, for strange presets whose
# imported layer uses source "last_render"
IMPORT_PROVIDER = None


class RequestError(ValueError):
    pass


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name or "")).strip("_")


def _set_path(d, path, value):
    keys = path.split(".")
    for k in keys[:-1]:
        if not isinstance(d.get(k), dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# --------------------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------------------
def list_presets():
    out = []
    for engine, d in PRESET_DIRS.items():
        for p in sorted(d.glob("*.json")):
            try:
                j = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if engine == "standard":
                ch, fs = (j.get("multichannel") or {}).get("channels", 1), (j.get("output") or {}).get("sample_rate", 48000)
            else:
                ch, fs = (j.get("output") or {}).get("channels", 2), (j.get("output") or {}).get("sample_rate", 48000)
            out.append({"id": f"{engine}/{p.stem}", "engine": engine, "name": p.stem, "about": j.get("_about", ""),
                        "channels": ch, "sample_rate": fs})
    return out


def load_preset(pid):
    engine, _, name = str(pid).partition("/")
    if engine not in PRESET_DIRS or not _safe(name):
        raise RequestError(f"unknown preset id '{pid}' (use <engine>/<name>, engine = {'|'.join(PRESET_DIRS)})")
    p = PRESET_DIRS[engine] / f"{_safe(name)}.json"
    if not p.is_file():
        raise RequestError(f"preset '{pid}' not found")
    return engine, json.loads(p.read_text())


# --------------------------------------------------------------------------------------
# requests -> canonical form -> set id
# --------------------------------------------------------------------------------------
def resolve_request(req):
    """Validate a set request and return its canonical form (fully determines the set).

    req: {preset: "<engine>/<name>" | {...inline preset...}, engine (inline only),
          overrides: {"dotted.key": value}, channels: 1..64, sample_rate, seed,
          normalize: energy | peak | none (default energy: equal loudness across sets),
          layout: separate | interleaved | both (default separate: one mono WAV per channel),
          format: FLOAT | PCM_24 | PCM_16 | PCM_32 (default FLOAT), import_ir: path to a WAV}"""
    if not isinstance(req, dict):
        raise RequestError("request must be a JSON object")
    src = req.get("preset")
    if isinstance(src, str):
        engine, preset = load_preset(src)
        pid = src
    elif isinstance(src, dict):
        engine, preset, pid = req.get("engine", "standard"), copy.deepcopy(src), "inline"
        if engine not in PRESET_DIRS:
            raise RequestError(f"engine must be one of {list(PRESET_DIRS)}")
    else:
        raise RequestError("'preset' must be a preset id or an inline preset object")
    for k, v in (req.get("overrides") or {}).items():
        _set_path(preset, k, v)
    n = req.get("channels")
    if n is None:
        n = (preset.get("multichannel") or {}).get("channels", 1) if engine == "standard" else (preset.get("output") or {}).get("channels", 2)
    n = int(n)
    if not 1 <= n <= MAX_CHANNELS:
        raise RequestError(f"channels must be 1..{MAX_CHANNELS}")
    oc = preset.setdefault("output", {})
    if req.get("sample_rate") is not None:
        oc["sample_rate"] = int(req["sample_rate"])
    if req.get("seed") is not None:
        preset["seed"] = int(req["seed"])
    norm = req.get("normalize", "energy")
    if norm not in ("energy", "peak", "none"):
        raise RequestError("normalize must be energy | peak | none")
    oc["normalize"] = norm
    if engine == "standard":
        preset.setdefault("multichannel", {})["channels"] = n
    else:
        oc["channels"] = n
    layout = req.get("layout", "separate")
    if layout not in ("separate", "interleaved", "both"):
        raise RequestError("layout must be separate | interleaved | both")
    fmt = req.get("format", "FLOAT")
    if fmt not in ("FLOAT", "PCM_16", "PCM_24", "PCM_32"):
        raise RequestError("format must be FLOAT | PCM_16 | PCM_24 | PCM_32")
    imp = req.get("import_ir")
    if imp and not Path(imp).is_file():
        raise RequestError(f"import_ir '{imp}' not found")
    canon = {"engine": engine, "preset_id": pid, "name": _safe(preset.get("name") or pid.split("/")[-1]) or "set",
             "preset": preset, "channels": n, "layout": layout, "format": fmt, "normalize": norm,
             "import_ir": str(Path(imp).resolve()) if imp else None}
    if engine == "strange" and (preset.get("imported") or {}).get("enabled") and not imp:
        # imported layer fed by the host app (GUI's last standard render): capture it now and
        # fingerprint it, so a different source IR can never be served from the cache
        got = IMPORT_PROVIDER() if IMPORT_PROVIDER is not None and (preset["imported"].get("source", "last_render") == "last_render") else None
        if got is None:
            raise RequestError("preset uses the imported layer: pass 'import_ir' (path to a WAV)")
        x, ifs = got
        canon["import_fingerprint"] = hashlib.sha256(np.ascontiguousarray(x, np.float64).tobytes() + str(ifs).encode()).hexdigest()[:16]
        canon["_import_data"] = (np.array(x, copy=True), ifs)
    return canon


def set_id(canon, stage="full"):
    key = {k: v for k, v in canon.items() if k != "name" and not k.startswith("_")}
    if canon.get("import_ir"):
        st = os.stat(canon["import_ir"])
        key["import_ir"] = [canon["import_ir"], st.st_size, int(st.st_mtime)]
    h = hashlib.sha256(json.dumps({"api": API_VERSION, "irgen": __version__, "set": key},
                                  sort_keys=True, default=str).encode()).hexdigest()[:16]
    return h + ("-preview" if stage == "preview" else "")


def _preview_preset(engine, preset):
    """Fast variant with the same timing, alignment and length: fewer modes, no calibration."""
    p = copy.deepcopy(preset)
    if engine == "standard":
        _set_path(p, "late.calibrate", 0)
    else:
        mm = int((p.get("modes") or {}).get("max_modes", 600))
        _set_path(p, "modes.max_modes", min(mm, 150))
    return p


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------
def _imported(canon):
    imp = (canon["preset"].get("imported") or {})
    if canon["engine"] != "strange" or not imp.get("enabled"):
        return None
    if canon.get("import_ir"):
        import soundfile as sf
        x, ifs = sf.read(canon["import_ir"], always_2d=True)
        return x.T, ifs
    if canon.get("_import_data") is not None:
        return canon["_import_data"]
    raise RequestError("preset uses the imported layer: pass 'import_ir' (path to a WAV)")


def find_set(sid):
    m = SETS / sid / "manifest.json"
    return json.loads(m.read_text()) if m.is_file() else None


def render_set(canon, stage="full"):
    """Render (or fetch from cache) one set. Returns its manifest."""
    from .export import write_wav
    sid = set_id(canon, stage)
    hit = find_set(sid)
    if hit is not None:
        return _with_paths({**hit, "cached": True})
    preset = _preview_preset(canon["engine"], canon["preset"]) if stage == "preview" else canon["preset"]
    imported = _imported(canon)
    t0 = time.time()
    if canon["engine"] == "standard":
        from .engine import generate
        ir, fs, info, cfg = generate(preset)
    else:
        from .strange import generate_strange
        ir, fs, info, cfg, _ = generate_strange(preset, imported)
    dt = time.time() - t0
    ir = np.atleast_2d(ir)
    N, n = ir.shape
    lead = int(round(float(cfg["output"].get("lead_ms", 1.0)) * fs / 1000))
    direct = info.get("direct_index")
    SETS.mkdir(parents=True, exist_ok=True)
    tmp = SETS / f".{sid}.tmp-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        files, inter = [], None
        dith, seed, fmt = cfg["output"].get("dither", True), int(cfg.get("seed", 1)), canon["format"]
        if canon["layout"] in ("separate", "both"):
            for ch in range(N):
                fn = f"ch{ch + 1:02d}.wav"
                write_wav(str(tmp / fn), ir[ch], fs, fmt, dith, seed + ch)
                files.append({"channel": ch + 1, "file": fn})
        if canon["layout"] in ("interleaved", "both"):
            write_wav(str(tmp / "set.wav"), ir.T if N > 1 else ir[0], fs, fmt, dith, seed)
            inter = {"file": "set.wav", "channels": N}
        (tmp / "preset.json").write_text(json.dumps(_jsonable(cfg), indent=2))
        manifest = {
            "api": API_VERSION, "irgen_version": __version__, "set_id": sid, "stage": stage,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "engine": canon["engine"], "preset": canon["preset_id"], "name": canon["name"],
            "sample_rate": int(fs), "channels": N, "length_samples": int(n), "length_s": round(n / fs, 6),
            "format": fmt, "layout": canon["layout"], "normalize": canon["normalize"],
            "gain_db": info.get("normalize_gain_db"),
            "direct_index": int(direct) if direct is not None else lead, "direct_present": direct is not None,
            "files": files, "interleaved": inter,
            "t60_bands": info.get("t60_bands"), "render_seconds": round(dt, 3),
            "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool)) or v is None},
        }
        (tmp / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2))
        try:
            os.replace(tmp, SETS / sid)
        except OSError:   # rendered concurrently by someone else: keep theirs
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return _with_paths(find_set(sid))


def _with_paths(m):
    """Absolute paths and URLs for each file (computed, so a moved cache stays valid)."""
    if m is None:
        return None
    d = SETS / m["set_id"]
    for f in m.get("files", []):
        f["path"] = str(d / f["file"])
        f["url"] = f"/v1/sets/{m['set_id']}/{f['file']}"
    if m.get("interleaved"):
        m["interleaved"]["path"] = str(d / m["interleaved"]["file"])
        m["interleaved"]["url"] = f"/v1/sets/{m['set_id']}/{m['interleaved']['file']}"
    return m


# --------------------------------------------------------------------------------------
# job queue (one worker: renders already use every core)
# --------------------------------------------------------------------------------------
class Jobs:
    def __init__(self):
        self.jobs = OrderedDict()
        self.queue = []
        self.cv = threading.Condition()
        self.worker = None

    def _public(self, j):
        return {k: v for k, v in j.items() if not k.startswith("_")}

    def submit(self, canon, preview=False):
        sid = set_id(canon)
        with self.cv:
            for j in self.jobs.values():   # same set already queued / rendering: share the job
                if j["set_id"] == sid and j["status"] in ("queued", "rendering") and (preview or not j["preview"]):
                    return self._public(j)
            hit = find_set(sid)
            j = {"id": uuid.uuid4().hex[:12], "set_id": sid, "status": "done" if hit else "queued",
                 "stage": "full" if hit else None, "preview": bool(preview) and not hit,
                 "created": time.time(), "started": None, "finished": time.time() if hit else None,
                 "error": None, "manifest": _with_paths(hit), "preview_manifest": None, "cached": bool(hit), "_canon": canon}
            self.jobs[j["id"]] = j
            while len(self.jobs) > MAX_JOBS_KEPT:
                old = next(iter(self.jobs))
                if self.jobs[old]["status"] in ("queued", "rendering"):
                    break
                self.jobs.pop(old)
            if not hit:
                self.queue.append(j["id"])
                if self.worker is None or not self.worker.is_alive():
                    self.worker = threading.Thread(target=self._run, daemon=True, name="irgen-sets")
                    self.worker.start()
                self.cv.notify()
            return self._public(j)

    def submit_fn(self, label, fn):
        """Queue any job that returns a manifest (e.g. a blend)."""
        with self.cv:
            j = {"id": uuid.uuid4().hex[:12], "set_id": label, "status": "queued", "stage": None, "preview": False,
                 "created": time.time(), "started": None, "finished": None, "error": None, "manifest": None,
                 "preview_manifest": None, "cached": False, "_fn": fn}
            self.jobs[j["id"]] = j
            self.queue.append(j["id"])
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self._run, daemon=True, name="irgen-sets")
                self.worker.start()
            self.cv.notify()
            return self._public(j)

    def get(self, jid):
        with self.cv:
            j = self.jobs.get(jid)
            return None if j is None else self._public(j)

    def list(self):
        with self.cv:
            return [self._public(j) for j in reversed(self.jobs.values())]

    def cancel(self, jid):
        with self.cv:
            j = self.jobs.get(jid)
            if j is None:
                return False
            if j["status"] == "queued":
                self.queue.remove(jid)
                j.update(status="cancelled", finished=time.time())
            elif j["status"] == "rendering":
                j["_cancel"] = True   # the running render finishes; its result is still cached
            else:
                self.jobs.pop(jid)
            return True

    def _run(self):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                j = self.jobs[self.queue.pop(0)]
                j.update(status="rendering", started=time.time())
            try:
                if j.get("_fn"):
                    j["stage"] = "full"
                    j["manifest"] = j["_fn"]()
                elif j["preview"]:
                    j["stage"] = "preview"
                    j["preview_manifest"] = render_set(j["_canon"], "preview")
                if not j.get("_fn") and not j.get("_cancel"):
                    j["stage"] = "full"
                    j["manifest"] = render_set(j["_canon"], "full")
                with self.cv:
                    j.update(status="cancelled" if j.get("_cancel") else "done", finished=time.time())
            except Exception as e:
                traceback.print_exc()
                with self.cv:
                    j.update(status="error", error=f"{type(e).__name__}: {e}", finished=time.time())


JOBS = Jobs()


# --------------------------------------------------------------------------------------
# HTTP routing (shared by `irgen serve` and the GUI server)
# --------------------------------------------------------------------------------------
def handle(method, path, body=None):
    """-> (status, payload, content_type). payload: dict/list (JSON) or bytes."""
    parts = [p for p in path.split("/") if p][1:]   # drop "v1"
    try:
        if method == "GET" and not parts:
            return 200, {"api": API_VERSION, "irgen_version": __version__, "engines": list(PRESET_DIRS),
                         "max_channels": MAX_CHANNELS, "endpoints": [l.strip() for l in __doc__.split("\n") if l.strip().startswith(("GET", "POST", "DELETE"))]}, None
        if parts[:1] == ["presets"]:
            if method == "GET" and len(parts) == 1:
                return 200, list_presets(), None
            if method == "GET" and len(parts) == 3:
                return 200, load_preset(f"{parts[1]}/{parts[2]}")[1], None
        if parts[:1] == ["sets"]:
            if method == "POST" and len(parts) == 1:
                body = body or {}
                canon = resolve_request(body)
                if body.get("wait"):
                    return 200, render_set(canon), None
                j = JOBS.submit(canon, preview=bool(body.get("preview")))
                return (200 if j["status"] == "done" else 202), j, None
            if method == "GET" and len(parts) == 1:
                out = []
                if SETS.is_dir():
                    for m in sorted(SETS.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                        j = json.loads(m.read_text())
                        out.append({k: j.get(k) for k in ("set_id", "stage", "created", "engine", "preset", "name", "channels", "sample_rate", "length_s")})
                return 200, out, None
            sid = _safe(parts[1]) if len(parts) > 1 else ""
            if len(parts) == 2 and method == "GET":
                m = _with_paths(find_set(sid))
                return (200, m, None) if m else (404, {"error": "set not found"}, None)
            if len(parts) == 2 and method == "DELETE":
                d = SETS / sid
                if not sid or not d.is_dir():
                    return 404, {"error": "set not found"}, None
                shutil.rmtree(d)
                return 200, {"ok": True}, None
            if len(parts) == 3 and method == "GET":
                p = (SETS / sid / parts[2]).resolve()
                if not str(p).startswith(str((SETS / sid).resolve()) + os.sep) or not p.is_file():
                    return 404, {"error": "file not found"}, None
                ctype = "audio/wav" if p.suffix == ".wav" else "application/json"
                return 200, p.read_bytes(), ctype
        if parts[:1] == ["folders"] and method == "GET":
            from .blend import list_folders
            return 200, list_folders(), None
        if parts[:1] == ["blends"]:
            return _blends(method, parts[1:], body)
        if parts[:1] == ["jobs"]:
            if method == "GET" and len(parts) == 1:
                return 200, JOBS.list(), None
            if method == "GET" and len(parts) == 2:
                j = JOBS.get(parts[1])
                return (200, j, None) if j else (404, {"error": "job not found"}, None)
            if method == "DELETE" and len(parts) == 2:
                return (200, {"ok": True}, None) if JOBS.cancel(parts[1]) else (404, {"error": "job not found"}, None)
        return 404, {"error": f"no route {method} {path}"}, None
    except RequestError as e:
        return 400, {"error": str(e)}, None
    except ValueError as e:   # BlendError and other request problems
        return 400, {"error": str(e)}, None
    except Exception as e:
        traceback.print_exc()
        return 500, {"error": f"{type(e).__name__}: {e}"}, None


def _blends(method, parts, body):
    from . import blend as bl
    if method == "POST" and not parts:
        b = body or {}
        args = dict(name=b.get("name"), a=b.get("a"), b=b.get("b"), mode=b.get("mode", "morph"),
                    amount=b.get("amount", 0.5), normalize=b.get("normalize", "energy"),
                    layout=b.get("layout", "separate"), fmt=b.get("format", "FLOAT"), overwrite=bool(b.get("overwrite")))
        # validate now (names, folders, options), so mistakes come back as 400 instead of a failed job
        bl.resolve_folder(args["a"]), bl.resolve_folder(args["b"])
        if not bl._safe(args["name"]):
            raise RequestError("give the blend a name")
        if (bl.BLENDS / bl._safe(args["name"])).exists() and not args["overwrite"]:
            raise RequestError(f"blend '{bl._safe(args['name'])}' exists (pass \"overwrite\": true)")
        if args["mode"] not in bl.MODES:
            raise RequestError(f"mode must be one of {bl.MODES}")
        if b.get("wait"):
            return 200, bl.blend_folders(**args)[0], None
        return 202, JOBS.submit_fn(f"blend/{bl._safe(args['name'])}", lambda: bl.blend_folders(**args)[0]), None
    if method == "GET" and not parts:
        out = []
        if bl.BLENDS.is_dir():
            for m in sorted(bl.BLENDS.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                j = json.loads(m.read_text())
                out.append({k: j.get(k) for k in ("name", "created", "channels", "sample_rate", "length_s", "blend")})
        return 200, out, None
    name = bl._safe(parts[0]) if parts else ""
    if len(parts) == 1 and method == "GET":
        m = bl.find_blend(name)
        return (200, m, None) if m else (404, {"error": "blend not found"}, None)
    if len(parts) == 1 and method == "DELETE":
        d = bl.BLENDS / name
        if not name or not d.is_dir():
            return 404, {"error": "blend not found"}, None
        shutil.rmtree(d)
        return 200, {"ok": True}, None
    if len(parts) == 2 and method == "GET":
        p = (bl.BLENDS / name / parts[1]).resolve()
        if not str(p).startswith(str((bl.BLENDS / name).resolve()) + os.sep) or not p.is_file():
            return 404, {"error": "file not found"}, None
        return 200, p.read_bytes(), ("audio/wav" if p.suffix == ".wav" else "application/json")
    return 404, {"error": f"no route {method} /v1/blends/{'/'.join(parts)}"}, None


def send(handler, code, payload, ctype=None):
    body = payload if isinstance(payload, bytes) else json.dumps(_jsonable(payload)).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", ctype or "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def serve(port=8766):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse

    class H(BaseHTTPRequestHandler):
        server_version = "irgen-sets"

        def log_message(self, fmt, *args):
            pass

        def _go(self, method):
            u = urlparse(self.path)
            if not (u.path == "/v1" or u.path.startswith("/v1/")):
                return send(self, 404, {"error": "irgen IR-set API lives under /v1"})
            body = None
            if method == "POST":
                n = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError as e:
                    return send(self, 400, {"error": f"bad JSON: {e}"})
            send(self, *handle(method, u.path, body))

        def do_GET(self):
            self._go("GET")

        def do_POST(self):
            self._go("POST")

        def do_DELETE(self):
            self._go("DELETE")

    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"irgen IR-set API v{API_VERSION} at http://127.0.0.1:{port}/v1  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
