"""One process for the website and the analysis API.

GET  /               the web page (public/index.html)
POST /api/process    raw image bytes -> {"output": sentence, "detections": [...]}
GET  /health
(POST /analyze is an alias, used by the optional Node proxy in server.js.)
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from .config import Config
from .detector import Detector
from .scene import describe

HQ_SIZE = 640  # still photos (?hq=1) use the lens model at its native size (960 gave noisier labels)
PAGES = {"/": "index.html", "/index.html": "index.html", "/lens": "lens.html"}
PUBLIC = Path(__file__).resolve().parents[2] / "public"
LENS_MODEL = "models/yoloe-v8l-seg-pf.pt"  # ~4,500 object classes
MAX_UPLOAD = 20 * 1024 * 1024
STATIC_TYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
                ".json": "application/json", ".onnx": "application/octet-stream"}


# The lens model has ~4,500 classes but names some things differently from everyday speech.
LABEL_ALIASES = {
    "cabinet": "cupboard", "kitchen cabinet": "cupboard", "cabinetry": "cupboard",
    "side cabinet": "cupboard", "bathroom cabinet": "cupboard", "closet": "cupboard",
    "beer can": "can", "spray can": "can", "milk can": "can",
    "earphone": "earphones", "stop watch": "watch", "pocket watch": "watch",
    "power plugs and sockets": "power socket",
}


def normalize(dets):
    return [replace(d, label=LABEL_ALIASES.get(d.label, d.label)) for d in dets]


def summarize(dets) -> list[dict]:
    """Unique objects with counts, most confident first (the "what is in this image" list)."""
    groups: dict[str, list[float]] = {}
    for d in dets:
        groups.setdefault(d.label, []).append(d.confidence)
    objs = [{"label": l, "count": len(c), "confidence": round(max(c), 3)} for l, c in groups.items()]
    return sorted(objs, key=lambda o: (-o["count"] * o["confidence"], o["label"]))


class Analyzer:
    """Live frames use the fast detector; still photos (hq) use the large-vocabulary lens model."""

    def __init__(self, cfg: Config, lens_model: str | None = LENS_MODEL):
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.lens_model = lens_model
        self._lens: Detector | None = None
        self._lens_lock = threading.Lock()

    def lens_detector(self) -> Detector:
        if self.lens_model is None:
            return self.detector
        with self._lens_lock:  # loaded on first use so startup stays fast
            if self._lens is None:
                try:
                    self._lens = Detector(replace(self.cfg, model_path=self.lens_model, confidence=0.3))
                except Exception as e:
                    print(f"Lens model unavailable ({e}); using {self.cfg.model_path}")
                    self.lens_model = None
                    return self.detector
            return self._lens

    def analyze(self, data: bytes, imgsz: int | None = None) -> dict:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
        h, w = frame.shape[:2]
        if imgsz:
            dets, _ = self.lens_detector().detect(frame, imgsz=imgsz)
            dets = normalize(dets)
        else:
            dets, _ = self.detector.detect(frame)
        return {
            "output": describe(dets, w, h, self.cfg),
            "objects": summarize(dets),
            "width": w,
            "height": h,
            "detections": [
                {"label": d.label, "confidence": round(d.confidence, 3), "box": [round(v) for v in d.box]}
                for d in dets
            ],
        }


def make_handler(analyzer: Analyzer, public: Path = PUBLIC):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")  # lets onnxruntime-web use threads
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client went away

        def _json(self, status: int, body: dict) -> None:
            self._send(status, json.dumps(body).encode(), "application/json")

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            name = PAGES.get(urlsplit(self.path).path)
            if name:
                page = public / name
                if page.is_file():
                    return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            rel = urlsplit(self.path).path.lstrip("/")
            f = (public / rel).resolve()
            if rel and f.is_file() and public.resolve() in f.parents:  # js/, ort/, models/ for in-browser detection
                ctype = STATIC_TYPES.get(f.suffix, "application/octet-stream")
                return self._send(200, f.read_bytes(), ctype)
            self._json(404, {"error": "not found"})

        def do_POST(self):
            url = urlsplit(self.path)
            if url.path not in ("/api/process", "/analyze"):
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_UPLOAD:
                self.close_connection = True
                return self._json(400, {"output": "Error: empty or oversized image", "error": "bad size"})
            data = self.rfile.read(length)
            if len(data) != length:
                return  # client aborted mid-upload
            try:
                hq = parse_qs(url.query).get("hq") == ["1"]
                self._json(200, analyzer.analyze(data, HQ_SIZE) if hq else analyzer.analyze(data))
            except ValueError as e:
                self._json(400, {"output": f"Error: {e}", "error": str(e)})
            except Exception as e:  # keep the service alive on model errors
                self._json(500, {"output": f"Error: {e}", "error": str(e)})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    from .app import add_model_args, config_from

    p = argparse.ArgumentParser(description="Vision Assist website + analysis API")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--lens-model", default=LENS_MODEL, help="large-vocabulary model for /lens; 'none' to reuse --model")
    p.add_argument("--page", default="/lens", help="page to open in the browser (/lens or /)")
    p.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    add_model_args(p)
    a = p.parse_args()
    analyzer = Analyzer(config_from(a), None if a.lens_model == "none" else a.lens_model)
    server = ThreadingHTTPServer((a.host, a.port), make_handler(analyzer))
    url = f"http://{a.host}:{a.port}"
    print(f"Vision Assist on {url}  (device: {analyzer.detector.device})")
    if not a.no_browser:
        webbrowser.open(url + a.page)  # model is loaded by now, so the page works immediately
    server.serve_forever()


if __name__ == "__main__":
    main()
