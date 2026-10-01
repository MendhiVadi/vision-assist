# Vision Assist

Real-time webcam scene description. Reads your camera, detects objects with YOLO, and tells you what it sees, on screen and out loud:

> "very close: a person ahead, two chairs on the left"

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

The default model is `models/yolo11m.pt`; Ultralytics downloads it on first run.

## Website + API (one process)

```bash
python -m vision_assist.api     # or double-click start.bat
```

Open http://127.0.0.1:8001. The same process serves the page and the API (`--port`, `--host` also available). Use "Start live" for continuous description with browser speech, or "Capture & analyze" for one frame. The browser downscales frames to 640px before upload.

Node is optional: `node server.js` (http://localhost:3000) is just a proxy to the Python API. `VISION_API` and `PORT` env vars override its defaults.

## Desktop app

```bash
python -m vision_assist                 # or: vision-assist
python -m vision_assist --no-speech     # text only
python -m vision_assist --camera 1 --conf 0.5 --model models/yolov8n.pt
```

Press `q` in the video window to quit.

### CLI flags (desktop app and API)

| Flag | Meaning |
| --- | --- |
| `--model` | YOLO weights (default `models/yolo11m.pt`) |
| `--conf` | Minimum detection confidence (default 0.45) |
| `--imgsz` | Inference size (default 640); larger is more accurate, slower |
| `--device` | `cpu`, `cuda`, or empty for auto |
| `--camera` | Camera index (desktop app only) |
| `--no-speech` | Disable voice output (desktop app only) |

## Model speed

CPU benchmark at 640: yolov8n ~73 ms, yolov8s ~114 ms, yolo11s ~126 ms per frame. Through the API, expect ~140-160 ms end to end. Use `--model models/yolov8n.pt` for speed.

## API

- `GET /` the web page
- `POST /api/process` raw image bytes, returns `{"output": sentence, "detections": [...]}` (`/analyze` is an alias used by the Node proxy)
- `GET /health`

## How it works

| Module | Role |
| --- | --- |
| `detector.py` | YOLO inference (thread-safe), returns detections and an annotated frame |
| `camera.py` | Capture thread that always keeps the latest frame, so no stale buffered frames |
| `scene.py` | Position (left/ahead/right), proximity (box size), grouping, plurals, flicker filter |
| `speaker.py` | Background TTS with cooldown and repeat suppression |
| `api.py` | Serves the website and the analysis API |
| `app.py` | Webcam loop, overlay caption, FPS, CLI |
| `config.py` | Thresholds and timings |

## Tests

```bash
pytest
```

11 tests pass.
