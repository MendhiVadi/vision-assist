# Vision Assist

Webcam scene description: YOLO detects objects, the app says what it sees.

## Architecture
- Python package `src/vision_assist/`: `detector`, `camera` (latest-frame thread), `scene` (position/proximity wording), `speaker`, `app` (desktop loop), `api`, `config`
- `python -m vision_assist.api` (or `start.bat`) serves the website AND the API in one process at http://127.0.0.1:8001 (`POST /api/process`, `GET /health`)
- Node `server.js` (:3000) is now an optional proxy to the Python API
- Page has Capture, upload, live mode with box overlay, browser speech; frames downscaled to 640px in the browser
- Default model `models/yolo11m.pt`. CPU benchmark at 640: yolov8n ~73 ms, yolov8s ~114 ms, yolo11s ~126 ms; ~140-160 ms end to end via the API
- CLI flags: `--model --conf --imgsz --device --camera --no-speech`

## Status (2026-10-01)
- [x] Desktop app, scene wording, speech cooldown, flicker filter
- [x] Analysis API serving the website; Node proxy optional
- [x] Website live mode and box overlay
- [x] Dependencies installed
- [x] Tests pass (11/11)
- [x] API verified on a real photo
- [ ] Try desktop app on a real webcam
- [ ] Try website on a real webcam (`start.bat`)
- [ ] Tune thresholds after real use
- [ ] git init
