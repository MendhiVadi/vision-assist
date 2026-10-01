"""Export the browser models into public/models (run from the repo root).

live.onnx       yolo11m          live camera (WebGPU)
live-lite.onnx  yolo11s          live camera fallback when WebGPU is unavailable
lens.onnx       yoloe-11s-seg-pf photo / lens mode (~4,500 classes)
"""
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

OUT = Path("public/models")
OUT.mkdir(parents=True, exist_ok=True)
JOBS = [("yolo11m", "live", True), ("yolo11s", "live-lite", False), ("yoloe-11s-seg-pf", "lens", True)]

for weights, name, labels in JOBS:
    model = YOLO(f"models/{weights}.pt")
    shutil.copy(model.export(format="onnx", imgsz=640, simplify=True, opset=17), OUT / f"{name}.onnx")
    if labels:
        (OUT / f"{name}.json").write_text(json.dumps([model.names[i] for i in range(len(model.names))]))
