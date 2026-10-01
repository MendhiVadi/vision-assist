from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .config import Config
from .scene import Detection


class Detector:
    def __init__(self, cfg: Config):
        import torch
        from ultralytics import YOLO

        self.cfg = cfg
        path = Path(cfg.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ultralytics downloads the weights to this path if it doesn't exist yet.
        self.model = YOLO(str(path))
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.startswith("cuda")  # FP16 on CUDA is faster, equally accurate
        self._lock = threading.Lock()  # one inference at a time (webcam loop or API threads)
        self.detect(np.zeros((cfg.image_size, cfg.image_size, 3), np.uint8))  # warm-up

    def detect(self, frame: np.ndarray, annotate: bool = False, imgsz: int | None = None) -> tuple[list[Detection], np.ndarray | None]:
        cfg = self.cfg
        with self._lock:
            result = self.model.predict(
                frame,
                conf=cfg.confidence,
                iou=cfg.iou,
                imgsz=imgsz or cfg.image_size,
                max_det=cfg.max_det,
                device=self.device,
                half=self.half,
                agnostic_nms=True,  # one box per object even if two classes overlap
                verbose=False,
            )[0]
        boxes = result.boxes
        dets = [
            Detection(self.model.names[int(c)], float(p), tuple(xy))
            for c, p, xy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())
        ]
        return dets, (result.plot() if annotate else None)
