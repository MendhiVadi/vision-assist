import cv2
import numpy as np
import pytest

from vision_assist.api import Analyzer
from vision_assist.config import Config
from vision_assist.scene import Detection


class FakeDetector:
    def detect(self, frame, annotate=False, imgsz=None):
        h, w = frame.shape[:2]
        return [Detection("cup", 0.9, (w * 0.4, h * 0.4, w * 0.6, h * 0.6))], frame


def make_analyzer():
    a = Analyzer.__new__(Analyzer)
    a.cfg = Config()
    a.detector = FakeDetector()
    a.lens_model = None
    return a


def test_analyze_jpeg():
    ok, buf = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))
    assert ok
    result = make_analyzer().analyze(buf.tobytes())
    assert result["output"] == "a cup in front"
    assert result["detections"][0]["label"] == "cup"


def test_analyze_bad_image():
    with pytest.raises(ValueError):
        make_analyzer().analyze(b"not an image")
