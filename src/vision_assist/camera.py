"""Webcam reader that always hands back the newest frame, never a stale buffered one."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

MAX_FAILED_READS = 30  # tolerate transient USB/startup hiccups before giving up


class Camera:
    def __init__(self, index: int):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame: np.ndarray | None = None
        self._seq = 0  # increments per captured frame
        self._last_returned = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures >= MAX_FAILED_READS:
                    break
                time.sleep(0.03)
                continue
            failures = 0
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def read(self, timeout: float = 2.0) -> np.ndarray | None:
        """Next unseen frame, or None if the camera stopped or timed out."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > self._last_returned or self._stop.is_set(), timeout)
            if self._seq == self._last_returned:
                return None
            self._last_returned = self._seq
            return self._frame

    def release(self) -> None:
        self._stop.set()
        self._thread.join()  # reader exits after at most one blocking read
        self.cap.release()
