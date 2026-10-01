"""Non-blocking text-to-speech with cooldown and de-duplication."""
from __future__ import annotations

import queue
import threading
import time


class Speaker:
    def __init__(self, enabled: bool, cooldown: float, repeat_after: float):
        self.enabled = enabled
        self.cooldown = cooldown
        self.repeat_after = repeat_after
        self._last_text = ""
        self._last_time = float("-inf")
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
        except Exception as e:
            print(f"Speech disabled: {e}")
            self.enabled = False
            return
        while (text := self._q.get()) is not None:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")

    def should_speak(self, text: str, now: float) -> bool:
        if now - self._last_time < self.cooldown:
            return False
        return text != self._last_text or now - self._last_time >= self.repeat_after

    def say(self, text: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if not self.should_speak(text, now):
            return False
        if self.enabled:
            try:
                self._q.put_nowait(text)
            except queue.Full:
                return False
        self._last_text, self._last_time = text, now
        return True

    def close(self) -> None:
        if self._thread is None:
            return
        try:
            self._q.put(None, timeout=1)
        except queue.Full:
            pass  # worker is stuck mid-speech; it's a daemon thread, so exit anyway
        self._thread.join(timeout=2)
