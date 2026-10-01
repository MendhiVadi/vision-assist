"""Turns raw detections into a stable, human-readable scene description."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def horizontal_position(box, frame_width: int) -> str:
    cx = (box[0] + box[2]) / 2 / frame_width
    if cx < 1 / 3:
        return "on the left"
    if cx > 2 / 3:
        return "on the right"
    return "in front"


def proximity(box, frame_w: int, frame_h: int, cfg: Config) -> str:
    ratio = ((box[2] - box[0]) * (box[3] - box[1])) / (frame_w * frame_h)
    if ratio >= cfg.close_ratio:
        return "close"
    if ratio >= cfg.near_ratio:
        return "near"
    return "far"


_NUMBERS = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five"}

_IRREGULAR_PLURALS = {
    "person": "people",
    "knife": "knives",
    "mouse": "mice",
    "bus": "buses",
    "tv": "TVs",
    "sheep": "sheep",
    "fish": "fish",
}
# Labels that are already plural; singular reads as "a pair of ...".
_PAIR_LABELS = {"scissors", "skis", "pants", "glasses"}


def _pluralize_word(word: str) -> str:
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]
    if word in _PAIR_LABELS:
        return word
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _plural(label: str, n: int) -> str:
    """Pluralize the last word of a label ("sports ball" -> "sports balls")."""
    if n == 1:
        return f"a pair of {label}" if label in _PAIR_LABELS else label
    head, sep, last = label.rpartition(" ")
    return head + sep + _pluralize_word(last)


def _phrase(label: str, n: int, pos: str, prox: str) -> str:
    prefix = "very close: " if prox == "close" else ""
    if n == 1 and label in _PAIR_LABELS:
        noun = _plural(label, 1)
        return f"{prefix}{noun} {pos}"
    count = _NUMBERS.get(n, "several")
    if count == "a" and label[0] in "aeiou":
        count = "an"
    return f"{prefix}{count} {_plural(label, n)} {pos}"


def describe(detections: list[Detection], frame_w: int, frame_h: int, cfg: Config) -> str:
    """Group detections by (label, position) and render a sentence."""
    if not detections:
        return "Nothing detected"
    groups: dict[tuple[str, str], list[str]] = {}
    for d in detections:
        pos = horizontal_position(d.box, frame_w)
        groups.setdefault((d.label, pos), []).append(proximity(d.box, frame_w, frame_h, cfg))
    parts = []
    for (label, pos), proxes in groups.items():
        closest = "close" if "close" in proxes else ("near" if "near" in proxes else "far")
        parts.append((closest != "close", -len(proxes), _phrase(label, len(proxes), pos, closest)))
    parts.sort()  # close objects first, then larger groups
    return ", ".join(p[2] for p in parts[:4])


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Stabilizer:
    """Suppresses flicker with per-label counts.

    A label is reported with count n only if at least n of that label were seen
    in at least `stability_min` of the last `stability_window` frames. So two
    people stay two when one briefly drops out, and one-frame blips vanish.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history: deque[list[Detection]] = deque(maxlen=cfg.stability_window)

    def _stable_count(self, label: str) -> int:
        counts = sorted(
            (sum(1 for d in frame if d.label == label) for frame in self.history),
            reverse=True,
        )
        m = self.cfg.stability_min
        # m-th largest per-frame count = largest n present in >= m frames
        return counts[m - 1] if 0 < m <= len(counts) else 0

    def update(self, detections: list[Detection]) -> list[Detection]:
        self.history.append(list(detections))
        out: list[Detection] = []
        for label in dict.fromkeys(d.label for f in self.history for d in f):
            k = self._stable_count(label)
            if k <= 0:
                continue
            cur = sorted((d for d in detections if d.label == label),
                         key=lambda d: -d.confidence)[:k]
            # Fill in flickering instances from the most recent frames.
            for frame in reversed(self.history):
                if len(cur) >= k:
                    break
                for d in frame:
                    if len(cur) >= k:
                        break
                    if d.label == label and all(_iou(d.box, c.box) < 0.3 for c in cur):
                        cur.append(d)
            out.extend(cur)
        return out
