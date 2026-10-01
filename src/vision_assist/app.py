from __future__ import annotations

import argparse
import time

import cv2

from .camera import Camera
from .config import Config
from .detector import Detector
from .scene import Stabilizer, describe
from .speaker import Speaker


def add_model_args(p: argparse.ArgumentParser) -> None:
    d = Config()
    p.add_argument("--model", default=d.model_path, help="YOLO weights (e.g. models/yolov8n.pt for speed)")
    p.add_argument("--conf", type=float, default=d.confidence, help="min detection confidence")
    p.add_argument("--imgsz", type=int, default=d.image_size, help="inference size; larger = more accurate, slower")
    p.add_argument("--device", default=d.device, help="cpu, cuda, or empty for auto")


def config_from(a: argparse.Namespace, **extra) -> Config:
    return Config(model_path=a.model, confidence=a.conf, image_size=a.imgsz, device=a.device, **extra)


def draw_caption(frame, text: str) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def main() -> None:
    p = argparse.ArgumentParser(description="Real-time webcam scene description")
    p.add_argument("--camera", type=int, default=Config.camera_index, help="camera index")
    p.add_argument("--no-speech", action="store_true", help="disable voice output")
    add_model_args(p)
    a = p.parse_args()
    cfg = config_from(a, camera_index=a.camera, speak=not a.no_speech)

    print("Loading model...")
    detector = Detector(cfg)
    print(f"Running on {detector.device}. Press q to quit.")
    stabilizer = Stabilizer(cfg)
    speaker = Speaker(cfg.speak, cfg.speak_cooldown, cfg.repeat_after)
    try:
        camera = Camera(cfg.camera_index)
    except RuntimeError as e:
        raise SystemExit(str(e))

    last_print, fps_t, fps = "", time.monotonic(), 0.0
    try:
        while True:
            frame = camera.read()
            if frame is None:
                break
            h, w = frame.shape[:2]
            dets, annotated = detector.detect(frame, annotate=True)
            text = describe(stabilizer.update(dets), w, h, cfg)

            if text != last_print:
                print(text)
                last_print = text
            if text != "Nothing detected":
                speaker.say(text)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-6)  # smoothed
            fps_t = now
            cv2.putText(annotated, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            draw_caption(annotated, text)
            cv2.imshow(cfg.window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        speaker.close()


if __name__ == "__main__":
    main()
