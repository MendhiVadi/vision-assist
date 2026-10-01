from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    model_path: str = "models/yolo11m.pt"
    device: str = ""  # "" = auto (CUDA if available, else CPU)
    confidence: float = 0.45
    iou: float = 0.5
    max_det: int = 30
    image_size: int = 640
    # Box area / frame area thresholds for "close" and "near".
    close_ratio: float = 0.25
    near_ratio: float = 0.08
    # Minimum seconds between spoken updates, and before repeating the same text.
    speak_cooldown: float = 3.0
    repeat_after: float = 15.0
    # A label must appear in at least `stability_min` of the last `stability_window` frames.
    stability_window: int = 5
    stability_min: int = 3
    speak: bool = True
    window_name: str = "Vision Assist"
