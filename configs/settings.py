"""
NetraVisionAi — Central Configuration
All tuneable parameters in one place
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CameraConfig:
    source: int = 0                     # 0 = default webcam
    width: int = 640
    height: int = 480
    fps: int = 30
    ip_camera_url: Optional[str] = None  # e.g., "http://192.168.1.5:8080/video"


@dataclass
class DetectorConfig:
    model_path: str = "models/yolov8n.onnx"
    input_size: int = 320               # 320 = fastest, 640 = most accurate
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.45


@dataclass  
class DepthConfig:
    model_path: str = "models/midas_v21_small_256.onnx"
    input_size: int = 256


@dataclass
class TTSConfig:
    engine: str = "espeak"              # "espeak" or "piper"
    language: str = "en"
    speed: int = 175                    # Words per minute
    voice: str = "en"                   # espeak voice


@dataclass
class AlertConfig:
    critical_depth: float = 0.70        # Very close — DANGER
    high_depth: float = 0.50            # Close — attention needed
    medium_depth: float = 0.30          # Nearby — informational
    
    # Cooldowns (seconds) — prevent repeating same alert
    critical_cooldown: float = 1.5
    high_cooldown: float = 3.0
    medium_cooldown: float = 8.0
    low_cooldown: float = 15.0
    
    max_alerts_per_cycle: int = 3       # Don't flood with alerts


@dataclass
class PipelineConfig:
    fast_lane_skip: int = 3             # Process every Nth frame
    medium_lane_skip: int = 15
    slow_lane_interval: float = 5.0     # Seconds between VLM descriptions
    enable_ocr: bool = False            # Disable OCR by default (heavy)
    enable_vlm: bool = False            # Disable VLM by default (heavy)


@dataclass
class NetraVisionAiConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Paths
    models_dir: str = "models"
    audio_cache_dir: str = "audio_cache"
    log_dir: str = "logs"


# Global default config
DEFAULT_CONFIG = NetraVisionAiConfig()

print("[Config] NetraVisionAi configuration loaded ✓") if __name__ == "__main__" else None