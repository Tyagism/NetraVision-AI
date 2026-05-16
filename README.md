# 🦯 NetraVision AI
## An Intelligent Assistive Vision System for the Visually Impaired

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](#)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [Supported Languages](#supported-languages)
- [Performance & Hardware Requirements](#performance--hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**NetraVision AI** is an intelligent assistive technology system designed to help visually impaired individuals navigate and understand their environment in real-time. The system combines **computer vision**, **depth estimation**, and **natural language processing** to provide rich audio descriptions of the surroundings.

The name "Netra" comes from Sanskrit meaning "eye," symbolizing how this system acts as an artificial vision aid for those with visual impairments.

### Core Capabilities:
- **Real-time Object Detection**: Identifies objects, people, and hazards using YOLOv8-nano
- **Depth Perception**: Estimates spatial distance using MiDaS depth estimation
- **Spatial Awareness**: Maps detected objects to spatial sectors (left, center, right, etc.)
- **Priority Alerts**: Intelligently prioritizes critical information based on proximity and relevance
- **Multilingual Audio Output**: Converts visual information to speech in 15+ languages
- **Low-Latency Processing**: Optimized for real-time performance on edge devices

---

## Key Features

✨ **Core Features:**
- 🎥 Multi-source camera support (USB webcams, IP cameras, video files)
- 🤖 Real-time object detection (640+ object classes)
- 📏 Monocular depth estimation with multiple model variants
- 🗣️ Realistic text-to-speech with 15+ languages
- 🎯 Intelligent alert prioritization system
- ⚡ Frame-skipping optimization for performance
- 🖥️ Optional display window for debugging
- 🔊 Audio caching for repetitive alerts

💡 **Advanced Features:**
- Multi-threaded processing pipeline (camera capture, detection, TTS)
- Sector-based spatial mapping (front, left, right, danger zones)
- Customizable alert thresholds and cooldowns
- Statistics tracking (frames processed, objects detected, alerts generated)
- Support for both CPU and GPU acceleration

---

## System Architecture

### High-Level Pipeline

```
Camera Input
    ↓
[Camera Module]
    ↓
[Object Detection + Depth Estimation] (parallel)
    ↓
[Spatial Mapping]
    ↓
[Priority Engine]
    ↓
[Alert Manager + TTS Engine]
    ↓
Audio Output (Speaker) + Optional Video Display
```

### Multi-Threaded Architecture

```
Thread 1: Camera Capture
  └─ Continuous frame capture from camera
  └─ Resolution: 640×480 @ 30 FPS
  
Thread 2: Detection & Processing (FAST Lane)
  └─ YOLO Detection (every Nth frame)
  └─ Depth Estimation
  └─ Spatial Mapping
  └─ Priority Calculation
  
Thread 3: TTS & Audio Output
  └─ Text-to-Speech synthesis
  └─ Speaker playback
  └─ Alert queue management
```

### Component Overview

| Component | Purpose | Model |
|-----------|---------|-------|
| **Camera** | Frame capture | OpenCV |
| **Detector** | Object detection | YOLOv8-nano (ONNX) |
| **Depth Estimator** | Distance estimation | MiDaS v2.1 |
| **Spatial Mapper** | Sector mapping | Custom algorithm |
| **Priority Engine** | Alert ranking | Rule-based |
| **TTS Engine** | Speech synthesis | Piper TTS / espeak-ng |

---

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Linux
- **Python**: 3.11 or higher
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Storage**: 2-3 GB for models and dependencies
- **Camera**: USB webcam or IP camera

### Hardware Recommendations

| Configuration | Device Type | Performance |
|---------------|-------------|-------------|
| **CPU-Only** | Intel i5/Ryzen 5+ | 15-20 FPS |
| **GPU (NVIDIA)** | RTX 3060+ | 25-30 FPS |
| **GPU (AMD)** | RX 6700+ | 25-30 FPS |
| **Edge Device** | Raspberry Pi 4 | 5-10 FPS |

### Optional
- **NVIDIA CUDA**: For GPU acceleration (CUDA 11.8+, cuDNN 8.6+)
- **Speaker/Headphones**: For audio output
- **Microphone**: For voice control (future feature)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Tyagism/NetraVision-AI.git
cd NetraVision-AI
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models

```bash
python scripts/download_models.py
```

This will automatically download:
- YOLOv8-nano (320×320 and 640×640 variants)
- MiDaS v2.1 Small & Hybrid models
- Piper TTS voice files
- Indic language models

Alternatively, models can be downloaded manually from:
- [YOLOv8 Models](https://github.com/ultralytics/yolov8)
- [MiDaS Models](https://github.com/isl-org/MiDaS)
- [Piper TTS Voices](https://huggingface.co/rhasspy/piper-voices)

### Step 5: Verify Installation

```bash
python scripts/test_camera.py
python scripts/test_detector.py
python scripts/test_tts.py
```

---

## Quick Start

### Basic Usages

```bash
# Default settings (small model, English, webcam)
python main.py

# Spanish language with custom speed
python main.py --language es --speed 150

# Use hybrid depth model (more accurate, slower)
python main.py --model hybrid

# Use GPU acceleration
python main.py --device cuda

# Process every 5th frame (faster, less frequent updates)
python main.py --skip-frames 5

# Disable video display (audio-only mode)
python main.py --no-video
```

### Command-Line Arguments

```
--camera              Camera index (0=default webcam) [default: 0]
--ip-camera          IP camera URL (e.g., http://192.168.1.5:8080/video)
--model              Depth model: small or hybrid [default: small]
--device             Processing device: auto, cpu, or cuda [default: auto]
--language           Language code (en, hi, es, etc.) [default: en]
--speed              TTS speed in words per minute [default: 185]
--no-video           Run without display window
--skip-frames        Process every Nth frame [default: 3]
```

### Example Commands

```bash
# Hindi language, fast processing
python main.py --language hi --speed 150 --skip-frames 5

# Spanish with accurate depth estimation
python main.py --language es --model hybrid --device cuda

# IP camera from surveillance system
python main.py --ip-camera http://192.168.1.100:8080/video --no-video

# GPU-accelerated with custom speed
python main.py --device cuda --speed 120
```

---

## Configuration

### Main Configuration File: `configs/settings.py`

Modify this file to customize default behavior:

```python
# Camera Configuration
CameraConfig:
  - source: Camera index or video file path
  - width/height: Resolution (640×480 recommended)
  - fps: Target frame rate

# Detector Configuration
DetectorConfig:
  - model_path: Path to YOLOv8-nano ONNX model
  - input_size: 320 (fast) or 640 (accurate)
  - confidence_threshold: 0.40 (40% confidence minimum)
  - iou_threshold: 0.45 (Non-Maximum Suppression)

# Depth Configuration
DepthConfig:
  - model_path: Path to MiDaS model
  - input_size: 256 (small) or 384 (hybrid)

# TTS Configuration
TTSConfig:
  - engine: "piper" or "espeak"
  - language: Language code
  - speed: Words per minute (100-300)

# Alert Configuration
AlertConfig:
  - critical_depth: 0.70 (very close - danger)
  - high_depth: 0.50 (close - attention)
  - medium_depth: 0.30 (nearby - info)
  - cooldowns: Alert frequency limits
```

### Environment Variables

Create a `.env` file for sensitive configuration:

```bash
# .env
CAMERA_INDEX=0
TTS_LANGUAGE=en
TTS_SPEED=185
DEVICE=auto
SKIP_FRAMES=3
```

Load with:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Usage Examples

### Example 1: Basic Real-time Vision Assistance

```bash
python main.py --model small --language en --speed 185
```

**Output**: 
- Real-time visual feed with bounding boxes
- Audio alerts for nearby obstacles
- Console statistics

### Example 2: Multilingual Support

```bash
# Hindi
python main.py --language hi

# Spanish
python main.py --language es

# French
python main.py --language fr
```

**Supported Languages**: en, hi, es, fr, de, it, pt, pl, nl, ru, ja, zh, ar, ur, bn, ta, te, ml, kn, gu, mr, pa, or, as

### Example 3: IP Camera Integration

```bash
python main.py \
  --ip-camera http://192.168.1.100:8080/video \
  --model hybrid \
  --language en \
  --no-video
```

### Example 4: Performance Optimization

```bash
# Fast processing (every 5th frame)
python main.py --skip-frames 5 --model small --device cuda

# Accurate processing (every frame, hybrid model)
python main.py --skip-frames 1 --model hybrid --device cuda
```

### Example 5: Automated Testing Script

```bash
# Run all tests
python tests/test_all.py

# Test specific components
python scripts/test_camera.py
python scripts/test_detector.py
python scripts/test_tts.py
python scripts/test_indic_voices.py
```

---

## Project Structure

```
NetraVision_ai/
├── README.md                          # This file
├── main.py                            # Entry point
├── netra_vision_ai_core.py           # Core pipeline
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup (optional)
│
├── configs/
│   └── settings.py                   # Central configuration
│
├── modules/
│   ├── __init__.py
│   ├── camera.py                     # Camera capture module
│   ├── detector.py                   # YOLOv8 object detection
│   ├── depth.py                      # MiDaS depth estimation
│   ├── spatial.py                    # Spatial mapping & sectors
│   ├── priority.py                   # Alert priority engine
│   ├── tts_engine.py                 # Text-to-Speech synthesis
│   ├── alert_manager.py              # Alert queue management
│   ├── scene_describer.py            # Scene description generation
│   └── ocr_reader.py                 # Optical Character Recognition
│
├── scripts/
│   ├── download_models.py            # Auto-download models
│   ├── download_piper_voices.py      # Download TTS voices
│   ├── test_camera.py                # Camera functionality test
│   ├── test_detector.py              # Detector accuracy test
│   ├── test_tts.py                   # TTS quality test
│   ├── test_indic_voices.py          # Indic language voices test
│   └── benchmark.py                  # Performance benchmarking
│
├── tests/
│   ├── __init__.py
│   └── test_all.py                   # Comprehensive test suite
│
├── models/                           # Pre-trained ML models
│   ├── yolov8n.pt                    # YOLOv8-nano (PyTorch)
│   ├── yolov8n.onnx                  # YOLOv8-nano (ONNX)
│   ├── midas_v21_small_256.pt        # MiDaS Small
│   ├── dpt_hybrid_384.pt             # MiDaS Hybrid
│   └── piper/                        # Piper TTS voices
│       ├── en_US-lessac-medium.onnx
│       ├── hi_IN-priyamvada-medium.onnx
│       └── ...
│
├── docs/
│   └── Major_Project_Report_Extended.md  # Academic documentation
│
├── logs/                             # Application logs
├── audio_cache/                      # Cached TTS audio
└── graphify-out/                     # Analysis outputs
```

---

## Module Documentation

### 📹 Camera Module (`modules/camera.py`)

Handles multi-source camera input.

**Supported Sources**:
- USB webcams (index: 0, 1, 2, ...)
- IP cameras (RTSP, HTTP streams)
- Video files (.mp4, .avi, .mov)

**Example**:
```python
from modules.camera import Camera

camera = Camera(source=0, width=640, height=480)
frame = camera.read()  # Returns numpy array (640×480×3)
```

---

### 🤖 Detector Module (`modules/detector.py`)

YOLOv8-nano object detection with ONNX runtime.

**Features**:
- 640+ object classes
- Input: 320×320 or 640×640
- Output: List of detections with confidence scores

**Example**:
```python
from modules.detector import ObjectDetector

detector = ObjectDetector(
    model_path="models/yolov8n.onnx",
    input_size=320,
    conf_threshold=0.40
)

detections = detector.detect(frame)
# Returns: [{'class': 'person', 'conf': 0.95, 'bbox': [x, y, w, h]}, ...]
```

---

### 📏 Depth Module (`modules/depth.py`)

Monocular depth estimation using MiDaS.

**Models Available**:
- `small`: MiDaS v2.1 Small (256×256, fast)
- `hybrid`: DPT Hybrid (384×384, accurate)

**Example**:
```python
from modules.depth import DepthEstimator

depth = DepthEstimator(model_type="small", device="cuda")
depth_map = depth.estimate(frame)
# Returns: numpy array with relative depth values (0-1)
```

---

### 🗺️ Spatial Module (`modules/spatial.py`)

Maps detected objects to spatial sectors.

**Sectors**:
- Front (center 40%)
- Left (left 30%)
- Right (right 30%)
- Danger zone (<50cm ahead)

**Example**:
```python
from modules.spatial import SpatialMapper

mapper = SpatialMapper()
spatial_objects = mapper.map(detections, depth_map)
# Returns: List of SpatialObject(class, distance, sector)
```

---

### 🎯 Priority Module (`modules/priority.py`)

Intelligent alert prioritization engine.

**Priority Levels**:
1. **CRITICAL**: Danger ahead (<70cm)
2. **HIGH**: Close proximity (50-70cm)
3. **MEDIUM**: Nearby objects (30-50cm)
4. **LOW**: General information (>50cm)

**Example**:
```python
from modules.priority import PriorityEngine, AlertPriority

priority_engine = PriorityEngine()
alerts = priority_engine.generate_alerts(spatial_objects)
# Returns: List of Alert(message, priority, cooldown)
```

---

### 🗣️ TTS Module (`modules/tts_engine.py`)

Text-to-speech synthesis with multiple backends.

**Supported Engines**:
- **Piper**: Lightweight, offline, 15+ languages
- **espeak-ng**: Fallback engine, wide language support

**Example**:
```python
from modules.tts_engine import TTSEngine

tts = TTSEngine(language="en", speed=185)
tts.speak("Person detected ahead")
```

---

### 🔔 Alert Manager (`modules/alert_manager.py`)

Queue-based alert management with cooldowns.

**Features**:
- Thread-safe alert queue
- Duplicate suppression
- Cooldown management
- Statistics tracking

**Example**:
```python
from modules.alert_manager import AlertManager

alert_manager = AlertManager(tts_engine)
alert_manager.queue_alert("Red car on the left", priority=AlertPriority.HIGH)
```

---

### 👁️ Scene Describer (`modules/scene_describer.py`)

Generates natural language descriptions of scenes.

**Example**:
```python
description = scene_describer.describe(detections, spatial_objects)
# Output: "Person 2 meters ahead. Red car on the left. Tree blocking right path."
```

---

### 📖 OCR Module (`modules/ocr_reader.py`)

Optical character recognition for reading text in images.

**Example**:
```python
text = ocr_reader.read_text(frame)
# Output: Text detected in image
```

---

## Supported Languages

NetraVision AI supports 15+ languages via Piper TTS:

| Code | Language | Voice |
|------|----------|-------|
| en | English | Lessac (Medium) |
| hi | Hindi | Priyamvada, Rohan |
| es | Spanish | Karla |
| fr | French | Blanche |
| de | German | Kim |
| it | Italian | Gian-Maria |
| pt | Portuguese | Claudio |
| pl | Polish | Iwona |
| nl | Dutch | Ruben |
| ru | Russian | Alexander |
| ja | Japanese | Asano |
| zh | Chinese (Mandarin) | Huayan |
| ar | Arabic | Jamal |
| ur | Urdu | Fariha |
| bn | Bengali | Rajesh |

**Download additional voices**:
```bash
python scripts/download_piper_voices.py --language hi
```

---

## Performance & Hardware Requirements

### Benchmark Results

| Configuration | FPS | Latency | Memory |
|---------------|-----|---------|--------|
| Small (CPU) | 15 | 66ms | 2.1 GB |
| Small (GPU) | 25 | 40ms | 2.8 GB |
| Hybrid (CPU) | 8 | 125ms | 2.5 GB |
| Hybrid (GPU) | 20 | 50ms | 3.2 GB |

### GPU Acceleration

**NVIDIA GPUs**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python main.py --device cuda
```

**AMD GPUs**:
```bash
pip install torch-directml
python main.py --device directml
```

### Optimization Tips

1. **Reduce latency**: Use `--skip-frames 5` to process fewer frames
2. **Improve accuracy**: Use `--model hybrid` (slower but more accurate)
3. **GPU acceleration**: `--device cuda` (requires NVIDIA GPU)
4. **Resolution**: Smaller camera input improves speed
5. **Audio caching**: Repeated alerts use cached audio

---

## Troubleshooting

### Issue 1: Camera Not Detected

**Problem**: "Cannot open camera" error

**Solutions**:
```bash
# Check camera index
python scripts/test_camera.py

# Try different camera index
python main.py --camera 1

# For IP cameras, verify URL
python main.py --ip-camera http://192.168.1.100:8080/video
```

---

### Issue 2: Low Frames Per Second (FPS)

**Problem**: Processing is slow

**Solutions**:
```bash
# Skip frames for faster processing
python main.py --skip-frames 5

# Use smaller depth model
python main.py --model small

# Use GPU acceleration
python main.py --device cuda

# Reduce resolution in configs/settings.py
```

---

### Issue 3: Models Not Found

**Problem**: "Model file not found" error

**Solutions**:
```bash
# Download all models
python scripts/download_models.py

# Download TTS voices
python scripts/download_piper_voices.py

# Check model paths in configs/settings.py
```

---

### Issue 4: No Audio Output

**Problem**: TTS produces no sound

**Solutions**:
```bash
# Test TTS separately
python scripts/test_tts.py

# Check system volume and speaker connectivity
# Verify language support
python main.py --language en

# Check available voices
python scripts/download_piper_voices.py --list
```

---

### Issue 5: Out of Memory (OOM)

**Problem**: "CUDA out of memory" or system freeze

**Solutions**:
```bash
# Use CPU instead of GPU
python main.py --device cpu

# Use smaller model
python main.py --model small

# Increase frame skip
python main.py --skip-frames 10

# Reduce camera resolution in configs/settings.py
```

---

### Issue 6: GPU Not Detected

**Problem**: "CUDA device not found" even with GPU present

**Solutions**:
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability in Python
python -c "import torch; print(torch.cuda.is_available())"
```

---

### Performance Profiling

```bash
# Run benchmark script
python scripts/benchmark.py

# Outputs detailed metrics:
# - FPS per component
# - Memory usage
# - Latency breakdown
# - Bottleneck identification
```

---

## Contributing

We welcome contributions! Please follow these guidelines:

### Getting Started
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature: description"`
5. Push to branch: `git push origin feature/your-feature`
6. Create a Pull Request

### Code Standards
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new features
- Update documentation

### Testing Before Submission
```bash
python tests/test_all.py
python -m pylint modules/*.py
python -m black modules/*.py
```

### Areas for Contribution
- 🌍 Support for additional languages
- 🚀 Performance optimizations
- 🐛 Bug fixes and edge case handling
- 📚 Documentation improvements
- 🧪 Additional test coverage
- 🎯 New feature implementations

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Citation

If you use NetraVision AI in your research or project, please cite:

```bibtex
@software{netravision_ai,
  author = {Harshit},
  title = {NetraVision AI: An Intelligent Assistive Vision System for the Visually Impaired},
  year = {2024},
  url = {https://github.com/Tyagism/NetraVision-AI}
}
```

---

## Acknowledgments

This project was developed as part of a B.Tech. (Artificial Intelligence and Machine Learning) major project at the University School of Automation and Robotics, Guru Gobind Singh Indraprastha University, Delhi.

**Special thanks to**:
- The YOLOv8 and MiDaS teams for their excellent models
- Piper TTS for multilingual speech synthesis
- The OpenCV and PyTorch communities
- All contributors and testers

---

## Support & Contact

For questions, issues, or suggestions:

- 🐛 **GitHub Issues**: [Report bugs and request features](https://github.com/Tyagism/NetraVision-AI/issues)
- 💬 **Discussions**: [Join the community](https://github.com/Tyagism/NetraVision-AI/discussions)
- 📖 **Documentation**: See [docs/](docs/) for detailed technical documentation

---

## Roadmap

### Version 2.0 (Planned)
- [ ] Real-time voice commands for control
- [ ] Integration with wearable devices (smartwatch, glasses)
- [ ] Cloud-based model serving
- [ ] Mobile app (iOS/Android)
- [ ] Gesture recognition for hand signals
- [ ] Improved depth with stereo cameras

### Version 3.0 (Future Vision)
- [ ] AR glasses integration
- [ ] Full SLAM (Simultaneous Localization and Mapping)
- [ ] Semantic scene understanding
- [ ] Personalized learning from user feedback
- [ ] Multi-person tracking and identification

---

## Disclaimer

This system is designed as an assistive technology aid and should **not replace professional medical advice or primary vision correction methods**. Users should operate this system responsibly and in safe environments. Always consult with healthcare professionals for vision-related concerns.

---

**Last Updated**: May 2026  
**Version**: 1.0.0  
**Status**: Active Development
