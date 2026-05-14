# 🚀 Running NetraVision AI with Streamlit

This guide explains how to run the web interface for NetraVision AI.

## Quick Start

### 1. Install Dependencies

```bash
# Install requirements (including Streamlit)
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
# Run the web interface
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Features

### 📺 Live Stream Tab
- Real-time camera processing
- Object detection visualization
- Spatial awareness mapping
- Audio alert generation

### 🖼️ Image Analysis Tab
- Upload and analyze individual images
- View detected objects with confidence scores
- Analyze depth maps

### 📊 Analytics Tab
- System configuration overview
- Performance metrics
- Tips for optimization

## Configuration

### Camera Source Options

1. **Webcam** - Default system camera (index 0)
2. **IP Camera** - Remote camera via RTSP/HTTP
3. **Image Upload** - Static image analysis

### Advanced Settings

- **Depth Model**: `small` (fast) or `hybrid` (accurate)
- **Device**: `auto`, `cpu`, or `cuda` (GPU)
- **Language**: 13+ languages supported
- **Speech Speed**: 100-250 WPM
- **Frame Skip**: Process every Nth frame

## System Requirements

### Minimum (CPU Mode)
- Python 3.11+
- 4GB RAM
- Webcam (optional)

### Recommended (GPU Mode)
- Python 3.11+
- 8GB+ RAM
- NVIDIA GPU with 2GB+ VRAM
- CUDA 11.8+

## Troubleshooting

### App won't start
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with verbose output
streamlit run app.py --logger.level=debug
```

### Slow performance
- Increase "Frame Skip" in sidebar
- Switch to "small" depth model
- Use CPU mode if GPU memory is limited

### Camera not working
- Check camera permissions
- Try a different camera index
- For IP cameras, verify URL format: `rtsp://192.168.x.x/stream`

## Deployment

### Local Network Access
```bash
streamlit run app.py --server.address 0.0.0.0
```

### Docker (Coming Soon)
```bash
docker build -t netravision-ai .
docker run -p 8501:8501 netravision-ai
```

### Cloud Deployment
Streamlit Community Cloud: https://streamlit.io/cloud

## Environment Variables

Create a `.env` file for configuration:
```env
# TTS Settings
TTS_LANGUAGE=en
TTS_SPEED=185

# Model Settings
DEPTH_MODEL=small
DEVICE=auto

# Camera Settings
CAMERA_SOURCE=0
```

## Performance Tips

1. **Frame Skip**: Increase for slower devices (default: 3)
2. **Depth Model**: Use "small" for real-time processing
3. **Device Selection**: Use GPU if available
4. **Resolution**: Lower if lagging
5. **Language**: Avoid heavy multilingual processing on CPU

## API Reference

### Streamlit App Structure
- `app.py` - Main Streamlit interface
- `netra_vision_ai_core.py` - Core pipeline
- `modules/` - Individual components

### Key Functions

```python
# Run the pipeline
streamlit run app.py

# Clear cache and restart
streamlit cache clear
```

## Next Steps

- 🎥 Configure your camera source
- 🗣️ Select your preferred language
- ⚙️ Adjust performance settings
- 🚀 Start processing!

For issues or feature requests: https://github.com/Tyagism/NetraVision-AI/issues
