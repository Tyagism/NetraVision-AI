"""
NetraVisionAI — Streamlit Web Interface
Interactive web app for real-time vision assistance
"""

import streamlit as st
import cv2
import numpy as np
import threading
import time
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from netra_vision_ai_core import NetraVisionAi


# ─── Page Configuration ───
st.set_page_config(
    page_title="🦯 NetraVision AI",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .feature-box {
        padding: 1em;
        border-radius: 0.5em;
        background-color: #f0f2f6;
        margin: 0.5em 0;
    }
    .alert-box {
        padding: 1em;
        border-left: 4px solid #ff6b6b;
        background-color: #ffe0e0;
        border-radius: 0.3em;
    }
</style>
""", unsafe_allow_html=True)

# ─── Initialize Session State ───
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None


# ─── Sidebar Configuration ───
st.sidebar.markdown("## ⚙️ Configuration")

# Model Selection
depth_model = st.sidebar.selectbox(
    "Depth Model",
    ["small", "hybrid"],
    help="small = faster, hybrid = more accurate"
)

# Device Selection
device = st.sidebar.selectbox(
    "Processing Device",
    ["auto", "cpu", "cuda"],
    help="auto = automatic selection, cuda = GPU, cpu = CPU only"
)

# Language Selection
language = st.sidebar.selectbox(
    "Audio Language",
    [
        "en", "hi", "bn", "te", "mr", "ta", "ur", "gu",
        "kn", "ml", "pa", "or", "as"
    ],
    help="Language for audio descriptions"
)

# TTS Speed
tts_speed = st.sidebar.slider(
    "Speech Speed (WPM)",
    min_value=100,
    max_value=250,
    value=185,
    step=5,
    help="Words per minute"
)

# Frame Skip
frame_skip = st.sidebar.slider(
    "Frame Skip",
    min_value=1,
    max_value=10,
    value=3,
    help="Process every Nth frame for performance"
)

# Camera Source
camera_source = st.sidebar.selectbox(
    "Camera Source",
    ["Webcam", "IP Camera", "Image Upload"],
    help="Select input source"
)

ip_camera_url = None
if camera_source == "IP Camera":
    ip_camera_url = st.sidebar.text_input(
        "IP Camera URL",
        placeholder="rtsp://192.168.1.100:554/stream"
    )

st.sidebar.markdown("---")

# Show Help
if st.sidebar.checkbox("📖 Show Help", value=False):
    st.sidebar.markdown("""
    ### How to Use:
    1. **Configure** your preferences in the sidebar
    2. **Select** your input source (webcam/camera/image)
    3. **Click** "Start Processing" to begin
    4. The system will:
       - Detect objects in real-time
       - Estimate depth/distance
       - Generate spatial descriptions
       - Provide audio alerts
    
    ### Features:
    - 🎥 Real-time object detection (640+ classes)
    - 📏 Depth estimation
    - 🗺️ Spatial awareness
    - 🔊 Multilingual audio output
    - ⚡ Low-latency processing
    """)


# ─── Main Content ───
st.markdown("""
<div class="main-header">🦯 NetraVision AI</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-box">
An intelligent assistive vision system that helps visually impaired individuals 
navigate and understand their environment in real-time.
</div>
""", unsafe_allow_html=True)

# ─── Tabs ───
tab1, tab2, tab3 = st.tabs(["🎥 Live Stream", "🖼️ Image Analysis", "📊 Analytics"])

with tab1:
    st.header("Live Stream Processing")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
    
    with col2:
        st.subheader("Controls")
        
        if st.button("🟢 Start Processing", key="start_btn", use_container_width=True):
            st.session_state.processing = True
            st.rerun()
        
        if st.button("🔴 Stop Processing", key="stop_btn", use_container_width=True):
            st.session_state.processing = False
            if st.session_state.pipeline:
                st.session_state.pipeline.stop()
            st.rerun()
        
        st.subheader("Status")
        status_placeholder = st.empty()
    
    # Processing Loop
    if st.session_state.processing:
        try:
            # Initialize pipeline
            if st.session_state.pipeline is None:
                status_placeholder.info("🔄 Initializing pipeline...")
                
                camera_idx = 0
                if camera_source == "IP Camera" and ip_camera_url:
                    camera_idx = ip_camera_url
                
                st.session_state.pipeline = NetraVisionAi(
                    camera_source=camera_idx,
                    depth_model=depth_model,
                    depth_device=device,
                    tts_language=language,
                    tts_speed=tts_speed,
                    show_video=False,
                    process_every_n=frame_skip
                )
                status_placeholder.success("✅ Pipeline ready")
            
            # Capture and process frames
            pipeline = st.session_state.pipeline
            
            cols = st.columns(2)
            with cols[0]:
                with st.spinner("📹 Capturing frames..."):
                    # Get frame from camera
                    frame, detections, spatial_info = pipeline.process_frame()
                    
                    if frame is not None:
                        # Display frame
                        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
                        
                        # Display info
                        with info_placeholder.container():
                            st.markdown("### 📊 Detection Info")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Objects Detected", len(detections))
                            with col_b:
                                st.metric("Alerts Generated", len(spatial_info))
            
            with cols[1]:
                st.markdown("### 🎯 Latest Detections")
                if detections:
                    for det in detections[:5]:  # Show top 5
                        st.write(f"• **{det.get('class', 'Unknown')}** - Confidence: {det.get('confidence', 0):.1%}")
                else:
                    st.info("No objects detected")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.processing = False


with tab2:
    st.header("Image Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Select an image to analyze"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Analyze Image", use_container_width=True):
                try:
                    # Convert PIL to OpenCV format
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Initialize pipeline if needed
                    if st.session_state.pipeline is None:
                        with st.spinner("Initializing..."):
                            st.session_state.pipeline = NetraVisionAi(
                                camera_source=0,
                                depth_model=depth_model,
                                depth_device=device,
                                tts_language=language,
                                tts_speed=tts_speed,
                                show_video=False
                            )
                    
                    # Analyze
                    with st.spinner("🔬 Analyzing image..."):
                        pipeline = st.session_state.pipeline
                        detections = pipeline.detector.detect(img_cv)
                        depth_map = pipeline.depth.estimate(img_cv)
                    
                    # Display results
                    st.success("✅ Analysis complete!")
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.subheader("Detected Objects")
                        if detections:
                            for det in detections:
                                st.write(f"• **{det.get('class', 'Unknown')}**")
                                st.write(f"  Confidence: {det.get('confidence', 0):.1%}")
                        else:
                            st.info("No objects detected")
                    
                    with col_res2:
                        st.subheader("Depth Analysis")
                        if depth_map is not None:
                            st.metric("Depth Map Size", f"{depth_map.shape}")
                            st.metric("Min Depth", f"{depth_map.min():.2f}")
                            st.metric("Max Depth", f"{depth_map.max():.2f}")
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")


with tab3:
    st.header("📊 System Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", "✅ Ready" if st.session_state.pipeline is None else "🟢 Running")
    
    with col2:
        st.metric("Camera Source", camera_source)
    
    with col3:
        st.metric("Processing Device", device)
    
    with col4:
        st.metric("Audio Language", language)
    
    st.markdown("---")
    
    st.subheader("System Information")
    sys_info = {
        "Depth Model": depth_model,
        "TTS Speed": f"{tts_speed} WPM",
        "Frame Skip": frame_skip,
        "Language": language
    }
    
    for key, value in sys_info.items():
        col_k, col_v = st.columns([1, 3])
        with col_k:
            st.write(f"**{key}:**")
        with col_v:
            st.write(value)
    
    st.markdown("---")
    
    st.subheader("Performance Tips")
    st.markdown("""
    - **Increase Frame Skip** for better performance on slower devices
    - **Use "small" depth model** for real-time processing
    - **Select "cpu"** if GPU memory is limited
    - **Lower TTS Speed** for clearer audio
    """)


# ─── Footer ───
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
🦯 NetraVision AI — Making Vision Accessible | 
<a href="https://github.com/Tyagism/NetraVision-AI">GitHub</a>
</div>
""", unsafe_allow_html=True)
