"""
NetraVisionAi — Camera Module
Handles frame capture from webcam / phone camera / video file
"""

import cv2
import threading
import time
import numpy as np
from typing import Optional, Tuple


class Camera:
    def __init__(self, source=0, width=640, height=480, fps=30):
        """
        Initialize camera.
        
        Args:
            source: Camera index (0=default) or URL string for IP camera
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.frame = None
        self.frame_count = 0
        self.running = False
        self.lock = threading.Lock()
        
    def start(self) -> bool:
        """Start camera capture in background thread"""
        print(f"[Camera] Opening source: {self.source}")
        
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"[Camera] ❌ Failed to open camera {self.source}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Read actual properties (camera may not support requested values)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[Camera] ✅ Opened — Resolution: {actual_w}x{actual_h} @ {actual_fps}fps")
        
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def _capture_loop(self):
        """Continuously capture frames in background"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
            else:
                time.sleep(0.01)
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], int]:
        """
        Get the latest frame.
        
        Returns:
            (frame, frame_count) or (None, 0) if no frame available
        """
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.frame_count
        return None, 0
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("[Camera] Stopped")
    
    def __del__(self):
        self.stop()


# ═══════════════════════════════════════════════
# QUICK TEST — Run this file directly
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  Camera Test — Press 'q' to quit")
    print("=" * 50)
    
    cam = Camera(source=0, width=640, height=480)
    
    if not cam.start():
        print("Failed to start camera!")
        exit(1)
    
    time.sleep(1)  # Wait for camera to warm up
    
    while True:
        frame, count = cam.get_frame()
        if frame is not None:
            # Show frame count on image
            cv2.putText(frame, f"Frame: {count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("NetraVisionAi Camera Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.stop()
    cv2.destroyAllWindows()
    print(f"Total frames captured: {count}")