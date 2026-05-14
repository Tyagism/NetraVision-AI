# netra_vision_ai_core.py
"""
NetraVisionAi — Core Pipeline
Camera → Detection + Depth → Spatial Mapping → Priority Alerts → TTS

Threads:
  Thread 1: Camera capture (continuous — handled by Camera class)
  Thread 2: FAST lane — YOLO + Depth + Alerts (every Nth frame)
  Thread 3: TTS speech output (handled by AlertManager/TTSEngine)
"""

import cv2
import numpy as np
import threading
import time

from modules.camera import Camera
from modules.detector import ObjectDetector
from modules.depth import DepthEstimator
from modules.spatial import SpatialMapper
from modules.priority import PriorityEngine, AlertPriority, Alert
from modules.tts_engine import TTSEngine, SpeechPriority
from modules.alert_manager import AlertManager


class NetraVisionAi:
    """
    Main NetraVisionAi pipeline.
    Captures camera → detects objects → estimates depth →
    maps to spatial descriptions → generates audio alerts.
    """

    def __init__(self,
                 camera_source=0,
                 depth_model="small",
                 depth_device="auto",
                 tts_language="en",
                 tts_speed=185,
                 show_video=True,
                 process_every_n=3):

        self.show_video = show_video
        self.process_every_n = process_every_n
        self.running = False

        print("=" * 60)
        print("  🦯 NetraVisionAi — Initializing...")
        print("=" * 60)

        # ─── 1. Camera ───
        print("\n[1/6] Camera...")
        self.camera = Camera(
            source=camera_source,
            width=640,
            height=480,
        )

        # ─── 2. Object Detector ───
        print("\n[2/6] Object Detector (YOLOv8-nano)...")
        self.detector = ObjectDetector(
            model_path="models/yolov8n.onnx",
            input_size=320,
            conf_threshold=0.40,
        )

        # ─── 3. Depth Estimator ───
        print("\n[3/6] Depth Estimator...")
        self.depth = DepthEstimator(
            model_type=depth_model,
            device=depth_device,
        )

        # ─── 4. Spatial Mapper ───
        print("\n[4/6] Spatial Mapper...")
        self.spatial = SpatialMapper()

        # ─── 5. Priority Engine ───
        print("\n[5/6] Priority Engine...")
        self.priority = PriorityEngine()

        # ─── 6. TTS + Alert Manager ───
        print("\n[6/6] TTS + Alert Manager...")
        self.tts = TTSEngine(
            language=tts_language,
            speed=tts_speed,
        )
        self.alert_manager = AlertManager(self.tts)

        # ─── Shared state (thread-safe) ───
        self._lock = threading.Lock()
        self._last_detections = []      # List[Dict]
        self._last_depth = None         # np.ndarray
        self._last_spatial = []         # List[SpatialObject]
        self._last_alerts = []          # List[Alert]

        # ─── Stats ───
        self.stats = {
            "frames_processed": 0,
            "objects_detected": 0,
            "alerts_generated": 0,
            "avg_fps": 0,
        }
        self.fps_list = []

        print("\n" + "=" * 60)
        print("  ✅ All modules initialized!")
        print("=" * 60)

    def start(self):
        """Start the full pipeline."""
        print("\n🚀 Starting NetraVisionAi...")

        # Start camera (returns bool)
        if not self.camera.start():
            print("❌ Failed to start camera!")
            return

        # Start alert system (starts TTS internally)
        self.alert_manager.start()

        # Startup announcement
        self.alert_manager.speak_startup(
            "NetraVisionAi started. I am scanning your surroundings."
        )

        self.running = True

        # Start processing thread
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._process_thread.start()

        print("\n🜢 NetraVisionAi is LIVE!")
        print("   Listening through your eyes...")
        print("   Press 'q' in video window or Ctrl+C to stop.\n")

        # Main loop (handles video display + keyboard)
        try:
            self._display_loop()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop everything."""
        if not self.running:
            return
        self.running = False

        # Goodbye message
        self.alert_manager.speak_shutdown(
            "NetraVisionAi stopping. Goodbye."
        )
        time.sleep(2)  # Let goodbye play

        # Stop all modules
        self.camera.stop()
        self.alert_manager.stop()

        if self.show_video:
            cv2.destroyAllWindows()

        self._print_stats()
        print("\n🔴 NetraVisionAi stopped.")

    # ══════════════════════════════════════════════════
    # PROCESSING THREAD (FAST LANE)
    # ══════════════════════════════════════════════════

    def _process_loop(self):
        """
        FAST LANE — Main processing loop.
        Runs detection + depth on every Nth frame.
        """
        last_processed = -1

        while self.running:
            # Get latest frame from camera
            frame, count = self.camera.get_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            # Skip if same frame
            if count == last_processed:
                time.sleep(0.01)
                continue

            # Skip frames for performance
            if count % self.process_every_n != 0:
                time.sleep(0.005)
                continue

            last_processed = count
            start_time = time.time()

            try:
                # ─── 1. Object Detection ───
                # Returns List[Dict] with keys: 'class', 'confidence',
                # 'bbox', 'center', 'area_ratio', 'is_danger', 'is_navigation'
                detections = self.detector.detect(frame)

                # ─── 2. Depth Estimation ───
                # Returns float32 depth map (0→1, higher=closer)
                depth_map = self.depth.estimate(frame)

                # ─── 3. Spatial Mapping ───
                h, w = frame.shape[:2]
                spatial_objects = self.spatial.map_objects(
                    detections, depth_map, w, h
                )

                # ─── 4. Generate Alerts ───
                alerts = self.priority.generate_alerts(spatial_objects)

                # ─── 5. Send to TTS ───
                if alerts:
                    self.alert_manager.process_alerts(alerts)

                # ─── Update shared state (thread-safe) ───
                with self._lock:
                    self._last_detections = detections
                    self._last_depth = depth_map
                    self._last_spatial = spatial_objects
                    self._last_alerts = alerts

                # ─── Stats ───
                elapsed = time.time() - start_time
                fps = 1.0 / max(elapsed, 0.001)
                self.fps_list.append(fps)

                self.stats["frames_processed"] += 1
                self.stats["objects_detected"] += len(detections)
                self.stats["alerts_generated"] += len(alerts)
                self.stats["avg_fps"] = float(np.mean(self.fps_list[-30:]))

            except Exception as e:
                print(f"\n[Pipeline] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    # ══════════════════════════════════════════════════
    # DISPLAY THREAD (MAIN THREAD)
    # ══════════════════════════════════════════════════

    def _display_loop(self):
        """Video display loop (runs on main thread)."""
        while self.running:
            frame, count = self.camera.get_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            if self.show_video:
                vis_frame = self._draw_visualization(frame)
                cv2.imshow("NetraVisionAi Live", vis_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n'q' pressed — stopping...")
                    self.running = False
                    break
                elif key == ord('p'):
                    if self.alert_manager.is_active:
                        self.alert_manager.pause()
                        print("⏸️ Alerts paused")
                    else:
                        self.alert_manager.resume()
                        print("▶️ Alerts resumed")
            else:
                # No video — just print status periodically
                if self.stats["frames_processed"] % 30 == 0 and self.stats["frames_processed"] > 0:
                    self._print_live_status()
                time.sleep(0.033)

    # ══════════════════════════════════════════════════
    # VISUALIZATION
    # ══════════════════════════════════════════════════

    def _draw_visualization(self, frame):
        """Draw debug visualization on frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Thread-safe read of latest results
        with self._lock:
            detections = self._last_detections
            depth_map = self._last_depth
            spatial_objects = self._last_spatial
            alerts = self._last_alerts

        # ─── Draw detection bounding boxes ───
        # detections are dicts: det['bbox'], det['class'], etc.
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['confidence']:.0%}"

            if det['is_danger']:
                color = (0, 0, 255)       # Red for danger
            elif det.get('is_navigation', False):
                color = (0, 255, 0)       # Green for navigation
            else:
                color = (255, 165, 0)     # Orange for other

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 8),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # ─── Draw spatial info (bottom of screen) ───
        # spatial_objects have: .class_name, .position, .distance, .is_danger
        for i, obj in enumerate(spatial_objects[:5]):
            txt = f"{obj.class_name}: {obj.position}, {obj.distance}"

            if obj.is_danger:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)

            y_pos = h - 20 - i * 22
            cv2.putText(vis, txt, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(vis, txt, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # ─── Draw active alerts (top-right) ───
        # alerts have: .priority, .message
        for i, alert in enumerate(alerts[:3]):
            icon = {0: "!!!", 1: "! ", 2: "* ", 3: "  "}.get(int(alert.priority), "  ")
            txt = f"{icon} {alert.message}"
            color = {
                0: (0, 0, 255),      # Critical = Red
                1: (0, 165, 255),    # High = Orange
                2: (0, 255, 255),    # Medium = Yellow
                3: (0, 255, 0),      # Low = Green
            }.get(int(alert.priority), (255, 255, 255))

            cv2.putText(vis, txt, (w - 450, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3)
            cv2.putText(vis, txt, (w - 450, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        # ─── Draw stats (top-left) ───
        avg_fps = self.stats["avg_fps"]
        queue_size = self.alert_manager.get_queue_size()

        stats_lines = [
            f"FPS: {avg_fps:.1f}",
            f"Objects: {len(detections)}",
            f"Alert Queue: {queue_size}",
            f"Processed: {self.stats['frames_processed']}",
        ]
        for i, txt in enumerate(stats_lines):
            cv2.putText(vis, txt, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
            cv2.putText(vis, txt, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)

        # ─── Draw depth overlay (bottom-right corner) ───
        if depth_map is not None:
            try:
                depth_small = cv2.resize(depth_map, (160, 120))
                depth_color = self.depth.get_depth_colormap(depth_small)
                vis[h - 120:h, w - 160:w] = depth_color
            except Exception:
                pass

        # ─── Controls help ───
        cv2.putText(vis, "q=quit | p=pause alerts", (w - 250, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        return vis

    def _print_live_status(self):
        """Print status to terminal (when no video)."""
        fps = self.stats["avg_fps"]
        objs = len(self._last_detections)
        queue_size = self.alert_manager.get_queue_size()
        print(f"\r  FPS: {fps:.1f} | Objects: {objs} | "
              f"Alert queue: {queue_size} | "
              f"Processed: {self.stats['frames_processed']}    ",
              end="", flush=True)

    def _print_stats(self):
        """Print final session stats."""
        print(f"\n{'=' * 55}")
        print(f"  📊 SESSION STATS")
        print(f"{'=' * 55}")
        print(f"  Frames processed:  {self.stats['frames_processed']}")
        print(f"  Objects detected:  {self.stats['objects_detected']}")
        print(f"  Alerts generated:  {self.stats['alerts_generated']}")
        print(f"  Alerts spoken:     {self.alert_manager.total_alerts_spoken}")
        if self.fps_list:
            print(f"  Average FPS:       {np.mean(self.fps_list):.1f}")
            print(f"  Best FPS:          {np.max(self.fps_list):.1f}")
        print(f"{'=' * 55}")