"""
NetraVisionAi — Object Detection Module
YOLOv8-nano via ONNX Runtime
"""

import numpy as np
import cv2
import time
import os
from typing import List, Dict


class ObjectDetector:
    # COCO 80 class names
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    # Objects that pose potential danger to blind person
    DANGER_OBJECTS = {
        'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train',
        'dog', 'knife', 'scissors', 'skateboard', 'sports ball'
    }
    
    # Objects relevant for navigation
    NAVIGATION_OBJECTS = {
        'person', 'chair', 'bench', 'potted plant', 'fire hydrant',
        'stop sign', 'traffic light', 'parking meter', 'couch',
        'bed', 'dining table', 'door', 'stairs'
    }

    def __init__(self, model_path: str = "models/yolov8n.onnx",
                 input_size: int = 320,
                 conf_threshold: float = 0.40,
                 iou_threshold: float = 0.45):
        """
        Initialize YOLOv8-nano object detector.
        
        Args:
            model_path: Path to ONNX model file
            input_size: Input image size (320=fast, 640=accurate)
            conf_threshold: Minimum confidence to keep detection
            iou_threshold: NMS IoU threshold
        """
        import onnxruntime as ort
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run: python scripts/download_models.py"
            )
        
        # Setup ONNX Runtime session
        providers = ['CPUExecutionProvider']
        
        # Try GPU if available
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("[Detector] 🟢 Using GPU (CUDA)")
        else:
            print("[Detector] Using CPU")
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Get model input info
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        
        # Warmup — first inference is always slow
        print(f"[Detector] Warming up model (input: {input_size}x{input_size})...")
        dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})
        
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[Detector] ✅ YOLOv8-nano loaded ({model_size:.1f} MB)")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of detections:
            [
                {
                    'class': 'person',
                    'confidence': 0.87,
                    'bbox': (x1, y1, x2, y2),     # pixel coordinates
                    'center': (cx, cy),             # center point
                    'area_ratio': 0.15,             # fraction of frame area
                    'is_danger': False,
                    'is_navigation': True,
                },
                ...
            ]
        """
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self._postprocess(outputs[0], orig_w, orig_h)
        
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO input"""
        # Resize
        img = cv2.resize(frame, (self.input_size, self.input_size))
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img

    def _postprocess(self, output: np.ndarray, 
                     orig_w: int, orig_h: int) -> List[Dict]:
        """Parse YOLOv8 output format"""
        predictions = output[0]  # Shape: (84, N) — 4 bbox + 80 classes
        
        # Transpose if needed
        if predictions.shape[0] == 84:
            predictions = predictions.T  # Now (N, 84)
        
        # Scale factors
        sx = orig_w / self.input_size
        sy = orig_h / self.input_size
        frame_area = orig_w * orig_h
        
        boxes = []
        scores = []
        class_ids = []
        
        for pred in predictions:
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence < self.conf_threshold:
                continue
            
            # Extract box (center x, center y, width, height)
            cx, cy, w, h = pred[:4]
            
            x1 = int((cx - w / 2) * sx)
            y1 = int((cy - h / 2) * sy)
            x2 = int((cx + w / 2) * sx)
            y2 = int((cy + h / 2) * sy)
            
            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w, x2)
            y2 = min(orig_h, y2)
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))
            class_ids.append(int(class_id))
        
        # Apply NMS (Non-Maximum Suppression)
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                # NMSBoxes expects [x, y, w, h]
                [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
                scores,
                self.conf_threshold,
                self.iou_threshold
            )
            if len(indices) > 0:
                indices = indices.flatten()
            else:
                indices = []
        else:
            indices = []
        
        # Build final detections
        detections = []
        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            class_name = self.COCO_CLASSES[class_ids[i]]
            
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            box_area = (x2 - x1) * (y2 - y1)
            
            detections.append({
                'class': class_name,
                'confidence': scores[i],
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'area_ratio': box_area / frame_area,
                'is_danger': class_name in self.DANGER_OBJECTS,
                'is_navigation': class_name in self.NAVIGATION_OBJECTS,
            })
        
        # Sort by area (larger objects = likely closer = more important)
        detections.sort(key=lambda d: d['area_ratio'], reverse=True)
        
        return detections

    def detect_and_draw(self, frame: np.ndarray) -> tuple:
        """Detect objects AND draw bounding boxes (for debugging/demo)"""
        detections = self.detect(frame)
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['confidence']:.0%}"
            
            # Color: Red for danger, Green for navigation, Blue for other
            if det['is_danger']:
                color = (0, 0, 255)      # Red
            elif det['is_navigation']:
                color = (0, 255, 0)      # Green
            else:
                color = (255, 165, 0)    # Orange
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return detections, vis_frame


# ═══════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  Object Detector Test — Press 'q' to quit")
    print("=" * 50)
    
    detector = ObjectDetector(
        model_path="models/yolov8n.onnx",
        input_size=320,
        conf_threshold=0.40
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit(1)
    
    fps_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        detections, vis_frame = detector.detect_and_draw(frame)
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_list.append(fps)
        
        # Show stats
        avg_fps = np.mean(fps_list[-30:])
        cv2.putText(vis_frame, f"FPS: {avg_fps:.1f} | Objects: {len(detections)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Print detections
        if detections:
            det_str = ", ".join([f"{d['class']}({d['confidence']:.0%})" for d in detections[:5]])
            print(f"\r  [{avg_fps:.1f} FPS] Detected: {det_str}          ", end="")
        
        cv2.imshow("NetraVisionAi Detector Test", vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n\nAverage FPS: {np.mean(fps_list):.1f}")