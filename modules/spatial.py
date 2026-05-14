"""
NetraVisionAi — Spatial Mapper
Converts detections + depth into spatial descriptions
"Person to your left, close" / "Car ahead, very close"
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class SpatialObject:
    """An object mapped to spatial coordinates"""
    class_name: str
    confidence: float
    position: str           # "far left", "left", "ahead", "right", "far right"
    distance: str           # "very close", "close", "nearby", "far"
    depth_value: float      # Raw depth value (0-1, higher=closer)
    is_danger: bool
    is_navigation: bool
    bbox: tuple
    center: tuple


class SpatialMapper:
    """Maps detected objects to spatial descriptions with depth"""
    
    # Horizontal zones (proportion of frame width)
    ZONES = [
        (0.00, 0.20, "far left"),
        (0.20, 0.40, "to your left"),
        (0.40, 0.60, "ahead"),
        (0.60, 0.80, "to your right"),
        (0.80, 1.00, "far right"),
    ]

    def __init__(self):
        print("[Spatial] ✅ Spatial Mapper ready")

    def map_objects(self, detections: List[Dict], 
                    depth_map: np.ndarray,
                    frame_width: int, 
                    frame_height: int) -> List[SpatialObject]:
        """
        Map detected objects to spatial descriptions.
        
        Args:
            detections: List from ObjectDetector.detect()
            depth_map: Depth map from DepthEstimator.estimate()
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            List of SpatialObject sorted by proximity (closest first)
        """
        spatial_objects = []
        
        for det in detections:
            cx, cy = det['center']
            
            # --- Horizontal Position ---
            h_ratio = cx / frame_width
            position = "ahead"  # default
            for low, high, label in self.ZONES:
                if low <= h_ratio < high:
                    position = label
                    break
            
            # --- Depth / Distance ---
            depth_value = self._get_object_depth(
                depth_map, det['bbox'], frame_width, frame_height
            )
            
            if depth_value > 0.75:
                distance = "very close"
            elif depth_value > 0.55:
                distance = "close"
            elif depth_value > 0.35:
                distance = "nearby"
            else:
                distance = "far"
            
            spatial_objects.append(SpatialObject(
                class_name=det['class'],
                confidence=det['confidence'],
                position=position,
                distance=distance,
                depth_value=depth_value,
                is_danger=det['is_danger'],
                is_navigation=det.get('is_navigation', False),
                bbox=det['bbox'],
                center=det['center'],
            ))
        
        # Sort by proximity (closest first)
        spatial_objects.sort(key=lambda obj: -obj.depth_value)
        
        return spatial_objects

    def _get_object_depth(self, depth_map: np.ndarray,
                          bbox: tuple,
                          frame_w: int, frame_h: int) -> float:
        """Get average depth within object bounding box"""
        x1, y1, x2, y2 = bbox
        
        # Clamp to frame bounds
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))
        
        if x2 <= x1 or y2 <= y1:
            return 0.5
        
        # Sample center region of bbox (more stable than full bbox)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        margin_x = max(5, (x2 - x1) // 4)
        margin_y = max(5, (y2 - y1) // 4)
        
        region = depth_map[
            max(0, cy - margin_y):min(frame_h, cy + margin_y),
            max(0, cx - margin_x):min(frame_w, cx + margin_x)
        ]
        
        if region.size == 0:
            return 0.5
        
        return float(np.mean(region))

    def describe_scene(self, spatial_objects: List[SpatialObject]) -> str:
        """
        Generate a natural language scene description.
        Used for the 'slow lane' scene summary.
        """
        if not spatial_objects:
            return "The path appears clear."
        
        parts = []
        for obj in spatial_objects[:5]:  # Max 5 objects in description
            parts.append(f"{obj.class_name} {obj.position}, {obj.distance}")
        
        return "I can see: " + ". ".join(parts) + "."


# ═══════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("Testing SpatialMapper...")
    
    mapper = SpatialMapper()
    
    # Simulate detections
    fake_detections = [
        {
            'class': 'person', 'confidence': 0.9,
            'bbox': (100, 100, 200, 400), 'center': (150, 250),
            'area_ratio': 0.1, 'is_danger': False, 'is_navigation': True,
        },
        {
            'class': 'car', 'confidence': 0.85,
            'bbox': (400, 200, 600, 400), 'center': (500, 300),
            'area_ratio': 0.15, 'is_danger': True, 'is_navigation': False,
        },
    ]
    
    # Simulate depth map
    fake_depth = np.random.rand(480, 640).astype(np.float32)
    fake_depth[100:400, 100:200] = 0.8  # Person is close
    fake_depth[200:400, 400:600] = 0.6  # Car is medium distance
    
    results = mapper.map_objects(fake_detections, fake_depth, 640, 480)
    
    print(f"\nMapped {len(results)} objects:")
    for obj in results:
        print(f"  {obj.class_name}: {obj.position}, {obj.distance} "
              f"(depth={obj.depth_value:.2f}, danger={obj.is_danger})")
    
    print(f"\nScene: {mapper.describe_scene(results)}")
    print("\n✅ Spatial Mapper test complete!")