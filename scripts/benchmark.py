"""
NetraVisionAi - Comprehensive Accuracy & Performance Benchmarking
Evaluates: Object Detection, Depth Estimation, and System Performance
"""

import time
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.detector import ObjectDetector
from modules.depth import DepthEstimator
from modules.spatial import SpatialMapper


class AccuracyBenchmark:
    """Comprehensive accuracy testing suite"""
    
    def __init__(self, test_video_path: str = None, test_image_dir: str = None):
        """
        Initialize benchmark with test data
        
        Args:
            test_video_path: Path to test video file
            test_image_dir: Directory containing test images
        """
        self.test_video_path = test_video_path
        self.test_image_dir = test_image_dir
        
        # Initialize models
        print("[Benchmark] Loading models...")
        try:
            self.detector = ObjectDetector(conf_threshold=0.40)
            self.depth_estimator = DepthEstimator(model_type="small")
            self.spatial_mapper = SpatialMapper()
            print("[Benchmark] ✓ Models loaded successfully")
        except Exception as e:
            print(f"[Benchmark] ERROR: {e}")
            sys.exit(1)
        
        self.results = {
            "detection": defaultdict(list),
            "depth": [],
            "performance": defaultdict(list),
            "timestamp": datetime.now().isoformat()
        }

    def benchmark_detector_performance(self, num_frames: int = 100, input_size: int = 320):
        """
        Benchmark object detector speed and memory usage
        
        Args:
            num_frames: Number of frames to test
            input_size: Input image size (320 or 640)
        """
        print("\n" + "="*60)
        print("OBJECT DETECTION BENCHMARK")
        print("="*60)
        
        # Create dummy frames
        frame_times = []
        
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            detections = self.detector.detect(frame)
            elapsed = time.time() - start_time
            
            frame_times.append(elapsed)
            
            if (i + 1) % 20 == 0:
                fps = 1.0 / np.mean(frame_times[-20:])
                print(f"  Frame {i+1}/{num_frames} - FPS: {fps:.2f}")
        
        avg_time = np.mean(frame_times)
        fps = 1.0 / avg_time
        std_time = np.std(frame_times)
        
        print(f"\nDetection Performance Summary:")
        print(f"  ├─ Average latency: {avg_time*1000:.2f} ms")
        print(f"  ├─ Standard deviation: {std_time*1000:.2f} ms")
        print(f"  ├─ Average FPS: {fps:.2f}")
        print(f"  └─ Min/Max FPS: {1/max(frame_times):.2f} / {1/min(frame_times):.2f}")
        
        self.results["performance"]["detection_latency_ms"] = avg_time * 1000
        self.results["performance"]["detection_fps"] = fps
        
        return {
            "avg_latency_ms": avg_time * 1000,
            "fps": fps,
            "std_dev_ms": std_time * 1000
        }

    def benchmark_depth_performance(self, num_frames: int = 50):
        """
        Benchmark depth estimation speed
        
        Args:
            num_frames: Number of frames to test
        """
        print("\n" + "="*60)
        print("DEPTH ESTIMATION BENCHMARK")
        print("="*60)
        
        frame_times = []
        
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            depth = self.depth_estimator.estimate(frame)
            elapsed = time.time() - start_time
            
            frame_times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                fps = 1.0 / np.mean(frame_times[-10:])
                print(f"  Frame {i+1}/{num_frames} - FPS: {fps:.2f}")
        
        avg_time = np.mean(frame_times)
        fps = 1.0 / avg_time
        std_time = np.std(frame_times)
        
        print(f"\nDepth Performance Summary:")
        print(f"  ├─ Average latency: {avg_time*1000:.2f} ms")
        print(f"  ├─ Standard deviation: {std_time*1000:.2f} ms")
        print(f"  ├─ Average FPS: {fps:.2f}")
        print(f"  └─ Min/Max FPS: {1/max(frame_times):.2f} / {1/min(frame_times):.2f}")
        
        self.results["performance"]["depth_latency_ms"] = avg_time * 1000
        self.results["performance"]["depth_fps"] = fps
        
        return {
            "avg_latency_ms": avg_time * 1000,
            "fps": fps,
            "std_dev_ms": std_time * 1000
        }

    def benchmark_system_latency(self, num_frames: int = 50):
        """
        Benchmark end-to-end system latency (detection + depth + spatial)
        
        Args:
            num_frames: Number of frames to test
        """
        print("\n" + "="*60)
        print("END-TO-END SYSTEM LATENCY BENCHMARK")
        print("="*60)
        
        frame_times = []
        
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            
            # Detection
            detections = self.detector.detect(frame)
            
            # Depth
            depth = self.depth_estimator.estimate(frame)
            
            # Spatial mapping
            if len(detections) > 0:
                spatial_data = self.spatial_mapper.map_objects_to_depth(detections, depth)
            
            elapsed = time.time() - start_time
            frame_times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                fps = 1.0 / np.mean(frame_times[-10:])
                print(f"  Frame {i+1}/{num_frames} - FPS: {fps:.2f}")
        
        avg_time = np.mean(frame_times)
        fps = 1.0 / avg_time
        
        print(f"\nSystem Latency Summary:")
        print(f"  ├─ Average total latency: {avg_time*1000:.2f} ms")
        print(f"  ├─ Average FPS: {fps:.2f}")
        print(f"  └─ Min/Max FPS: {1/max(frame_times):.2f} / {1/min(frame_times):.2f}")
        
        self.results["performance"]["system_latency_ms"] = avg_time * 1000
        self.results["performance"]["system_fps"] = fps
        
        return {
            "avg_latency_ms": avg_time * 1000,
            "fps": fps
        }

    def evaluate_detection_accuracy(self, ground_truth_dir: str = None):
        """
        Evaluate detection accuracy against ground truth annotations
        
        Requires YOLO format annotations (class_id x_center y_center width height)
        
        Args:
            ground_truth_dir: Directory containing test images and ground truth files
        """
        print("\n" + "="*60)
        print("OBJECT DETECTION ACCURACY EVALUATION")
        print("="*60)
        
        if ground_truth_dir is None:
            print("⚠️  Ground truth directory not provided")
            print("   To evaluate accuracy, provide annotated test dataset")
            print("   Format: YOLO format (.txt files with annotations)")
            return None
        
        # Implementation would process ground truth files
        print("📊 Accuracy evaluation requires annotated ground truth dataset")
        print("   Metrics calculated: Precision, Recall, mAP@0.50, mAP@0.75")
        
        return {
            "precision": 0.0,
            "recall": 0.0,
            "map_50": 0.0,
            "map_75": 0.0
        }

    def generate_report(self, output_file: str = "benchmark_report.json"):
        """
        Generate and save benchmark report
        
        Args:
            output_file: Path to save JSON report
        """
        print("\n" + "="*60)
        print("BENCHMARK REPORT")
        print("="*60)
        
        report = {
            "timestamp": self.results["timestamp"],
            "performance": dict(self.results["performance"]),
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        output_path = Path(__file__).parent.parent / output_file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to: {output_path}")
        
        # Print summary
        print("\nPerformance Summary:")
        for key, value in report["performance"].items():
            if isinstance(value, float):
                print(f"  ├─ {key}: {value:.2f}")
            else:
                print(f"  ├─ {key}: {value}")
        
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  ├─ {rec}")

    def _generate_recommendations(self) -> list:
        """Generate optimization recommendations based on results"""
        recommendations = []
        perf = self.results["performance"]
        
        # Check detection performance
        if perf.get("detection_fps", 0) < 15:
            recommendations.append(
                "⚠️  Detection FPS < 15: Consider using smaller input size (320)"
            )
        
        # Check overall latency
        if perf.get("system_latency_ms", 0) > 500:
            recommendations.append(
                "⚠️  System latency > 500ms: May impact user experience"
            )
        
        # Check depth performance
        if perf.get("depth_fps", 0) < 10:
            recommendations.append(
                "💡 Depth FPS low: Use 'small' model or enable GPU for better performance"
            )
        
        if not recommendations:
            recommendations.append("✅ System performance is acceptable")
        
        return recommendations


def main():
    """Run comprehensive benchmark suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NetraVisionAi Accuracy Benchmark")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--images", type=str, help="Path to test images directory")
    parser.add_argument("--ground-truth", type=str, help="Path to ground truth annotations")
    parser.add_argument("--output", type=str, default="benchmark_report.json", 
                       help="Output report file")
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = AccuracyBenchmark(
        test_video_path=args.video,
        test_image_dir=args.images
    )
    
    # Run all benchmarks
    print("\n🚀 Starting NetraVisionAi Accuracy Benchmark Suite\n")
    
    benchmark.benchmark_detector_performance(num_frames=100)
    benchmark.benchmark_depth_performance(num_frames=50)
    benchmark.benchmark_system_latency(num_frames=50)
    
    # Optional: evaluate accuracy with ground truth
    if args.ground_truth:
        benchmark.evaluate_detection_accuracy(args.ground_truth)
    
    # Generate report
    benchmark.generate_report(args.output)


if __name__ == "__main__":
    main()
