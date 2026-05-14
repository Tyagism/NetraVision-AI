"""
NetraVisionAi — Download all required AI models
Run this ONCE before first use
"""

import os
import sys
import urllib.request
import hashlib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def download_file(url: str, filepath: str, description: str):
    """Download a file with progress indicator"""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✅ {description} already exists ({size_mb:.1f} MB)")
        return True
    
    print(f"  ⬇️  Downloading {description}...")
    print(f"     URL: {url}")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = min(100, int(count * block_size * 100 / total_size))
            mb_done = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r     Progress: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print()  # New line after progress
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✅ Downloaded: {description} ({size_mb:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Failed to download {description}: {e}")
        return False


def export_yolo():
    """Export YOLOv8-nano to ONNX format"""
    onnx_path = os.path.join(MODELS_DIR, "yolov8n.onnx")
    
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ✅ YOLOv8-nano ONNX already exists ({size_mb:.1f} MB)")
        return True
    
    print("  ⬇️  Downloading & exporting YOLOv8-nano to ONNX...")
    print("     (This downloads ~6MB model + exports to ONNX)")
    
    try:
        from ultralytics import YOLO
        
        # Download YOLOv8-nano
        model = YOLO("yolov8n.pt")
        
        # Export to ONNX (optimized for our use)
        model.export(
            format="onnx",
            imgsz=320,           # Smaller input = faster inference
            simplify=True,       # Simplify ONNX graph
            opset=13,
        )
        
        # Move exported file to models directory
        exported = "yolov8n.onnx"
        if os.path.exists(exported):
            os.rename(exported, onnx_path)
        
        # Clean up .pt file
        if os.path.exists("yolov8n.pt"):
            os.rename("yolov8n.pt", os.path.join(MODELS_DIR, "yolov8n.pt"))
        
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ✅ YOLOv8-nano exported ({size_mb:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to export YOLO: {e}")
        return False


def download_midas():
    """Download MiDaS depth estimation model"""
    url = "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt"
    filepath = os.path.join(MODELS_DIR, "dpt_hybrid_384.pt")
    return download_file(url, filepath, "MiDaS v3 Hybrid (Depth Estimation)")


def main():
    print("=" * 60)
    print("  🦯 NetraVisionAi — Model Downloader")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    results = {}
    
    print("\n📦 [1/2] YOLOv8-nano (Object Detection)")
    results["yolo"] = export_yolo()
    
    print("\n📦 [2/2] MiDaS Small (Depth Estimation)")
    results["midas"] = download_midas()
    
    # Summary
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY:")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ Ready" if success else "❌ Failed"
        print(f"  {name}: {status}")
    
    # List model files
    print(f"\n  Models directory: {MODELS_DIR}")
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            path = os.path.join(MODELS_DIR, f)
            if os.path.isfile(path):
                size = os.path.getsize(path) / (1024 * 1024)
                print(f"    📄 {f} ({size:.1f} MB)")
    
    all_good = all(results.values())
    if all_good:
        print("\n  🎉 All models downloaded successfully!")
        print("  You can now run: python main.py")
    else:
        print("\n  ⚠️  Some downloads failed. Check errors above.")
    
    return all_good


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)