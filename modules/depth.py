"""
NetraVisionAi — Depth Estimation
Smart model selection + automatic GPU/CPU detection

Models:
  "small"  → MiDaS v2.1 Small (FAST: 15-30 FPS CPU, 60+ FPS GPU)
  "hybrid" → MiDaS v3 DPT-Hybrid (QUALITY: 0.5-2 FPS CPU, 15-30 FPS GPU)

Usage:
  python modules/depth.py                    # Auto-select fastest
  python modules/depth.py --model small      # Force fast model
  python modules/depth.py --model hybrid     # Force quality model
  python modules/depth.py --device cpu       # Force CPU
  python modules/depth.py --device cuda      # Force GPU
"""

import numpy as np
import cv2
import time
import os
import glob
import torch
import torch.nn as nn
from typing import Optional, Tuple


class DepthEstimator:

    MODELS = {
        "small": {
            "hub_name": "MiDaS_small",
            "transform_name": "small_transform",
            "input_size": 256,
            "pt_filenames": ["midas_v21_small_256.pt", "midas_small.pt"],
            "description": "MiDaS Small v2.1",
            "speed": "FAST (15-30 FPS CPU)",
        },
        "hybrid": {
            "hub_name": "DPT_Hybrid",
            "transform_name": "dpt_transform",
            "input_size": 384,
            "pt_filenames": ["dpt_hybrid_384.pt", "dpt_hybrid.pt"],
            "description": "MiDaS v3 DPT-Hybrid",
            "speed": "QUALITY (0.5-2 FPS CPU, 15-30 FPS GPU)",
        },
    }

    def __init__(self, model_type: str = "small", device: str = "auto"):
        """
        Args:
            model_type: "small" (fast) or "hybrid" (quality)
            device: "auto", "cpu", "cuda"
        """
        if model_type not in self.MODELS:
            model_type = "small"

        self.model_type = model_type
        self.model_info = self.MODELS[model_type]
        self.model = None
        self.transform = None

        # ─── Device Selection ───
        self.device = self._select_device(device)

        print(f"[Depth] {'='*50}")
        print(f"[Depth] Model:  {self.model_info['description']}")
        print(f"[Depth] Speed:  {self.model_info['speed']}")
        print(f"[Depth] Input:  {self.model_info['input_size']}x{self.model_info['input_size']}")
        print(f"[Depth] Device: {self.device}")
        if self.device.type == "cuda":
            print(f"[Depth] GPU:    {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Depth] VRAM:   {gpu_mem:.1f} GB")
        print(f"[Depth] {'='*50}")

        # ─── Load ───
        self._load_model()

        # ─── Warmup ───
        self._warmup()

    def _select_device(self, device_str: str) -> torch.device:
        """Select best available device"""

        if device_str == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                print("[Depth] ⚠️  CUDA requested but not available!")
                print("[Depth]     Install NVIDIA PyTorch:")
                print("[Depth]     pip uninstall torch torchvision torchaudio -y")
                print("[Depth]     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                print("[Depth]     Falling back to CPU...")
                return torch.device("cpu")

        elif device_str == "cpu":
            return torch.device("cpu")

        else:  # "auto"
            if torch.cuda.is_available():
                print("[Depth] 🟢 NVIDIA GPU detected — using CUDA!")
                return torch.device("cuda")
            else:
                print("[Depth] ℹ️  No GPU detected — using CPU")
                print("[Depth]     To enable GPU:")
                print("[Depth]     pip uninstall torch torchvision torchaudio -y")
                print("[Depth]     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                return torch.device("cpu")

    def _find_local_pt(self) -> Optional[str]:
        """Find local .pt weights file"""
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models"
        )

        for filename in self.model_info["pt_filenames"]:
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"[Depth] Found local weights: {filename} ({size_mb:.1f} MB)")
                return path

        return None

    def _load_model(self):
        """Load model — tries local .pt first, then torch.hub"""

        hub_name = self.model_info["hub_name"]
        transform_name = self.model_info["transform_name"]

        # ─── Try LOCAL .pt file first (no download!) ───
        pt_path = self._find_local_pt()

        if pt_path:
            try:
                print(f"[Depth] Loading architecture (pretrained=False, no weight download)...")

                self.model = torch.hub.load(
                    "intel-isl/MiDaS",
                    hub_name,
                    pretrained=False,
                    trust_repo=True,
                )

                print(f"[Depth] Loading YOUR local weights...")
                state_dict = torch.load(
                    pt_path,
                    map_location=self.device,
                    weights_only=False,
                )
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()

                print(f"[Depth] ✅ Loaded from local .pt (NO re-download!)")
                self._load_transform(transform_name)
                return

            except Exception as e:
                print(f"[Depth] ⚠️  Local load failed: {e}")
                print(f"[Depth]     Trying torch.hub with pretrained...")

        # ─── Fallback: torch.hub with pretrained=True ───
        print(f"[Depth] Loading {hub_name} via torch.hub...")
        if self.model_type == "small":
            print(f"[Depth] ℹ️  MiDaS Small = ~50MB one-time download")
        else:
            print(f"[Depth] ℹ️  DPT-Hybrid = ~480MB one-time download")

        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            hub_name,
            trust_repo=True,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[Depth] ✅ Model loaded")
        self._load_transform(transform_name)

    def _load_transform(self, transform_name: str):
        """Load image preprocessing transform"""
        try:
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True,
            )
            self.transform = getattr(midas_transforms, transform_name)
        except Exception:
            self.transform = None

    def _warmup(self):
        """Run warmup inference to initialize everything"""
        print("[Depth] Warming up...")
        inp_size = self.model_info["input_size"]
        dummy = torch.randn(1, 3, inp_size, inp_size).to(self.device)

        # Run 3 warmup iterations (GPU especially needs this)
        times = []
        with torch.no_grad():
            for i in range(3):
                start = time.time()
                self.model(dummy)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                ms = (time.time() - start) * 1000
                times.append(ms)

        avg_ms = np.mean(times)
        best_ms = np.min(times)
        print(f"[Depth] ✅ Warmup done: avg={avg_ms:.0f}ms, best={best_ms:.0f}ms")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from BGR frame.

        Returns:
            depth_map: same size as frame, float32, 0→1
                       HIGHER = CLOSER to camera
        """
        orig_h, orig_w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess
        if self.transform is not None:
            input_batch = self.transform(img_rgb).to(self.device)
        else:
            input_batch = self._manual_preprocess(img_rgb)

        # Inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Normalize
        depth = prediction.cpu().numpy()
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-8:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.full_like(depth, 0.5)

        return depth.astype(np.float32)

    def _manual_preprocess(self, img_rgb):
        inp_size = self.model_info["input_size"]
        img = cv2.resize(img_rgb, (inp_size, inp_size))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    # ═══════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════

    def get_depth_at_point(self, depth_map, x, y, radius=15):
        h, w = depth_map.shape[:2]
        y1, y2 = max(0, y-radius), min(h, y+radius)
        x1, x2 = max(0, x-radius), min(w, x+radius)
        region = depth_map[y1:y2, x1:x2]
        return float(np.mean(region)) if region.size > 0 else 0.5

    def get_depth_at_bbox(self, depth_map, bbox):
        x1, y1, x2, y2 = bbox
        return self.get_depth_at_point(depth_map, (x1+x2)//2, (y1+y2)//2, 20)

    def get_distance_label(self, depth_value):
        if depth_value > 0.75: return "very close"
        elif depth_value > 0.55: return "close"
        elif depth_value > 0.35: return "nearby"
        elif depth_value > 0.20: return "few meters"
        else: return "far"

    def get_depth_colormap(self, depth_map):
        return cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

    def get_obstacle_warning(self, depth_map, threshold=0.70):
        h, w = depth_map.shape[:2]
        mx, my = int(w*0.3), int(h*0.3)
        center = depth_map[my:h-my, mx:w-mx]
        if center.size == 0: return None
        max_d = float(np.max(center))
        if max_d > threshold and float(np.mean(center > threshold)) > 0.3:
            return f"Obstacle ahead, {self.get_distance_label(max_d)}"
        return None


# ═══════════════════════════════════════════════════════
# SYSTEM INFO PRINTER
# ═══════════════════════════════════════════════════════
def print_system_info():
    """Print GPU/CPU info to help user decide"""
    print("=" * 60)
    print("  SYSTEM INFO")
    print("=" * 60)
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA version:   {torch.version.cuda}")
        print(f"  GPU name:       {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU memory:     {gpu_mem:.1f} GB")
        print(f"\n  [OK] GPU available! Both models will be fast.")
        print(f"     small:  60+ FPS on GPU")
        print(f"     hybrid: 15-30 FPS on GPU")
    else:
        print(f"  CUDA version:   Not installed")
        print(f"  GPU:            Not available for PyTorch")
        print(f"\n  [!] Running on CPU only.")
        print(f"     small:  15-25 FPS on CPU <- RECOMMENDED")
        print(f"     hybrid: 0.5-2 FPS on CPU <- TOO SLOW for real-time")
        print(f"\n  [TIP] To enable GPU (NVIDIA only):")
        print(f"     pip uninstall torch torchvision torchaudio -y")
        print(f"     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    # Check what models are downloaded
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models"
    )
    print(f"\n  Models directory: {models_dir}:")
    if os.path.exists(models_dir):
        found = False
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".pt") or f.endswith(".onnx"):
                sz = os.path.getsize(os.path.join(models_dir, f)) / (1024*1024)
                marker = "[PT]" if f.endswith('.pt') else "[ONNX]"
                print(f"     {marker} {f} ({sz:.1f} MB)")
                found = True
        if not found:
            print(f"     (none found)")
    print("=" * 60)


# ═══════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NetraVisionAi Depth Estimation Test")
    parser.add_argument(
        "--model", type=str, default="small",
        choices=["small", "hybrid"],
        help="small=FAST(15-30fps) | hybrid=QUALITY(needs GPU for real-time)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto=best available | cpu=force CPU | cuda=force GPU"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Print system info and exit"
    )
    args = parser.parse_args()

    # Just print info?
    if args.info:
        print_system_info()
        exit(0)

    print("=" * 60)
    print("  NetraVisionAi Depth Test")
    print("=" * 60)

    # Print system info first
    print_system_info()

    print(f"\n  Selected: --model {args.model} --device {args.device}\n")

    # ─── Init ───
    try:
        estimator = DepthEstimator(model_type=args.model, device=args.device)
    except Exception as e:
        print(f"\n❌ {e}")
        print("\nFix: pip install timm==0.9.16")
        exit(1)

    # ─── Camera ───
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit(1)

    print(f"\n  Camera ready!")
    print("   Bright = CLOSE | Dark = FAR | Press 'q' to quit\n")

    fps_list = []
    ms_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        depth_map = estimator.estimate(frame)
        if estimator.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - start) * 1000
        ms_list.append(ms)
        fps_list.append(1000.0 / max(ms, 1))

        depth_vis = estimator.get_depth_colormap(depth_map)
        h, w = frame.shape[:2]

        c = estimator.get_depth_at_point(depth_map, w//2, h//2)
        l = estimator.get_depth_at_point(depth_map, w//4, h//2)
        r = estimator.get_depth_at_point(depth_map, 3*w//4, h//2)
        warning = estimator.get_obstacle_warning(depth_map)

        avg_fps = np.mean(fps_list[-30:])
        avg_ms = np.mean(ms_list[-30:])

        # Draw info
        dev_str = "GPU" if estimator.device.type == "cuda" else "CPU"
        lines = [
            f"{estimator.model_info['description']} ({dev_str}) | {avg_fps:.1f} FPS ({avg_ms:.0f}ms)",
            f"Left:   {estimator.get_distance_label(l)} ({l:.2f})",
            f"Center: {estimator.get_distance_label(c)} ({c:.2f})",
            f"Right:  {estimator.get_distance_label(r)} ({r:.2f})",
        ]
        if warning:
            lines.append(f"[!] {warning} [!]")

        for i, txt in enumerate(lines):
            fg = (0,0,255) if txt.startswith("!!") else (255,255,255)
            cv2.putText(depth_vis, txt, (10, 25+i*28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 3)
            cv2.putText(depth_vis, txt, (10, 25+i*28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, fg, 1)

        for px, clr in [(w//4,(255,255,0)), (w//2,(0,255,0)), (3*w//4,(255,255,0))]:
            cv2.drawMarker(depth_vis, (px,h//2), clr, cv2.MARKER_CROSS, 20, 2)
            cv2.drawMarker(frame, (px,h//2), clr, cv2.MARKER_CROSS, 20, 2)

        combined = np.hstack([frame, depth_vis])
        cv2.imshow("NetraVisionAi Depth", combined)

        w_txt = f" !! {warning}" if warning else ""
        print(
            f"\r  {avg_fps:.1f} FPS ({avg_ms:.0f}ms) [{dev_str}] | "
            f"L:{estimator.get_distance_label(l):>10s} "
            f"C:{estimator.get_distance_label(c):>10s} "
            f"R:{estimator.get_distance_label(r):>10s}"
            f"{w_txt}          ",
            end="", flush=True
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Model:   {estimator.model_info['description']}")
    print(f"  Device:  {estimator.device} {'(' + torch.cuda.get_device_name(0) + ')' if estimator.device.type == 'cuda' else ''}")
    print(f"  FPS:     {np.mean(fps_list):.1f} avg | {np.max(fps_list):.1f} max")
    print(f"  Latency: {np.mean(ms_list):.0f}ms avg | {np.min(ms_list):.0f}ms best")
    print(f"  Frames:  {len(fps_list)} total")

    # Recommendation
    print(f"\n  RECOMMENDATION:")
    if estimator.device.type == "cpu" and args.model == "hybrid":
        print(f"     DPT-Hybrid is too slow on CPU!")
        print(f"     Use: python modules/depth.py --model small")
        print(f"     Or install CUDA PyTorch for GPU acceleration")
    elif estimator.device.type == "cpu" and avg_fps < 10:
        print(f"     Consider installing CUDA PyTorch for better FPS")
    elif avg_fps >= 15:
        print(f"     [OK] Great performance! Ready for real-time use.")

    print(f"\n  Done!")