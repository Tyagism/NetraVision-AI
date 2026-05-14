# main.py
"""
NetraVisionAi — Main Entry Point
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="🦯 NetraVisionAi — Vision Assistant for the Visually Impaired"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (0=default webcam)")
    parser.add_argument("--ip-camera", type=str, default=None,
                        help="IP camera URL")
    parser.add_argument("--model", type=str, default="small",
                        choices=["small", "hybrid"],
                        help="Depth model type")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Processing device")
    parser.add_argument("--language", type=str, default="en",
                        help="Language: en, hi, bn, te, mr, ta, ur, gu, kn, ml, pa, or, as, etc.")
    parser.add_argument("--speed", type=int, default=185,
                        help="TTS speed (words per minute)")
    parser.add_argument("--no-video", action="store_true",
                        help="Run without display window")
    parser.add_argument("--skip-frames", type=int, default=3,
                        help="Process every Nth frame")

    args = parser.parse_args()

    # ─── Banner ───
    camera_display = str(args.camera) if not args.ip_camera else args.ip_camera

    print()
    print("╔═══════════════════════════════════════════════════╗")
    print("║                                                   ║")
    print("║   🦯  NetraVisionAi — Vision Assistant               ║")
    print("║       For the Visually Impaired                   ║")
    print("║                                                   ║")
    print("╠═══════════════════════════════════════════════════╣")
    print(f"║   Camera:     {camera_display:<35s} ║")
    print(f"║   Depth:      {args.model:<35s} ║")
    print(f"║   Device:     {args.device:<35s} ║")
    print(f"║   Language:   {args.language:<35s} ║")
    print(f"║   Video:      {'OFF' if args.no_video else 'ON':<35s} ║")
    print("╚═══════════════════════════════════════════════════╝")
    print()

    # ─── Check models ───
    if not os.path.exists("models/yolov8n.onnx"):
        print("❌ YOLOv8-nano model not found!")
        print("   Run: python scripts/download_models.py")
        return 1

    # ─── Camera source ───
    camera_source = args.ip_camera if args.ip_camera else args.camera

    # ─── Create and start ───
    from netra_vision_ai_core import NetraVisionAi

    try:
        assistant = NetraVisionAi(
            camera_source=0,
            depth_model=args.model,
            depth_device=args.device,
            tts_language=args.language,
            tts_speed=args.speed,
            show_video=not args.no_video,
            process_every_n=args.skip_frames,
        )
        assistant.start()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())