"""
Scene Description using lightweight Vision Language Model
Options: Moondream2, SmolVLM-256M
"""

import numpy as np
import cv2
import time
from PIL import Image
import torch


class SceneDescriber:
    def __init__(self, model_name="moondream2", device="auto"):
        """
        Lightweight Vision Language Model for scene description.
        
        Args:
            model_name: "moondream2" (recommended) - best balance of quality/size
            device: "auto", "cpu", or "cuda"
        """
        self.model_name = "moondream2"  # Force moondream2
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = self._select_device(device)
        self._load_model()

    def _select_device(self, device_str: str):
        """Select best available device"""
        if device_str == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("[SceneDescriber] CUDA not available, using CPU")
                return "cpu"
        elif device_str == "cpu":
            return "cpu"
        else:  # "auto"
            return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Load moondream2 model"""
        self._load_moondream()

    def _load_moondream(self):
        """Load Moondream2 — best quality for size"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print("[SceneDescriber] Loading Moondream2...")
            model_id = "vikhyatk/moondream2"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                device_map=self.device
            )
            self.model.eval()
            print(f"[SceneDescriber] Moondream2 loaded on {self.device}")
        except Exception as e:
            print(f"[SceneDescriber] Error loading Moondream2: {e}")
            raise

    def _load_smolvlm(self):
        """Load SmolVLM — NOT AVAILABLE due to dependency conflicts"""
        raise NotImplementedError("SmolVLM is not available. Use moondream2 instead.")

    def describe(self, frame) -> str:
        """Generate scene description from BGR frame"""
        if self.model is None:
            return "Model not loaded"
        
        try:
            # Convert BGR to RGB
            if isinstance(frame, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                image = frame
            
            return self._describe_moondream(image)
        except Exception as e:
            print(f"[SceneDescriber] Error: {e}")
            return f"Error: {str(e)}"

    def _describe_moondream(self, image) -> str:
        """Generate description using Moondream2"""
        try:
            prompt = (
                "Describe this scene in one short sentence for a blind person. "
                "Focus on obstacles, layout, and navigation-relevant details."
            )
            
            # Encode image
            enc_image = self.model.encode_image(image)
            
            # Generate answer
            with torch.no_grad():
                answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
            
            return f"Scene: {answer}"
        except Exception as e:
            print(f"[SceneDescriber] Moondream error: {e}")
            return f"Error: {str(e)}"

    def _describe_smolvlm(self, image) -> str:
        """SmolVLM is not available"""
        raise NotImplementedError("SmolVLM not available. Using moondream2 instead.")


# ═══════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scene Describer Test")
    parser.add_argument("--model", type=str, default="moondream2", 
                        choices=["moondream2", "smolvlm"],
                        help="Which VLM to use")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Scene Describer Test")
    print("=" * 60)
    
    # Load model
    try:
        describer = SceneDescriber(model_name=args.model, device=args.device)
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        print("\nTry installing transformers:")
        print("  pip install transformers pillow")
        exit(1)
    
    # Try camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n[INFO] No camera found. Creating test image...")
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        print("\n[INFO] Camera opened. Press 'q' to quit.\n")
        ret, test_frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read camera frame")
            exit(1)
    
    # Describe
    print("[*] Analyzing scene...")
    start = time.time()
    description = describer.describe(test_frame)
    elapsed = time.time() - start
    
    print(f"\n{description}")
    print(f"\nTime: {elapsed:.2f}s")
    
    if cap.isOpened():
        cap.release()
    
    print("\n[OK] Done!")
