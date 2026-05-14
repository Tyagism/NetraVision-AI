# scripts/download_piper_voices.py
"""Download Piper TTS voice models for NetraVision AI."""

import urllib.request
import os

VOICES = {
    "en_male": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    },
}

def download():
    output_dir = "models/piper"
    os.makedirs(output_dir, exist_ok=True)

    for name, info in VOICES.items():
        for key in ["url", "config"]:
            url = info[key]
            filename = os.path.basename(url)
            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  ✅ {filename} ({size:.1f} MB)")
                continue

            try:
                print(f"  ⬇️  Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  ✅ {filename} ({size:.1f} MB)")
            except Exception as e:
                print(f"  ⚠️  Failed to download {filename}: {e}")
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                continue

    print("\n✅ Piper voices ready!")

if __name__ == "__main__":
    print("=" * 50)
    print("👁️ NetraVision AI — Downloading Piper Voices")
    print("=" * 50)
    download()