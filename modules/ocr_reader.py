# modules/ocr_reader.py
"""
NetraVisionAi — OCR Reader Module
Reads text from signs, labels, books, screens in the camera view.
Uses EasyOCR (lightweight, supports Hindi + English).
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextDetection:
    """A detected text region."""
    text: str
    confidence: float
    bbox: list          # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    center: tuple


class OCRReader:
    """
    Reads text in camera frames using EasyOCR.
    Designed for the MEDIUM LANE (every 5th frame).
    """

    def __init__(self, languages=None, gpu=False,
                 confidence_threshold=0.4, min_text_length=2):
        """
        Args:
            languages: List of language codes ['en', 'hi']
            gpu: Use GPU for OCR
            confidence_threshold: Min confidence to keep text
            min_text_length: Min characters to report
        """
        if languages is None:
            languages = ['en']

        self.languages = languages
        self.gpu = gpu
        self.conf_threshold = confidence_threshold
        self.min_text_length = min_text_length
        self.reader = None
        self.last_inference_time = 0.0

        self._load_reader()

    def _load_reader(self):
        """Load EasyOCR reader."""
        try:
            import easyocr
            logger.info(f"Loading EasyOCR with languages: {self.languages}...")
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False
            )
            logger.info(f"✅ OCR Reader loaded: {self.languages}")
        except ImportError:
            logger.error("❌ EasyOCR not installed!")
            logger.error("   Install: pip install easyocr")
            self.reader = None

    def read(self, frame) -> List[TextDetection]:
        """
        Read text from a frame.

        Args:
            frame: BGR numpy array

        Returns:
            List of TextDetection objects
        """
        if self.reader is None or frame is None:
            return []

        start_time = time.time()

        try:
            # Run OCR
            results = self.reader.readtext(frame)

            detections = []
            for (bbox, text, confidence) in results:
                # Filter by confidence and length
                if confidence < self.conf_threshold:
                    continue
                if len(text.strip()) < self.min_text_length:
                    continue

                # Calculate center
                pts = np.array(bbox)
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))

                det = TextDetection(
                    text=text.strip(),
                    confidence=confidence,
                    bbox=bbox,
                    center=(cx, cy)
                )
                detections.append(det)

            self.last_inference_time = (time.time() - start_time) * 1000

            if detections:
                logger.debug(f"📝 OCR found {len(detections)} text regions "
                             f"in {self.last_inference_time:.0f}ms")

            return detections

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []

    def get_all_text(self, frame) -> str:
        """
        Get all readable text from frame as a single string.
        Useful for "read everything" voice command.
        """
        detections = self.read(frame)
        if not detections:
            return ""

        # Sort by vertical position (top to bottom), then left to right
        detections.sort(key=lambda d: (d.center[1], d.center[0]))

        texts = [d.text for d in detections]
        return " ".join(texts)

    def draw_text_detections(self, frame, detections: List[TextDetection]):
        """Draw OCR results on frame."""
        for det in detections:
            pts = np.array(det.bbox, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 255), 2)

            label = f'"{det.text}" ({det.confidence:.0%})'
            cv2.putText(frame, label,
                        (det.center[0], det.center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return frame


# ═══════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    import os

    logging.basicConfig(level=logging.INFO)
    print("=" * 50)
    print("📝 NetraVisionAi OCR Reader Test")
    print("=" * 50)
    print("Point camera at text (book, sign, screen)")
    print("Press 'r' to read all text aloud")
    print("Press 'q' to quit\n")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.camera import Camera

    # Try to import TTS for reading aloud
    try:
        from modules.tts_engine import TTSEngine
        tts = TTSEngine(engine="espeak", language="en", speed=160)
        tts.start()
        has_tts = True
    except:
        has_tts = False

    ocr = OCRReader(languages=['en'], gpu=False)
    cam = Camera(source=0, width=640, height=480)
    cam.start()

    frame_count = 0

    try:
        while cam.is_opened():
            ret, frame = cam.read()
            if not ret:
                continue

            frame_count += 1

            # Run OCR every 15 frames (medium lane)
            if frame_count % 15 == 0:
                detections = ocr.read(frame)
                if detections:
                    for det in detections:
                        print(f'  📝 "{det.text}" (confidence: {det.confidence:.0%})')
                    frame = ocr.draw_text_detections(frame, detections)

            # Show info
            cv2.putText(frame, f"FPS: {cam.get_fps()} | OCR: {ocr.last_inference_time:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'r' to read text aloud",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow("NetraVisionAi OCR Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                all_text = ocr.get_all_text(frame)
                if all_text and has_tts:
                    print(f"\n🔊 Reading: {all_text}")
                    tts.speak(f"I can read: {all_text}")
                elif all_text:
                    print(f"\n📝 Text found: {all_text}")
                else:
                    print("\n❌ No text found in view")

    finally:
        cam.stop()
        if has_tts:
            tts.stop()
        cv2.destroyAllWindows()
        print("\n✅ OCR test complete!")