# scripts/test_indic_voices.py
import sys
import os
import time
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.tts_engine import TTSEngine, SpeechPriority

logging.basicConfig(level=logging.DEBUG)

def test_language(lang_code, lang_name):
    print(f"\n--- Testing {lang_name} ({lang_code}) ---")
    try:
        # We use piper engine and the specific language code
        # The engine should handle translation to the language and transliteration to Devanagari
        tts = TTSEngine(engine="piper", language=lang_code)
        tts.start()
        
        test_phrase = "Person ahead"
        print(f"Original: {test_phrase}")
        tts.speak(test_phrase, SpeechPriority.HIGH)
        
        # Give it time to translate, transliterate and speak
        time.sleep(5)
        tts.stop()
    except Exception as e:
        print(f"Error testing {lang_name}: {e}")

if __name__ == "__main__":
    languages = [
        ("hi", "Hindi"),
        ("ta", "Tamil"),
        ("bn", "Bengali"),
        ("gu", "Gujarati"),
        ("te", "Telugu"),
        ("bho", "Bhojpuri")
    ]
    
    for code, name in languages:
        test_language(code, name)
    
    print("\nTest complete!")
