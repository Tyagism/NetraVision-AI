# modules/tts_engine.py
"""
NetraVisionAi — Text-to-Speech Engine
Priority-based speech output with interrupt capability.
"""

import threading
import subprocess
import logging
import queue
import time
import platform
import shutil
import io
import os
import json
from enum import IntEnum
from pathlib import Path

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    from piper.voice import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    try:
        from piper import voice as piper_voice
        PIPER_AVAILABLE = True
    except ImportError:
        PIPER_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_TRANSLIT_AVAILABLE = True
except ImportError:
    INDIC_TRANSLIT_AVAILABLE = False

try:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf
    import torch
    PARLER_AVAILABLE = True
except ImportError:
    PARLER_AVAILABLE = False

logger = logging.getLogger(__name__)


class SpeechPriority(IntEnum):
    """Speech priority levels — lower number = higher priority."""
    CRITICAL = 0   # Immediate danger: "Car approaching!"
    HIGH = 1       # Important: "Person ahead", "Stairs"
    MEDIUM = 2     # Useful: "Sign says Exit", "Shop name"
    LOW = 3        # Background: "You are in a park"


class TTSEngine:
    """
    Text-to-Speech engine with priority queue and interrupts.
    Uses espeak-ng (lightweight) or piper (natural voice).
    """

    def __init__(self, engine="piper", language="en",
                 speed=220, volume=100):
        """
        Args:
            engine: 'espeak' or 'piper' (default: piper for better quality)
            language: Language code ('en', 'hi', etc.)
            speed: Words per minute (100-300)
            volume: Volume level (0-200)
        """
        self.engine = engine
        self.language = language
        self.speed = speed
        self.volume = volume

        # Priority queue for speech
        self.speech_queue = queue.PriorityQueue()
        self.is_running = False
        self.is_speaking = False
        self._thread = None
        self._current_process = None
        self._lock = threading.Lock()

        # Anti-spam: don't repeat same message within cooldown
        self._recent_messages = {}
        self._translation_cache = {}  # Cache for translated sentences
        self.cooldown_seconds = 3.0

        self.use_parler = False
        self.parler_model = None
        self.parler_tokenizer = None

        # Track message count for FIFO ordering within same priority
        self._counter = 0

        self.translator = None
        if TRANSLATOR_AVAILABLE and self.language != "en":
            try:
                base_lang = self.language.split('_')[0]
                self.translator = GoogleTranslator(source='en', target=base_lang)
                logger.info(f"🌐 Translator initialized for native language: {base_lang}")
                
                indic_langs = {
                    "hi": sanscript.DEVANAGARI,
                    "mr": sanscript.DEVANAGARI,
                    "bn": sanscript.BENGALI,
                    "ta": sanscript.TAMIL,
                    "te": sanscript.TELUGU,
                    "gu": sanscript.GUJARATI,
                    "kn": sanscript.KANNADA,
                    "ml": sanscript.MALAYALAM,
                    "pa": sanscript.GURMUKHI,
                    "or": sanscript.ORIYA,
                    "sa": sanscript.DEVANAGARI,
                    "as": sanscript.BENGALI,
                    "mai": sanscript.DEVANAGARI,
                    "kok": sanscript.DEVANAGARI,
                    "ne": sanscript.DEVANAGARI,
                    "mni": sanscript.BENGALI,
                    "doi": sanscript.DEVANAGARI,
                    "brx": sanscript.DEVANAGARI,
                    "sat": sanscript.DEVANAGARI,
                    "bho": sanscript.DEVANAGARI
                }
                self.indic_script = indic_langs.get(base_lang)
                
                if base_lang in indic_langs and getattr(self, "PARLER_AVAILABLE", PARLER_AVAILABLE):
                    # Keep Parler as an option but prioritize Piper for now per user request
                    pass
                        
            except Exception as e:
                logger.error(f"[ERROR] Failed to initialize translator: {e}")

        self._verify_engine()

    def _verify_engine(self):
        """Verify TTS engine is available."""
        if self.engine == "espeak":
            cmd = "espeak-ng" if shutil.which("espeak-ng") else "espeak"
            if not shutil.which(cmd):
                logger.error("[ERROR] espeak-ng not found! Install it:")
                logger.error("   Windows: Download from github.com/espeak-ng/espeak-ng/releases")
                logger.error("   Linux: sudo apt install espeak-ng")
                logger.error("   macOS: brew install espeak")
                raise RuntimeError("espeak-ng not found")
            self._espeak_cmd = cmd
            logger.info(f"[OK] TTS Engine: {cmd}")
        elif self.engine == "piper":
            if not PIPER_AVAILABLE:
                logger.warning("[WARN] Piper not available, falling back to espeak")
                logger.warning("   Install with: pip install piper-tts")
                self.engine = "espeak"
                self._verify_engine()
                return
            if not SOUNDDEVICE_AVAILABLE:
                logger.warning("[WARN] sounddevice not available for Piper audio output")
                logger.warning("   Install with: pip install sounddevice")
            self._init_piper()
            logger.info("[OK] TTS Engine: Piper")

    def _init_piper(self):
        """Initialize Piper voice model."""
        try:
            from piper.voice import PiperVoice
            
            # Primary voice model preferences
            voice_model_map = {
                "en": "models/piper/en_US-lessac-medium.onnx",
                "en_male": "models/piper/en_US-lessac-medium.onnx",
                "hi": "models/piper/hi_IN-priyamvada-medium.onnx",
                "hi_female": "models/piper/hi_IN-priyamvada-medium.onnx",
            }
            
            requested_lang = self.language
            model_path = voice_model_map.get(self.language, voice_model_map["en"])
            
            # For Indic languages, force use of Hindi model if native model not found
            if getattr(self, "indic_script", None) and self.language not in voice_model_map:
                model_path = voice_model_map["hi"]
                logger.info(f"[INFO] Indic language '{self.language}' detected. Using Hindi voice model for synthesis.")
            if not os.path.exists(model_path):
                # If primary model doesn't exist, try to find any available model for that language
                if self.language.startswith("hi") or getattr(self, "indic_script", None):
                    # Look for any hi_IN model available (Priyamvada preferred)
                    model_dir = Path("models/piper")
                    if model_dir.exists():
                        hi_models = list(model_dir.glob("hi_IN-*.onnx"))
                        if hi_models:
                            # Prioritize priyamvada
                            priyamvada = [m for m in hi_models if "priyamvada" in m.name]
                            model_path = str(priyamvada[0] if priyamvada else hi_models[0])
                            logger.info(f"[INFO] Using Hindi model for Indic support: {Path(model_path).name}")
                
                # If still not found, try to find any model for requested language
                if not os.path.exists(model_path) and self.language != "en":
                    model_dir = Path("models/piper")
                    if model_dir.exists():
                        lang_prefix = self.language.split("_")[0] + "_"
                        lang_models = list(model_dir.glob(f"{lang_prefix}*.onnx"))
                        if lang_models:
                            model_path = str(lang_models[0])
                            logger.info(f"[INFO] Found model for {self.language}: {lang_models[0].name}")
            
            # Check if model exists for requested language
            if not os.path.exists(model_path):
                if self.language != "en":
                    # Requested language model doesn't exist, try English
                    logger.warning(f"[WARN] MISSING: Piper voice model for '{requested_lang}'")
                    logger.warning(f"    Expected path: {model_path}")
                    logger.warning(f"    ")
                    logger.warning(f"[DOWNLOAD] To use {requested_lang.upper()}:")
                    logger.warning(f"    1. Download from: https://huggingface.co/rhasspy/piper-voices")
                    logger.warning(f"    2. Place in: models/piper/")
                    logger.warning(f"    3. Re-run the application")
                    logger.warning(f"    ")
                    
                    # Fall back to English model
                    en_model = voice_model_map["en"]
                    if os.path.exists(en_model):
                        logger.warning(f"    Using English as fallback for now...")
                        model_path = en_model
                    else:
                        raise FileNotFoundError(f"Neither {requested_lang} nor English model found!")
                else:
                    # English model doesn't exist
                    logger.error(f"[ERROR] CRITICAL: English Piper model not found!")
                    logger.error(f"    Expected: {model_path}")
                    raise FileNotFoundError(f"Piper English model not found: {model_path}")
            
            # Load the voice model
            config_path = model_path + ".json"
            self._piper_voice = PiperVoice.load(
                model_path, 
                config_path=config_path if os.path.exists(config_path) else None
            )
            logger.info(f"[OK] Loaded Piper voice: {model_path}")
            # Keep original language setting even if using fallback voice model
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize Piper: {e}")
            logger.warning(f"[WARN] Falling back to espeak-ng...")
            self.engine = "espeak"
            try:
                self._verify_engine()
            except:
                # If both fail, we'll fail when trying to speak
                logger.error(f"[ERROR] Neither Piper nor espeak available!")
                raise RuntimeError("No TTS engine available (install espeak-ng or check Piper setup)")


    def start(self):
        """Start the speech output thread."""
        self.is_running = True
        self._thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._thread.start()
        logger.info("[START] TTS Engine started")
        return self

    def _speech_loop(self):
        """Background thread: processes speech queue."""
        while self.is_running:
            try:
                # Get next speech item (blocks until available)
                priority, counter, text = self.speech_queue.get(timeout=0.5)

                if self.translator and self.language != "en":
                    try:
                        # Use cache to avoid redundant network calls
                        if text in self._translation_cache:
                            text = self._translation_cache[text]
                        else:
                            translated_text = self.translator.translate(text)
                            if translated_text:
                                self._translation_cache[text] = translated_text
                                text = translated_text
                        
                        # Transliterate to Devanagari if it's an Indic language but not in Devanagari
                        if getattr(self, "indic_script", None) and self.indic_script != sanscript.DEVANAGARI and INDIC_TRANSLIT_AVAILABLE:
                            try:
                                text = transliterate(text, self.indic_script, sanscript.DEVANAGARI)
                                logger.debug(f"Transliterated to Devanagari: {text}")
                            except Exception as te:
                                logger.warning(f"Transliteration failed: {te}")
                    except Exception as e:
                        logger.error(f"Translation/Transliteration failed: {e}")

                # Check if still relevant (not stale)
                self.is_speaking = True
                self._speak(text)
                self.is_speaking = False

                self.speech_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS error: {e}")
                self.is_speaking = False

    def speak(self, text, priority=SpeechPriority.MEDIUM):
        """
        Add text to speech queue.
        Args:
            text: Text to speak
            priority: SpeechPriority level
        """
        if not text or not text.strip():
            return

        text = text.strip()

        # Anti-spam check
        now = time.time()
        if text in self._recent_messages:
            if now - self._recent_messages[text] < self.cooldown_seconds:
                return  # Skip duplicate
        self._recent_messages[text] = now

        # Clean old messages from recent cache
        self._recent_messages = {
            k: v for k, v in self._recent_messages.items()
            if now - v < self.cooldown_seconds * 2
        }

        # If CRITICAL, interrupt current speech
        if priority == SpeechPriority.CRITICAL:
            self._interrupt()

        # Add to priority queue
        self._counter += 1
        self.speech_queue.put((priority, self._counter, text))

    def _speak(self, text):
        """Actually speak the text using the selected engine."""
        try:
            if getattr(self, "use_parler", False) and self.parler_model:
                self._speak_parler(text)
            elif self.engine == "espeak":
                self._speak_espeak(text)
            elif self.engine == "piper":
                self._speak_piper(text)
        except Exception as e:
            logger.error(f"Speech failed: {e}")

    def _speak_espeak(self, text):
        """Speak using espeak-ng."""
        # Map language codes to espeak voices
        voice_map = {
            "en": "en",
            "hi": "hi",
            "ta": "ta",
            "te": "te",
            "bn": "bn",
            "mr": "mr",
            "gu": "gu",
            "kn": "kn",
        }
        voice = voice_map.get(self.language, "en")

        cmd = [
            self._espeak_cmd,
            "-v", voice,
            "-s", str(self.speed),
            "-a", str(self.volume),
            text
        ]

        with self._lock:
            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        self._current_process.wait()

        with self._lock:
            self._current_process = None

    def _speak_piper(self, text):
        """Speak using Piper TTS (Python API - cross-platform)."""
        try:
            if not PIPER_AVAILABLE:
                logger.warning("Piper not available, falling back to espeak")
                self.engine = "espeak"
                self._speak_espeak(text)
                return
            
            import tempfile
            import wave
            
            # Synthesize to WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            with wave.open(temp_path, 'wb') as wav_file:
                self._piper_voice.synthesize_wav(text, wav_file)
            
            # Play audio with sounddevice if available
            if SOUNDDEVICE_AVAILABLE:
                import scipy.io.wavfile as wavfile
                sample_rate, audio_data = wavfile.read(temp_path)
                
                # Normalize audio if needed
                if audio_data.dtype != 'float32':
                    audio_data = audio_data.astype('float32')
                    if audio_data.max() > 1.0:
                        audio_data = audio_data / 32768.0
                
                # Play audio
                with self._lock:
                    sd.play(audio_data, samplerate=sample_rate)
                
                # Wait outside lock so interrupts can get the lock
                sd.wait()
            else:
                # Fallback: use system audio player
                logger.warning("sounddevice not available, attempting system playback")
                if platform.system() == "Windows":
                    import winsound
                    winsound.PlaySound(temp_path, winsound.SND_FILENAME)
                elif platform.system() == "Darwin":
                    subprocess.run(["afplay", temp_path], check=False)
                else:
                    subprocess.run(["aplay", temp_path], check=False)
            
            # Cleanup
            try:
                os.unlink(temp_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
            # Fallback to espeak
            self.engine = "espeak"
            try:
                self._speak_espeak(text)
            except:
                pass

    def _speak_parler(self, text):
        """Speak using Indic Parler-TTS."""
        try:
            prompt = "A female speaker delivers a clear and paced alert in a calm voice."
            # Tokenize description and text
            input_ids = self.parler_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            prompt_input_ids = self.parler_tokenizer(text, return_tensors="pt").input_ids.to(self.device)

            generation = self.parler_model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids
            )
            
            audio_arr = generation.cpu().numpy().squeeze()
            sample_rate = self.parler_model.config.sampling_rate

            if SOUNDDEVICE_AVAILABLE:
                with self._lock:
                    sd.play(audio_arr, samplerate=sample_rate)
                    sd.wait()
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                sf.write(temp_path, audio_arr, sample_rate)
                if platform.system() == "Windows":
                    import winsound
                    winsound.PlaySound(temp_path, winsound.SND_FILENAME)
                elif platform.system() == "Darwin":
                    subprocess.run(["afplay", temp_path], check=False)
                else:
                    subprocess.run(["aplay", temp_path], check=False)
                try:
                    os.unlink(temp_path)
                except:
                    pass
        except Exception as e:
            logger.error(f"Parler-TTS synthesis failed: {e}")
            self.use_parler = False  # disable on fail to fallback next time
            self._speak_piper(text)

    def _interrupt(self):
        """Interrupt current speech for critical alerts."""
        with self._lock:
            # 1. Stop subprocess (espeak)
            if self._current_process and self._current_process.poll() is None:
                self._current_process.terminate()
                logger.info("[STOP] Speech process interrupted")
            
            # 2. Stop sounddevice (piper/parler)
            if SOUNDDEVICE_AVAILABLE:
                import sounddevice as sd
                sd.stop()
                logger.info("[STOP] Audio playback stopped")

        # Clear only LOW priority items from queue, keep HIGH/MEDIUM
        temp_items = []
        while not self.speech_queue.empty():
            try:
                item = self.speech_queue.get_nowait()
                # Keep critical (0), high (1), and medium (2). Discard low (3).
                if item[0] < SpeechPriority.LOW:
                    temp_items.append(item)
            except queue.Empty:
                break

        for item in temp_items:
            self.speech_queue.put(item)

    def speak_blocking(self, text):
        """Speak immediately and wait (for startup messages)."""
        if self.translator and self.language != "en":
            try:
                translated_text = self.translator.translate(text)
                if translated_text:
                    text = translated_text
                    # Transliterate to Devanagari if it's an Indic language but not in Devanagari
                    if getattr(self, "indic_script", None) and self.indic_script != sanscript.DEVANAGARI and INDIC_TRANSLIT_AVAILABLE:
                        try:
                            text = transliterate(text, self.indic_script, sanscript.DEVANAGARI)
                            logger.debug(f"Transliterated to Devanagari: {text}")
                        except Exception as te:
                            logger.warning(f"Transliteration failed: {te}")
            except Exception as e:
                logger.error(f"Translation failed: {e}")
        self._speak(text)

    def stop(self):
        """Stop TTS engine."""
        logger.info("🛑 Stopping TTS engine...")
        self.is_running = False
        self._interrupt()

        if self._thread:
            self._thread.join(timeout=3.0)

        logger.info("✅ TTS engine stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ═══════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 50)
    print("[START] NetraVisionAi TTS Test")
    print("=" * 50)

    with TTSEngine(engine="espeak", language="en", speed=195) as tts:
        print("\n[TEST] Testing priority levels...\n")

        tts.speak("Low priority: The weather seems pleasant", SpeechPriority.LOW)
        tts.speak("Medium priority: There is a bench on your right", SpeechPriority.MEDIUM)
        tts.speak("High priority: Person walking towards you", SpeechPriority.HIGH)

        time.sleep(2)

        tts.speak("Critical alert: Car approaching from left!", SpeechPriority.CRITICAL)

        # Wait for all speech to finish
        time.sleep(8)

    print("\n[OK] TTS test complete!")