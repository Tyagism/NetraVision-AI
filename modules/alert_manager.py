# modules/alert_manager.py
"""
NetraVision AI — Smart Alert Manager
Natural, paced speech output with scene summarization.

Design principles:
  1. Speak in SENTENCES, not individual object names
  2. BREATHE — maintain silence gaps between alerts
  3. Only speak what's NEW or CHANGED
  4. CRITICAL alerts bypass all pacing rules
"""

import time
import logging
import threading
from typing import List, Dict, Set, Optional
from collections import deque

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Intelligent alert manager that produces calm, natural speech output.
    """

    def __init__(self, tts_engine, max_history=50):
        self.tts = tts_engine
        self.alert_history = deque(maxlen=max_history)
        self.is_active = False

        # ══════════════════════════════════════
        # PACING — "Breathe Between Alerts"
        # ══════════════════════════════════════
        self.min_gap_seconds = 2.5          # Was 5.0 -> Now 2.5
        self.critical_gap_seconds = 1.0     # Gap for critical alerts
        self._last_speak_time = 1.0         # Last time we spoke a NORMAL alert
        self._last_critical_time = 0.0      # Last time we spoke a CRITICAL alert

        # ══════════════════════════════════════
        # CHANGE DETECTION — "Only Speak What's New"
        # ══════════════════════════════════════
        self._previous_scene = {}           # {object_name: {position, distance, depth}}
        self._previous_scene_time = {}      # {object_name: last_spoken_timestamp}
        self._scene_stable_count = 0
        self._scene_change_threshold = 3
        self._scene_stale_seconds = 45.0    # Re-announce static objects after this

        # ══════════════════════════════════════
        # SUMMARIZATION
        # ══════════════════════════════════════
        self._pending_objects = []          # Accumulate objects for next summary
        self._summary_interval = 5.0        # Summarize every N seconds
        self._last_summary_time = 0.0

        # Statistics
        self.total_alerts_spoken = 0
        self.alerts_suppressed = 0

        logger.info("[OK] AlertManager initialized (Natural Mode)")

    # ══════════════════════════════════════════════
    # LIFECYCLE
    # ══════════════════════════════════════════════

    def start(self):
        """Start the alert manager and TTS engine."""
        self.is_active = True
        if self.tts and not self.tts.is_running:
            self.tts.start()
        logger.info("[OK] AlertManager started (Natural Mode)")
        return self

    def stop(self):
        """Stop the alert manager and TTS engine."""
        logger.info("🛑 Stopping AlertManager...")
        self.is_active = False
        if self.tts:
            self.tts.stop()
        logger.info("[OK] AlertManager stopped")

    def pause(self):
        self.is_active = False
        logger.info("⏸️ Alerts paused")

    def resume(self):
        self.is_active = True
        logger.info("▶️ Alerts resumed")

    # ══════════════════════════════════════════════
    # SPECIAL MESSAGES
    # ══════════════════════════════════════════════

    def speak_startup(self, text="NetraVision AI activated. Walk safely."):
        if self.tts:
            self.tts.speak_blocking(text)
        self._last_speak_time = time.time()

    def speak_shutdown(self, text="NetraVision AI shutting down. Goodbye."):
        if self.tts:
            self.tts.speak_blocking(text)

    def speak_now(self, text, priority=None):
        if not self.tts:
            return
        if priority is not None:
            from modules.tts_engine import SpeechPriority
            self.tts.speak(text, SpeechPriority.HIGH)
        else:
            self.tts.speak_blocking(text)

    # ══════════════════════════════════════════════
    # MAIN ALERT PROCESSING — The Smart Part
    # ══════════════════════════════════════════════

    def process_alerts(self, alerts):
        """
        Process alerts with natural pacing and summarization.

        Instead of speaking every alert immediately, this:
        1. Separates CRITICAL from normal alerts
        2. Speaks CRITICAL immediately (interrupt)
        3. Accumulates normal alerts
        4. Summarizes into natural sentences
        5. Respects silence gaps between speech
        """
        if not self.is_active or not alerts:
            return

        now = time.time()

        # ── Split: Critical vs Normal ──
        critical_alerts = [a for a in alerts if a.priority == 0]
        normal_alerts = [a for a in alerts if a.priority > 0]

        # ── CRITICAL: Speak immediately, bypass all pacing ──
        for alert in critical_alerts:
            time_since_last_critical = now - self._last_critical_time
            if time_since_last_critical >= self.critical_gap_seconds:
                self._speak_critical(alert)
                self._last_critical_time = now
                # Note: We DO NOT reset self._last_speak_time here anymore,
                # so that critical alerts don't mute the background info.

        # ── NORMAL: Check if enough time has passed ──
        time_since_last_speak = now - self._last_speak_time
        if time_since_last_speak < self.min_gap_seconds:
            self.alerts_suppressed += len(normal_alerts)
            return  # Too soon — stay quiet

        # ── Detect what's NEW or CHANGED ──
        new_alerts = self._filter_new_or_changed(normal_alerts)

        if not new_alerts:
            return  # Nothing new to say

        # ── Build natural sentence ──
        sentence = self._build_natural_sentence(new_alerts)

        if sentence:
            from modules.tts_engine import SpeechPriority

            # Determine priority of the sentence
            highest = min(a.priority for a in new_alerts)
            priority_map = {
                1: SpeechPriority.HIGH,
                2: SpeechPriority.MEDIUM,
                3: SpeechPriority.LOW,
            }
            speech_priority = priority_map.get(int(highest), SpeechPriority.MEDIUM)

            self.tts.speak(sentence, speech_priority)
            self._last_speak_time = now
            self.total_alerts_spoken += 1

            for alert in new_alerts:
                self.alert_history.append(alert)

    def add_alerts(self, alerts):
        """Alias for process_alerts."""
        self.process_alerts(alerts)

    # ══════════════════════════════════════════════
    # CRITICAL ALERT — Immediate Interrupt
    # ══════════════════════════════════════════════

    def _speak_critical(self, alert):
        """Speak critical alert immediately — interrupt everything."""
        from modules.tts_engine import SpeechPriority

        self.tts.speak(alert.message, SpeechPriority.CRITICAL)
        self.total_alerts_spoken += 1
        self.alert_history.append(alert)

        logger.info(f"CRITICAL: {alert.message}")

    # ══════════════════════════════════════════════
    # CHANGE DETECTION — Only Speak What's New
    # ══════════════════════════════════════════════

    def _filter_new_or_changed(self, alerts) -> list:
        """Only keep alerts about NEW objects or CHANGED distances."""
        now = time.time()
        new_alerts = []
        current_scene = {}

        for alert in alerts:
            key = f"{alert.object_name}_{self._get_position(alert)}"
            current_scene[key] = {
                "distance": alert.distance,
                "message": alert.message,
                "depth": alert.depth_value,
            }

            # Check if this is new or changed
            is_stale = (now - self._previous_scene_time.get(key, 0) > self._scene_stale_seconds)
            
            if key not in self._previous_scene or is_stale:
                # New object or old one we haven't mentioned in a while
                new_alerts.append(alert)
                self._previous_scene_time[key] = now
            else:
                old = self._previous_scene[key]
                # Distance changed significantly
                if old["distance"] != alert.distance:
                    new_alerts.append(alert)
                    self._previous_scene_time[key] = now
                # Object got much closer (depth increased by >0.15)
                elif alert.depth_value - old.get("depth", 0) > 0.15:
                    new_alerts.append(alert)
                    self._previous_scene_time[key] = now

        # Check if objects DISAPPEARED (path cleared)
        disappeared = set(self._previous_scene.keys()) - set(current_scene.keys())
        if disappeared and len(current_scene) == 0 and len(self._previous_scene) > 0:
            # Scene cleared — mention it
            from modules.priority import Alert, AlertPriority
            clear_alert = Alert(
                priority=AlertPriority.LOW,
                message="The path ahead appears clear.",
                object_name="clear",
                distance="",
                depth_value=0.0,
            )
            new_alerts.append(clear_alert)

        self._previous_scene = current_scene
        return new_alerts

    def _get_position(self, alert) -> str:
        """Extract position from alert message."""
        # Try to find position keywords in message
        positions = ["far left", "to your left", "ahead",
                     "to your right", "far right",
                     "left", "right", "center"]
        msg = alert.message.lower()
        for pos in positions:
            if pos in msg:
                return pos
        return "unknown"

    # ══════════════════════════════════════════════
    # NATURAL SENTENCE BUILDER
    # ══════════════════════════════════════════════

    def _build_natural_sentence(self, alerts) -> str:
        """
        Combine multiple alerts into one natural sentence.

        Instead of:
          "Person ahead" + "Chair to your right" + "Bottle ahead"
        
        Produces:
          "There's a person ahead and a chair on your right."
        """
        if not alerts:
            return ""

        if len(alerts) == 1:
            return self._humanize_single(alerts[0])

        # Group by urgency
        high_alerts = [a for a in alerts if a.priority <= 1]
        other_alerts = [a for a in alerts if a.priority > 1]

        parts = []

        # High priority items first
        if high_alerts:
            high_parts = [self._describe_object(a) for a in high_alerts[:2]]
            parts.extend(high_parts)

        # Then other items
        if other_alerts:
            other_parts = [self._describe_object(a) for a in other_alerts[:2]]
            parts.extend(other_parts)

        # Limit to 3 items max
        parts = parts[:3]

        if len(parts) == 1:
            return f"There's {parts[0]}."
        elif len(parts) == 2:
            return f"There's {parts[0]}, and {parts[1]}."
        else:
            return f"There's {parts[0]}, {parts[1]}, and {parts[2]}."

    def _humanize_single(self, alert) -> str:
        """Make a single alert sound natural."""
        obj = alert.object_name
        msg = alert.message

        # If already a good sentence, use it
        if msg.startswith("Warning") or msg.startswith("Obstacle"):
            return msg

        # Build natural sentence
        distance = alert.distance if alert.distance else ""

        if "very close" in msg or alert.depth_value > 0.7:
            return f"Careful, {obj} {self._extract_position(msg)}, very close."
        elif "close" in msg or alert.depth_value > 0.5:
            return f"There's a {obj} {self._extract_position(msg)}, getting close."
        else:
            return f"I notice a {obj} {self._extract_position(msg)}."

    def _describe_object(self, alert) -> str:
        """Short natural description for sentence combining."""
        obj = alert.object_name
        pos = self._extract_position(alert.message)
        dist = alert.distance

        if dist and dist != "":
            return f"a {obj} {pos}, {dist}"
        else:
            return f"a {obj} {pos}"

    def _extract_position(self, message) -> str:
        """Extract position phrase from alert message."""
        positions = [
            "to your left", "to your right",
            "far left", "far right",
            "ahead",
        ]
        msg_lower = message.lower()
        for pos in positions:
            if pos in msg_lower:
                return pos
        return "nearby"

    # ══════════════════════════════════════════════
    # QUEUE INFO
    # ══════════════════════════════════════════════

    def get_queue_size(self):
        if self.tts and hasattr(self.tts, 'speech_queue'):
            return self.tts.speech_queue.qsize()
        return 0

    def get_stats(self) -> dict:
        return {
            "total_spoken": self.total_alerts_spoken,
            "suppressed": self.alerts_suppressed,
            "history_size": len(self.alert_history),
            "is_active": self.is_active,
            "queue_size": self.get_queue_size(),
            "scene_objects": len(self._previous_scene),
        }


# ═══════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    import os

    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("[TEST] NetraVision AI Smart Alert Manager Test")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.tts_engine import TTSEngine, SpeechPriority
    from modules.priority import Alert, AlertPriority

    tts = TTSEngine(language="en", speed=150)
    manager = AlertManager(tts_engine=tts)
    manager.start()

    manager.speak_startup("NetraVision AI started. Natural alert mode active.")
    time.sleep(2)

    # --- Cycle 1: New scene ---
    print("\n[CYCLE 1] First scene")
    alerts1 = [
        Alert(priority=AlertPriority.HIGH,
              message="person ahead, close",
              object_name="person", distance="close", depth_value=0.6),
        Alert(priority=AlertPriority.MEDIUM,
              message="chair to your right",
              object_name="chair", distance="nearby", depth_value=0.4),
    ]
    manager.process_alerts(alerts1)
    print("  -> Should speak: natural sentence about person + chair")

    # --- Cycle 2: Same scene (too soon) ---
    print("\n[CYCLE 2] Same scene, 2 seconds later")
    manager.process_alerts(alerts1)
    print("  -> Should stay SILENT (min gap not reached)")

    # --- Cycle 3: Same scene (gap reached but nothing new) ---
    print("\n[CYCLE 3] Same scene, after gap")
    manager.process_alerts(alerts1)
    print("  -> Should stay SILENT (nothing changed)")

    # --- Cycle 4: Distance changed ---
    print("\n[CYCLE 4] Person got closer")
    alerts4 = [
        Alert(priority=AlertPriority.HIGH,
              message="person ahead, very close",
              object_name="person", distance="very close", depth_value=0.8),
        Alert(priority=AlertPriority.MEDIUM,
              message="chair to your right",
              object_name="chair", distance="nearby", depth_value=0.4),
    ]
    manager.process_alerts(alerts4)
    print("  -> Should speak: person getting closer")

    # --- Cycle 5: CRITICAL ---
    print("\n[CYCLE 5] CRITICAL -- car approaching!")
    alerts5 = [
        Alert(priority=AlertPriority.CRITICAL,
              message="Warning! Car to your left, very close!",
              object_name="car", distance="very close", depth_value=0.85),
    ]
    manager.process_alerts(alerts5)
    print("  -> Should INTERRUPT and speak immediately")

    # --- Cycle 6: Scene cleared ---
    print("\n[CYCLE 6] Scene cleared")
    manager.process_alerts([])
    print("  -> Should mention path is clear")

    # Wait for all speech
    time.sleep(8)

    stats = manager.get_stats()
    print(f"\n[STATS] Stats: {stats}")
    print(f"\nStats: {stats}")

    manager.speak_shutdown()
    time.sleep(3)
    manager.stop()

    print("\n[OK] Smart Alert Manager test complete!")