# modules/priority.py
"""
NetraVisionAi — Priority Engine
Decides WHAT to say, WHEN to say it, prevents alert flooding.

Priority Levels:
  CRITICAL (0) → Immediate danger — interrupts current speech
  HIGH (1)     → Important nearby object — spoken next
  MEDIUM (2)   → Informational — spoken when queue is free  
  LOW (3)      → Background info — spoken during silence
"""

import time
from typing import List, Optional
from dataclasses import dataclass, field
from enum import IntEnum


class AlertPriority(IntEnum):
    """Alert priority levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass(order=True)
class Alert:
    """A single alert to be spoken."""
    priority: AlertPriority
    message: str = field(compare=False)
    object_name: str = field(default="", compare=False)
    distance: str = field(default="", compare=False)
    depth_value: float = field(default=0.0, compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
    spoken: bool = field(default=False, compare=False)


class PriorityEngine:
    """
    Evaluates spatial objects and generates prioritized alerts.
    Uses cooldown timers to prevent repeating the same alert.
    """

    def __init__(self,
                 critical_depth=0.70,
                 high_depth=0.50,
                 critical_cooldown=2.0,      # Was 1.5 → Now 2.0
                 high_cooldown=4.0,          # Was 3.0 → Now 4.0
                 medium_cooldown=7.0,        # Was 10.0 → Now 7.0
                 low_cooldown=15.0,          # Was 20.0 → Now 15.0
                 max_alerts_per_cycle=4):    # Increased from 3

        self.critical_depth = critical_depth
        self.high_depth = high_depth

        self.cooldowns = {
            AlertPriority.CRITICAL: critical_cooldown,
            AlertPriority.HIGH: high_cooldown,
            AlertPriority.MEDIUM: medium_cooldown,
            AlertPriority.LOW: low_cooldown,
        }

        self.max_alerts = max_alerts_per_cycle

        # Track when each alert was last spoken
        self.last_alerts = {}

        print("[Priority] [OK] Priority Engine ready")
        print(f"[Priority] Thresholds: critical>{critical_depth}, high>{high_depth}")

    def generate_alerts(self, spatial_objects) -> List[Alert]:
        """
        Generate prioritized alerts from spatial objects.

        Args:
            spatial_objects: List of SpatialObject from SpatialMapper

        Returns:
            List of Alert, sorted by priority, max self.max_alerts
        """
        alerts = []
        now = time.time()

        for obj in spatial_objects:
            alert = self._evaluate_object(obj, now)
            if alert is not None:
                alerts.append(alert)

        # Sort by priority (CRITICAL first)
        alerts.sort(key=lambda a: a.priority)

        # Limit count
        return alerts[:self.max_alerts]

    def _evaluate_object(self, obj, now: float) -> Optional[Alert]:
        """Evaluate one spatial object → Alert or None"""

        # ─── CRITICAL: Danger object very close ───
        if obj.is_danger and obj.depth_value > self.critical_depth:
            priority = AlertPriority.CRITICAL
            message = f"Warning! {obj.class_name} {obj.position}, very close!"

        # ─── CRITICAL: ANY object extremely close ───
        elif obj.depth_value > 0.80:
            priority = AlertPriority.CRITICAL
            message = f"Obstacle {obj.position}, very close!"

        # ─── HIGH: Danger object close ───
        elif obj.is_danger and obj.depth_value > self.high_depth:
            priority = AlertPriority.HIGH
            message = f"{obj.class_name} {obj.position}, close"

        # ─── HIGH: Non-danger but very close ───
        elif obj.depth_value > self.critical_depth:
            priority = AlertPriority.HIGH
            message = f"{obj.class_name} {obj.position}, very close"

        # ─── MEDIUM: Object moderately close ───
        elif obj.depth_value > 0.35:  # Lowered from high_depth (0.50)
            priority = AlertPriority.MEDIUM
            message = f"{obj.class_name} {obj.position}"

        # ─── LOW: Far object (Surrounding details) ───
        elif obj.depth_value > 0.20:
            priority = AlertPriority.LOW
            message = f"{obj.class_name} further {obj.position}"

        # ─── Skip very far objects ───
        else:
            return None

        # ─── Cooldown check ───
        alert_key = f"{obj.class_name}_{obj.position}_{obj.distance}"
        cooldown = self.cooldowns[priority]

        if alert_key in self.last_alerts:
            time_since = now - self.last_alerts[alert_key]
            if time_since < cooldown:
                return None  # Too soon to repeat

        # Update last alert time
        self.last_alerts[alert_key] = now

        # Clean old entries (prevent memory leak)
        if len(self.last_alerts) > 100:
            cutoff = now - 30
            self.last_alerts = {
                k: v for k, v in self.last_alerts.items() if v > cutoff
            }

        return Alert(
            priority=priority,
            message=message,
            object_name=obj.class_name,
            distance=obj.distance,
            depth_value=obj.depth_value,
        )

    def generate_scene_alert(self, description: str) -> Alert:
        """Create a LOW priority scene description alert."""
        return Alert(
            priority=AlertPriority.LOW,
            message=description,
            object_name="scene",
            distance="",
            depth_value=0.0,
        )


# ═══════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════
if __name__ == "__main__":
    from spatial import SpatialObject

    print("=" * 50)
    print("  Priority Engine Test")
    print("=" * 50)

    engine = PriorityEngine()

    test_objects = [
        SpatialObject("car", 0.9, "to your left", "very close",
                       0.80, True, False, (100, 100, 300, 400), (200, 250)),
        SpatialObject("person", 0.85, "ahead", "close",
                       0.60, False, True, (250, 100, 400, 450), (325, 275)),
        SpatialObject("bench", 0.7, "to your right", "nearby",
                       0.40, False, True, (450, 200, 600, 400), (525, 300)),
        SpatialObject("dog", 0.6, "far left", "far",
                       0.15, True, False, (10, 300, 80, 400), (45, 350)),
    ]

    print("\n[INPUT] Input objects:")
    for obj in test_objects:
        print(f"  {obj.class_name}: {obj.position}, {obj.distance} "
              f"(depth={obj.depth_value:.2f}, danger={obj.is_danger})")

    alerts = engine.generate_alerts(test_objects)

    print(f"\n[ALERTS] Generated {len(alerts)} alerts:")
    for a in alerts:
        p_name = AlertPriority(a.priority).name
        print(f"  [{p_name}] {a.message}")

    # Test cooldown
    print("\n[COOLDOWN] Testing cooldown (same objects again)...")
    alerts2 = engine.generate_alerts(test_objects)
    print(f"  Alerts after cooldown: {len(alerts2)} (should be fewer)")

    print("\n[OK] Priority Engine test complete!")