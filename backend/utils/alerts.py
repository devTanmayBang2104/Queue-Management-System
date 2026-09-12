"""
Smart Alert System
Monitors queue metrics and triggers configurable alerts with deduplication.
Alert types: queue_length, wait_time, crowd_spike, abandonment
"""

import time
from typing import List, Dict
from collections import deque


class Alert:
    """Single alert instance."""
    def __init__(self, alert_type: str, message: str, severity: str,
                 threshold: float = 0, actual: float = 0):
        self.alert_type = alert_type
        self.message = message
        self.severity = severity  # info, warning, critical
        self.threshold = threshold
        self.actual = actual
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "type": self.alert_type,
            "message": self.message,
            "severity": self.severity,
            "threshold": self.threshold,
            "actual": self.actual,
            "timestamp": self.timestamp,
        }


class AlertManager:
    """Manages alert generation with configurable thresholds and deduplication."""

    def __init__(self, queue_config=None):
        self.max_queue = queue_config.max_queue_length_alert if queue_config else 10
        self.max_wait = queue_config.max_wait_time_alert if queue_config else 300.0
        self.crowd_thresh = queue_config.crowd_spike_threshold if queue_config else 15

        # Deduplication: cooldown per alert type (seconds)
        self._cooldowns = {}
        self.cooldown_period = 30  # seconds between same alert type

        # Alert history
        self.history = deque(maxlen=200)
        self.recent_alerts = []

    def check(self, queue_length: int, avg_wait_time: float,
              max_wait_time: float, crowd_count: int,
              abandonment_rate: float) -> List[Dict]:
        """
        Check all conditions and return triggered alerts.
        """
        self.recent_alerts = []
        now = time.time()

        # 1. Queue length alert
        if queue_length >= self.max_queue:
            severity = "critical" if queue_length >= self.max_queue * 1.5 else "warning"
            self._maybe_alert(
                "queue_length",
                f"Queue length ({queue_length}) exceeds threshold ({self.max_queue}). Consider opening a new counter.",
                severity, self.max_queue, queue_length, now
            )

        # 2. Wait time alert
        if max_wait_time >= self.max_wait:
            self._maybe_alert(
                "wait_time",
                f"Max wait time ({max_wait_time:.0f}s) exceeds limit ({self.max_wait:.0f}s).",
                "warning", self.max_wait, max_wait_time, now
            )

        # 3. Crowd spike
        if crowd_count >= self.crowd_thresh:
            self._maybe_alert(
                "crowd_spike",
                f"Crowd count ({crowd_count}) exceeds threshold ({self.crowd_thresh}). Potential congestion.",
                "warning", self.crowd_thresh, crowd_count, now
            )

        # 4. High abandonment
        if abandonment_rate > 0.3:  # 30% abandonment rate
            self._maybe_alert(
                "abandonment",
                f"High abandonment rate ({abandonment_rate:.0%}). Service may be too slow.",
                "info", 0.3, abandonment_rate, now
            )

        return [a.to_dict() for a in self.recent_alerts]

    def _maybe_alert(self, alert_type, message, severity, threshold, actual, now):
        """Create alert if not in cooldown."""
        last = self._cooldowns.get(alert_type, 0)
        if now - last >= self.cooldown_period:
            alert = Alert(alert_type, message, severity, threshold, actual)
            self.recent_alerts.append(alert)
            self.history.append(alert)
            self._cooldowns[alert_type] = now

            # Log alert to SQLite database
            from backend.database import SessionLocal, AlertHistory
            from datetime import datetime
            db = SessionLocal()
            try:
                db_alert = AlertHistory(
                    timestamp=datetime.fromtimestamp(alert.timestamp),
                    alert_type=alert.alert_type,
                    message=alert.message,
                    severity=alert.severity,
                    threshold_value=alert.threshold,
                    actual_value=alert.actual
                )
                db.add(db_alert)
                db.commit()
            except Exception as db_err:
                print(f"[DB] Error logging alert: {db_err}")
            finally:
                db.close()

    def update_thresholds(self, max_queue=None, max_wait=None, crowd_thresh=None):
        """Update alert thresholds at runtime."""
        if max_queue is not None:
            self.max_queue = max_queue
        if max_wait is not None:
            self.max_wait = max_wait
        if crowd_thresh is not None:
            self.crowd_thresh = crowd_thresh

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history."""
        items = list(self.history)[-limit:]
        return [a.to_dict() for a in items]
