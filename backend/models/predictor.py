"""
Queue Prediction System
Uses Linear Regression to predict expected waiting time for new entrants
based on current queue length, arrival rate, and service rate.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from collections import deque
import time


class QueuePredictor:
    """Predicts wait times using rolling statistics and linear regression."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.model = LinearRegression()
        self.is_trained = False

        # Rolling data for training
        self.history = deque(maxlen=window_size)
        self._last_queue_length = 0
        self._last_time = time.time()

        # Track arrivals and departures for rate calculation
        self._arrival_times = deque(maxlen=200)
        self._service_times = deque(maxlen=200)
        self._prev_count = 0

    def update_rates(self, current_count: int):
        """Update arrival and service rates based on queue count changes."""
        now = time.time()
        if current_count > self._prev_count:
            # People arrived
            for _ in range(current_count - self._prev_count):
                self._arrival_times.append(now)
        elif current_count < self._prev_count:
            # People served/left
            for _ in range(self._prev_count - current_count):
                self._service_times.append(now)
        self._prev_count = current_count

    @property
    def arrival_rate(self) -> float:
        """Arrivals per minute over last 2 minutes."""
        if len(self._arrival_times) < 2:
            return 0.0
        now = time.time()
        cutoff = now - 120  # 2-minute window
        recent = [t for t in self._arrival_times if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        duration = (now - recent[0]) / 60.0
        return len(recent) / max(duration, 1/60)

    @property
    def service_rate(self) -> float:
        """Services per minute over last 2 minutes."""
        if len(self._service_times) < 2:
            return 0.0
        now = time.time()
        cutoff = now - 120
        recent = [t for t in self._service_times if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        duration = (now - recent[0]) / 60.0
        return len(recent) / max(duration, 1/60)

    def add_training_point(self, queue_length: int, arrival_rate: float,
                           service_rate: float, actual_wait: float):
        """Record an actual wait time data point when a person finishes waiting."""
        self.history.append({
            "queue_length": queue_length,
            "arrival_rate": arrival_rate,
            "service_rate": service_rate,
            "actual_wait": actual_wait,
        })

    def predict(self, queue_length: int, arrival_rate: float = None,
                service_rate: float = None) -> float:
        """
        Predict waiting time for a new entrant.

        Uses Little's Law as baseline: W = L / λ (where L=queue length, λ=service rate)
        Enhanced with linear regression when enough data is available.
        """
        self.update_rates(queue_length)

        ar = arrival_rate if arrival_rate is not None else self.arrival_rate
        sr = service_rate if service_rate is not None else self.service_rate

        # Try ML prediction if we have enough data
        if len(self.history) >= 20:
            try:
                X = np.array([[d["queue_length"], d["arrival_rate"], d["service_rate"]]
                              for d in self.history])
                y = np.array([d["actual_wait"] for d in self.history])
                self.model.fit(X, y)
                self.is_trained = True

                features = np.array([[queue_length, ar, sr]])
                prediction = float(self.model.predict(features)[0])
                return max(prediction, 0)
            except Exception:
                pass

        # Fallback: Little's Law estimation
        if sr > 0:
            avg_service_time = 60.0 / sr  # seconds per service
            return max(queue_length * avg_service_time, 0)
        elif queue_length > 0:
            return queue_length * 30.0  # Default 30s per person
        return 0.0

    def reset(self):
        self.history.clear()
        self.is_trained = False
        self._arrival_times.clear()
        self._service_times.clear()
        self._prev_count = 0
