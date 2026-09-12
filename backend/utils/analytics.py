"""
Behavior Analytics Module
Computes queue-level and individual-level metrics:
- Individual waiting times
- Average / max waiting times
- Queue abandonment detection and rate
- Crowd spike detection
- Arrival and service rates
"""

import time
from collections import deque
from typing import Dict, List, Any


class BehaviorAnalytics:
    """Tracks behavior patterns and computes analytics from tracking data."""

    def __init__(self):
        # Per-person tracking
        self.person_entry_times = {}    # track_id -> first_seen_in_queue time
        self.person_exit_times = {}     # track_id -> left_queue time
        self.person_served = set()      # IDs that were served (exited after waiting)
        self.person_abandoned = set()   # IDs that abandoned queue

        # Rate tracking
        self._count_history = deque(maxlen=600)  # (timestamp, count) pairs
        self._prev_in_queue_ids = set()

        # Crowd spike detection
        self._crowd_history = deque(maxlen=60)
        self._spike_cooldown = 0

    def update(self, active_tracks, all_tracks: Dict, queue_info: Dict) -> Dict:
        """
        Update analytics with current frame data.

        Returns dict with:
            avg_wait_time, max_wait_time, person_wait_times,
            abandonment_rate, arrival_rate, service_rate, crowd_spike
        """
        now = time.time()
        current_in_queue = set()
        person_wait_times = {}

        # Track who is currently in queue
        pqm = queue_info.get("person_queue_map", {})
        for track_id, in_queue in pqm.items():
            if in_queue:
                current_in_queue.add(track_id)
                if track_id not in self.person_entry_times:
                    self.person_entry_times[track_id] = now
                # Compute current wait time
                person_wait_times[track_id] = now - self.person_entry_times[track_id]

        # Detect exits
        just_left = self._prev_in_queue_ids - current_in_queue
        just_served_waits = []
        for tid in just_left:
            self.person_exit_times[tid] = now
            wait = now - self.person_entry_times.get(tid, now)
            if wait < 5.0:
                # Very short stay = likely abandonment
                self.person_abandoned.add(tid)
            else:
                self.person_served.add(tid)
                just_served_waits.append(wait)

        # Detect abandoned from lost/removed tracks
        for tid, track in all_tracks.items():
            if tid in self.person_entry_times and tid not in self.person_exit_times:
                if track.state == 3:  # Removed
                    entry = self.person_entry_times[tid]
                    if (now - entry) < 3.0:
                        self.person_abandoned.add(tid)

        # Compute aggregate wait times
        active_waits = [wt for wt in person_wait_times.values() if wt > 0]
        avg_wait = sum(active_waits) / len(active_waits) if active_waits else 0.0
        max_wait = max(active_waits) if active_waits else 0.0

        # Abandonment rate
        total_tracked = len(self.person_served) + len(self.person_abandoned)
        abandonment_rate = (len(self.person_abandoned) / total_tracked) if total_tracked > 0 else 0.0

        # Count history for rate calculation
        q_count = queue_info.get("total_in_queue", 0)
        self._count_history.append((now, q_count))

        # Arrival / service rates (per minute, 2-min window)
        arrival_rate = self._calc_arrival_rate()
        service_rate = self._calc_service_rate()

        # Crowd spike detection
        self._crowd_history.append(len(active_tracks))
        crowd_spike = False
        if len(self._crowd_history) >= 10 and self._spike_cooldown <= 0:
            recent_avg = sum(list(self._crowd_history)[-5:]) / 5
            older_avg = sum(list(self._crowd_history)[-10:-5]) / 5 if len(self._crowd_history) >= 10 else recent_avg
            if recent_avg > older_avg * 1.5 and recent_avg > 3:
                crowd_spike = True
                self._spike_cooldown = 30
        self._spike_cooldown = max(0, self._spike_cooldown - 1)

        self._prev_in_queue_ids = current_in_queue

        return {
            "avg_wait_time": avg_wait,
            "max_wait_time": max_wait,
            "person_wait_times": person_wait_times,
            "abandonment_rate": abandonment_rate,
            "total_abandoned": len(self.person_abandoned),
            "total_served": len(self.person_served),
            "arrival_rate": arrival_rate,
            "service_rate": service_rate,
            "crowd_spike": crowd_spike,
            "just_served_waits": just_served_waits,
        }

    def _calc_arrival_rate(self) -> float:
        """Estimate arrivals per minute from count increases."""
        if len(self._count_history) < 10:
            return 0.0
        now = time.time()
        cutoff = now - 120
        recent = [(t, c) for t, c in self._count_history if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        arrivals = sum(max(0, recent[i][1] - recent[i-1][1]) for i in range(1, len(recent)))
        duration = (recent[-1][0] - recent[0][0]) / 60.0
        return arrivals / max(duration, 1/60)

    def _calc_service_rate(self) -> float:
        """Estimate services per minute from count decreases."""
        if len(self._count_history) < 10:
            return 0.0
        now = time.time()
        cutoff = now - 120
        recent = [(t, c) for t, c in self._count_history if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        services = sum(max(0, recent[i-1][1] - recent[i][1]) for i in range(1, len(recent)))
        duration = (recent[-1][0] - recent[0][0]) / 60.0
        return services / max(duration, 1/60)

    def reset(self):
        self.person_entry_times.clear()
        self.person_exit_times.clear()
        self.person_served.clear()
        self.person_abandoned.clear()
        self._count_history.clear()
        self._prev_in_queue_ids.clear()
        self._crowd_history.clear()
