"""
Queue Detection Logic
Determines which tracked persons are part of a queue based on:
1. Position inside ROI (Region of Interest)
2. Movement speed (slow/stationary = likely in queue)
Supports multiple queue zones.
"""

import numpy as np
import cv2
import time
from typing import List, Dict


class QueueDetector:
    """Detects which tracked persons are in a queue using ROI + velocity analysis."""

    def __init__(self, rois=None):
        """
        Args:
            rois: List of ROIConfig objects defining queue zones
        """
        self.rois = rois or []
        self.velocity_threshold = 3.0  # pixels/frame - below = stationary
        self.queue_entry_times = {}  # track_id -> entry timestamp
        self.queue_exit_times = {}   # track_id -> exit timestamp
        self._prev_in_queue = set()

    def set_rois(self, rois):
        """Update ROI definitions."""
        self.rois = rois

    def process(self, active_tracks, frame_shape) -> Dict:
        """
        Determine queue membership for each tracked person.

        Args:
            active_tracks: List of STrack objects
            frame_shape: (H, W, C) of the frame

        Returns:
            dict with:
                - total_in_queue: int
                - person_queue_map: {track_id: bool}
                - queue_entry_times: {track_id: timestamp}
                - queue_exit_times: {track_id: timestamp}
                - per_queue_counts: {queue_id: count}
        """
        h, w = frame_shape[:2]
        person_queue_map = {}
        per_queue_counts = {}
        current_in_queue = set()

        for track in active_tracks:
            # Get person center (bottom-center for ground position)
            tx, ty, tw, th = track.tlwh
            center_x = tx + tw / 2
            center_y = ty + th  # bottom of bbox = approximate ground pos

            in_queue = False
            assigned_queue = None

            if self.rois:
                # Check each ROI
                for roi in self.rois:
                    if len(roi.points) >= 3:
                        polygon = np.array(roi.points, dtype=np.int32)
                        result = cv2.pointPolygonTest(polygon, (center_x, center_y), False)
                        if result >= 0:  # Inside or on edge
                            # Also check velocity
                            if track.velocity < self.velocity_threshold:
                                in_queue = True
                                assigned_queue = roi.id
                                per_queue_counts[roi.id] = per_queue_counts.get(roi.id, 0) + 1
                                break
            else:
                # Default ROI: center 80% of frame
                roi_x1, roi_y1 = int(w * 0.1), int(h * 0.1)
                roi_x2, roi_y2 = int(w * 0.9), int(h * 0.9)

                if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
                    if track.velocity < self.velocity_threshold:
                        in_queue = True
                        assigned_queue = "default"
                        per_queue_counts["default"] = per_queue_counts.get("default", 0) + 1

            person_queue_map[track.track_id] = in_queue

            if in_queue:
                current_in_queue.add(track.track_id)
                # Record entry time
                if track.track_id not in self.queue_entry_times:
                    self.queue_entry_times[track.track_id] = time.time()
            else:
                # Record exit time if person was previously in queue
                if track.track_id in self._prev_in_queue:
                    self.queue_exit_times[track.track_id] = time.time()

        self._prev_in_queue = current_in_queue
        total_in_queue = sum(1 for v in person_queue_map.values() if v)

        return {
            "total_in_queue": total_in_queue,
            "person_queue_map": person_queue_map,
            "queue_entry_times": dict(self.queue_entry_times),
            "queue_exit_times": dict(self.queue_exit_times),
            "per_queue_counts": per_queue_counts,
        }

    def get_wait_time(self, track_id: int) -> float:
        """Get current wait time for a person in queue."""
        entry = self.queue_entry_times.get(track_id)
        if entry is None:
            return 0.0
        exit_t = self.queue_exit_times.get(track_id, time.time())
        return exit_t - entry

    def reset(self):
        """Reset queue state."""
        self.queue_entry_times.clear()
        self.queue_exit_times.clear()
        self._prev_in_queue.clear()
