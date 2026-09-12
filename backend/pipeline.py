"""
CV Pipeline Orchestrator
Ties together: Detection → Tracking → Queue Logic → Analytics → Prediction → Alerts
Runs in a background thread and broadcasts results via WebSocket.
"""

import cv2
import numpy as np
import time
import threading
import base64
import asyncio
from collections import deque

from backend.models.detector import PersonDetector
from backend.models.tracker import BYTETracker, STrack
from backend.models.queue_detector import QueueDetector
from backend.models.predictor import QueuePredictor
from backend.utils.analytics import BehaviorAnalytics
from backend.utils.heatmap import HeatmapGenerator
from backend.utils.alerts import AlertManager
from backend.config import AppConfig


class PipelineState:
    """Thread-safe shared state between pipeline and API handlers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_b64 = None
        self._stats = {}
        self._persons = []
        self._alerts = []
        self._heatmap_b64 = None
        self._history = deque(maxlen=500)
        self._running = False
        self._video_source = None

    @property
    def running(self):
        with self._lock:
            return self._running

    @running.setter
    def running(self, val):
        with self._lock:
            self._running = val

    def update(self, frame_b64, stats, persons, alerts, heatmap_b64=None):
        with self._lock:
            self._frame_b64 = frame_b64
            self._stats = stats
            self._persons = persons
            self._alerts = alerts
            if heatmap_b64 is not None:
                self._heatmap_b64 = heatmap_b64
            self._history.append({"timestamp": time.time(), "stats": stats.copy()})

    def get_snapshot(self):
        with self._lock:
            return {
                "frame": self._frame_b64,
                "stats": dict(self._stats),
                "persons": list(self._persons),
                "alerts": list(self._alerts),
            }

    def get_heatmap(self):
        with self._lock:
            return self._heatmap_b64

    def get_history(self):
        with self._lock:
            return list(self._history)

    def get_stats(self):
        with self._lock:
            return dict(self._stats)


class CVPipeline:
    """Main Computer Vision Pipeline."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.state = PipelineState()
        self._thread = None
        self._stop_event = threading.Event()

        # WebSocket clients
        self.ws_clients = set()
        self._ws_lock = threading.Lock()
        self._loop = None

        # CV Components
        self.detector = None
        self.tracker = None
        self.queue_detector = None
        self.predictor = None
        self.analytics = None
        self.heatmap_gen = None
        self.alert_manager = None

        self._fps_history = deque(maxlen=30)

    def _init_components(self):
        """Lazy-initialize CV components (heavy imports happen here)."""
        print("[Pipeline] Initializing components...")
        self.detector = PersonDetector(
            model_name=self.config.detector.model_name,
            confidence=self.config.detector.confidence_threshold,
        )
        self.tracker = BYTETracker(
            track_high_thresh=self.config.tracker.track_high_thresh,
            track_low_thresh=self.config.tracker.track_low_thresh,
            new_track_thresh=self.config.tracker.new_track_thresh,
            track_buffer=self.config.tracker.track_buffer,
            match_thresh=self.config.tracker.match_thresh,
        )
        self.queue_detector = QueueDetector(self.config.rois)
        self.predictor = QueuePredictor()
        self.analytics = BehaviorAnalytics()
        self.heatmap_gen = HeatmapGenerator(self.config.frame_width, self.config.frame_height)
        self.alert_manager = AlertManager(self.config.queue)
        print("[Pipeline] All components initialized")

    def start(self, video_source=None, loop=None):
        """Start pipeline in background thread."""
        if self.state.running:
            return
        self._loop = loop
        source = video_source or self.config.video_source
        self.state._video_source = source
        self._stop_event.clear()
        self.state.running = True
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the pipeline."""
        self._stop_event.set()
        self.state.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self, source):
        """Main processing loop (runs in background thread)."""
        # Initialize components in the thread
        if self.detector is None:
            self._init_components()

        # Open video source
        src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"[Pipeline] ERROR: Cannot open source: {source}")
            self.state.running = False
            return

        # Try to set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.heatmap_gen = HeatmapGenerator(actual_w, actual_h)

        frame_count = 0
        last_db_log_time = 0
        print(f"[Pipeline] Processing from: {source} ({actual_w}x{actual_h})")

        while not self._stop_event.is_set():
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                if not isinstance(src, int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_count += 1

            # 1. Detect
            detections = self.detector.detect(frame)

            # 2. Track
            active_tracks = self.tracker.update(detections)

            # 3. Queue detection
            queue_info = self.queue_detector.process(active_tracks, frame.shape)

            # 4. Analytics
            all_tracks = self.tracker.get_all_track_info()
            analytics_data = self.analytics.update(active_tracks, all_tracks, queue_info)

            # 5. Prediction
            # Feed completed wait times to the predictor for training
            for wait_time in analytics_data.get("just_served_waits", []):
                self.predictor.add_training_point(
                    queue_length=queue_info["total_in_queue"],
                    arrival_rate=analytics_data.get("arrival_rate", 0),
                    service_rate=analytics_data.get("service_rate", 0),
                    actual_wait=wait_time
                )

            prediction = self.predictor.predict(
                queue_length=queue_info["total_in_queue"],
                arrival_rate=analytics_data.get("arrival_rate", 0),
                service_rate=analytics_data.get("service_rate", 0),
            )

            # 6. Heatmap (update every frame, generate image periodically)
            self.heatmap_gen.update(active_tracks)
            heatmap_b64 = None
            if frame_count % 30 == 0:
                hm = self.heatmap_gen.generate()
                _, hm_buf = cv2.imencode(".jpg", hm, [cv2.IMWRITE_JPEG_QUALITY, 80])
                heatmap_b64 = base64.b64encode(hm_buf).decode("utf-8")

            # 7. Alerts
            alerts = self.alert_manager.check(
                queue_length=queue_info["total_in_queue"],
                avg_wait_time=analytics_data.get("avg_wait_time", 0),
                max_wait_time=analytics_data.get("max_wait_time", 0),
                crowd_count=len(active_tracks),
                abandonment_rate=analytics_data.get("abandonment_rate", 0),
            )

            # 8. Annotate frame
            annotated = self._draw(frame, active_tracks, queue_info, analytics_data, prediction)

            # FPS
            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)
            self._fps_history.append(fps)
            avg_fps = sum(self._fps_history) / len(self._fps_history)

            # Encode frame
            _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(jpg).decode("utf-8")

            # Stats
            stats = {
                "queue_count": queue_info["total_in_queue"],
                "avg_wait_time": round(analytics_data.get("avg_wait_time", 0), 1),
                "max_wait_time": round(analytics_data.get("max_wait_time", 0), 1),
                "predicted_wait_time": round(prediction, 1),
                "fps": round(avg_fps, 1),
                "total_tracked": len(all_tracks),
                "active_tracks": len(active_tracks),
                "abandonment_rate": round(analytics_data.get("abandonment_rate", 0), 3),
                "total_abandoned": analytics_data.get("total_abandoned", 0),
                "total_served": analytics_data.get("total_served", 0),
                "arrival_rate": round(analytics_data.get("arrival_rate", 0), 2),
                "service_rate": round(analytics_data.get("service_rate", 0), 2),
                "frame_count": frame_count,
                "timestamp": time.time(),
            }

            # Periodic SQLite DB write
            if time.time() - last_db_log_time >= 5.0:
                from backend.database import SessionLocal, QueueStat
                db = SessionLocal()
                try:
                    stat_record = QueueStat(
                        queue_length=stats["queue_count"],
                        avg_wait_time=stats["avg_wait_time"],
                        max_wait_time=stats["max_wait_time"],
                        predicted_wait_time=stats["predicted_wait_time"],
                        arrival_rate=stats["arrival_rate"],
                        service_rate=stats["service_rate"],
                        abandonment_rate=stats["abandonment_rate"]
                    )
                    db.add(stat_record)
                    db.commit()
                    last_db_log_time = time.time()
                except Exception as db_err:
                    print(f"[DB] Error logging queue stat: {db_err}")
                finally:
                    db.close()

            persons = []
            for t in active_tracks:
                persons.append({
                    "id": t.track_id,
                    "bbox": t.tlwh.tolist(),
                    "in_queue": queue_info["person_queue_map"].get(t.track_id, False),
                    "wait_time": round(analytics_data.get("person_wait_times", {}).get(t.track_id, 0), 1),
                    "velocity": round(t.velocity, 2),
                })

            # Update shared state
            self.state.update(frame_b64, stats, persons, alerts, heatmap_b64)

            # Broadcast via WebSocket
            self._broadcast_ws(frame_b64, stats, persons, alerts)

            # Frame rate control
            elapsed = time.time() - t0
            target = 1.0 / self.config.target_fps
            if elapsed < target:
                time.sleep(target - elapsed)

        cap.release()
        self.state.running = False
        print("[Pipeline] Stopped")

    def _draw(self, frame, tracks, queue_info, analytics, prediction):
        """Draw annotations on frame."""
        out = frame.copy()
        h, w = frame.shape[:2]

        # Draw ROI
        if self.config.rois:
            for roi in self.config.rois:
                if len(roi.points) >= 3:
                    pts = np.array(roi.points, dtype=np.int32)
                    cv2.polylines(out, [pts], True, (0, 255, 255), 2)
                    cv2.putText(out, roi.name, tuple(pts[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            rx1, ry1 = int(w * 0.1), int(h * 0.1)
            rx2, ry2 = int(w * 0.9), int(h * 0.9)
            overlay = out.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 255), -1)
            cv2.addWeighted(overlay, 0.05, out, 0.95, 0, out)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(out, "Queue Zone", (rx1 + 5, ry1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw persons
        for t in tracks:
            x, y, bw, bh = t.tlwh.astype(int)
            tid = t.track_id
            in_q = queue_info["person_queue_map"].get(tid, False)
            color = (0, 255, 100) if in_q else (255, 180, 0)

            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)

            label = f"ID:{tid}"
            if in_q:
                wt = analytics.get("person_wait_times", {}).get(tid, 0)
                label += f" {wt:.0f}s"

            (tw, th2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x, y - th2 - 8), (x + tw + 4, y), color, -1)
            cv2.putText(out, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Stats overlay
        info = [
            f"Queue: {queue_info['total_in_queue']}",
            f"Pred Wait: {prediction:.0f}s",
            f"FPS: {sum(self._fps_history)/max(len(self._fps_history),1):.1f}",
        ]
        for i, text in enumerate(info):
            y_pos = 30 + i * 30
            cv2.putText(out, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return out

    def _broadcast_ws(self, frame_b64, stats, persons, alerts):
        """Broadcast data to all connected WebSocket clients."""
        if not self._loop:
            return
        with self._ws_lock:
            clients = list(self.ws_clients)
        if not clients:
            return

        msg = {
            "type": "frame_data",
            "frame": frame_b64,
            "stats": stats,
            "persons": persons,
            "alerts": alerts,
        }

        import json
        data = json.dumps(msg)

        async def _send():
            dead = []
            for ws in clients:
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.append(ws)
            if dead:
                with self._ws_lock:
                    for ws in dead:
                        self.ws_clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)
        except Exception:
            pass

    def add_ws_client(self, ws):
        with self._ws_lock:
            self.ws_clients.add(ws)

    def remove_ws_client(self, ws):
        with self._ws_lock:
            self.ws_clients.discard(ws)
