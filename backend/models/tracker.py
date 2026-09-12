"""
ByteTrack Multi-Object Tracker
Based on: https://arxiv.org/abs/2110.06864

Key Innovation: Two-stage association using BOTH high and low confidence
detections. Low-confidence detections (partially occluded people) are matched
with tracks that weren't matched by high-confidence detections, dramatically
improving tracking in crowded scenes.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import time


class KalmanFilter:
    """
    Kalman Filter for bounding box tracking.
    State: [cx, cy, a, h, vx, vy, va, vh]
    Measurement: [cx, cy, a, h]
    where (cx,cy)=center, a=aspect ratio, h=height
    """

    def __init__(self):
        self.ndim = 4
        self.dt = 1.0

        # State transition (constant velocity model)
        self._motion_mat = np.eye(2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt

        # Measurement matrix
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Initialize track from first measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Predict next state."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Update state with new measurement."""
        projected_mean = self._update_mat @ mean

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov

        kalman_gain = covariance @ self._update_mat.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean

        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Single object track with Kalman Filter state and timing info."""

    _count = 0

    def __init__(self, tlwh, score):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.score = score
        self.kalman_filter = KalmanFilter()
        self.mean = None
        self.covariance = None
        self.track_id = 0
        self.state = TrackState.New
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.entry_time = None
        self.last_seen_time = None
        self.exit_time = None
        self.position_history = []

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id():
        STrack._count = 0

    def activate(self, frame_id):
        """Start a new track."""
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self._to_xyah(self.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0
        self.entry_time = time.time()
        self.last_seen_time = self.entry_time
        self.position_history = [self.tlwh[:2].copy()]

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.tlwh = new_track.tlwh
        self.last_seen_time = time.time()
        if new_id:
            self.track_id = self.next_id()
        self.position_history.append(self.tlwh[:2].copy())

    def predict(self):
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def update(self, new_track, frame_id):
        """Update with matched detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.tlwh = new_track.tlwh
        self.score = new_track.score
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self._to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.last_seen_time = time.time()
        self.position_history.append(self.tlwh[:2].copy())
        if len(self.position_history) > 60:
            self.position_history = self.position_history[-60:]

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
        self.exit_time = time.time()

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def velocity(self):
        if len(self.position_history) < 2:
            return 0.0
        pts = np.array(self.position_history[-10:])
        disps = np.diff(pts, axis=0)
        speeds = np.linalg.norm(disps, axis=1)
        return float(np.mean(speeds))

    @property
    def time_alive(self):
        if self.entry_time is None:
            return 0.0
        return (self.last_seen_time or time.time()) - self.entry_time

    @staticmethod
    def _to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[2:] += ret[:2]
        return ret


def _iou_batch(atlbrs, btlbrs):
    """Compute IoU matrix between two sets of boxes."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.empty((len(atlbrs), len(btlbrs)))
    a = np.asarray(atlbrs)
    b = np.asarray(btlbrs)
    ious = np.zeros((len(a), len(b)), dtype=np.float64)
    for i in range(len(a)):
        xx1 = np.maximum(a[i, 0], b[:, 0])
        yy1 = np.maximum(a[i, 1], b[:, 1])
        xx2 = np.minimum(a[i, 2], b[:, 2])
        yy2 = np.minimum(a[i, 3], b[:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_a = (a[i, 2] - a[i, 0]) * (a[i, 3] - a[i, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a + area_b - inter
        ious[i] = np.where(union > 0, inter / union, 0)
    return ious


def _linear_assignment(cost_matrix, thresh):
    """Solve linear assignment and filter by threshold."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_a, unmatched_b = [], [], []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)
    unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
    unmatched_b = [i for i in range(cost_matrix.shape[1]) if i not in matched_cols]
    return matches, unmatched_a, unmatched_b


class BYTETracker:
    """
    ByteTrack multi-object tracker.
    Two-stage association: high-conf detections first, then low-conf.
    """

    def __init__(self, track_high_thresh=0.5, track_low_thresh=0.1,
                 new_track_thresh=0.6, track_buffer=30, match_thresh=0.8):
        self.high_thresh = track_high_thresh
        self.low_thresh = track_low_thresh
        self.new_thresh = new_track_thresh
        self.buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.all_tracks = {}

    def update(self, detections: np.ndarray):
        """
        Process new detections and return active tracks.
        Args: detections (N,5) [x1,y1,x2,y2,score]
        Returns: list of active STrack
        """
        self.frame_id += 1
        activated, refound, lost, removed = [], [], [], []

        # Parse detections into STrack objects
        if len(detections) > 0:
            scores = detections[:, 4]
            bboxes = detections[:, :4]
            tlwhs = bboxes.copy()
            tlwhs[:, 2:] -= tlwhs[:, :2]  # xyxy -> tlwh

            high_mask = scores > self.high_thresh
            low_mask = (scores > self.low_thresh) & (~high_mask)
            dets_high = [STrack(tlwhs[i], scores[i]) for i in np.where(high_mask)[0]]
            dets_low = [STrack(tlwhs[i], scores[i]) for i in np.where(low_mask)[0]]
        else:
            dets_high, dets_low = [], []

        # Separate tracked into confirmed and unconfirmed
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        confirmed = [t for t in self.tracked_stracks if t.is_activated]

        strack_pool = confirmed + self.lost_stracks
        for t in strack_pool:
            t.predict()

        # === First Association: high-conf detections ===
        if strack_pool and dets_high:
            ious = _iou_batch([t.tlbr for t in strack_pool],
                              [STrack.tlwh_to_tlbr(d.tlwh) for d in dets_high])
            cost = 1 - ious
            matches, um_tracks, um_dets = _linear_assignment(cost, 0.7)
            for r, c in matches:
                track = strack_pool[r]
                det = dets_high[c]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated.append(track)
                else:
                    track.re_activate(det, self.frame_id)
                    refound.append(track)
        else:
            um_tracks = list(range(len(strack_pool)))
            um_dets = list(range(len(dets_high)))

        r_tracked = [strack_pool[i] for i in um_tracks if strack_pool[i].state == TrackState.Tracked]

        # === Second Association: low-conf detections with remaining tracked ===
        if r_tracked and dets_low:
            ious = _iou_batch([t.tlbr for t in r_tracked],
                              [STrack.tlwh_to_tlbr(d.tlwh) for d in dets_low])
            cost = 1 - ious
            matches2, um_tracks2, _ = _linear_assignment(cost, 0.5)
            for r, c in matches2:
                r_tracked[r].update(dets_low[c], self.frame_id)
                activated.append(r_tracked[r])
            for i in um_tracks2:
                if r_tracked[i].state != TrackState.Lost:
                    r_tracked[i].mark_lost()
                    lost.append(r_tracked[i])
        else:
            for t in r_tracked:
                if t.state != TrackState.Lost:
                    t.mark_lost()
                    lost.append(t)

        # Handle unconfirmed tracks
        if unconfirmed and um_dets:
            rem_dets = [dets_high[i] for i in um_dets]
            if rem_dets:
                ious = _iou_batch([t.tlbr for t in unconfirmed],
                                  [STrack.tlwh_to_tlbr(d.tlwh) for d in rem_dets])
                cost = 1 - ious
                matches3, um_unc, um_det3 = _linear_assignment(cost, 0.7)
                for r, c in matches3:
                    unconfirmed[r].update(rem_dets[c], self.frame_id)
                    activated.append(unconfirmed[r])
                for i in um_unc:
                    unconfirmed[i].mark_removed()
                    removed.append(unconfirmed[i])
                # Update remaining unmatched detections
                um_dets = [um_dets[i] for i in um_det3]
            else:
                for t in unconfirmed:
                    t.mark_removed()
                    removed.append(t)
        else:
            for t in unconfirmed:
                t.mark_removed()
                removed.append(t)

        # Initialize new tracks from unmatched high-conf detections
        for i in um_dets:
            det = dets_high[i]
            if det.score >= self.new_thresh:
                det.activate(self.frame_id)
                activated.append(det)
                self.all_tracks[det.track_id] = det

        # Remove stale lost tracks
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.buffer:
                t.mark_removed()
                removed.append(t)

        # Update lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self._merge(self.tracked_stracks, activated)
        self.tracked_stracks = self._merge(self.tracked_stracks, refound)
        self.lost_stracks = self._sub(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = self._sub(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)

        for t in activated + refound:
            self.all_tracks[t.track_id] = t

        return [t for t in self.tracked_stracks if t.is_activated]

    def get_all_track_info(self):
        return dict(self.all_tracks)

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.all_tracks = {}
        STrack.reset_id()

    @staticmethod
    def _merge(a, b):
        ids = {t.track_id for t in a}
        result = list(a)
        for t in b:
            if t.track_id not in ids:
                ids.add(t.track_id)
                result.append(t)
        return result

    @staticmethod
    def _sub(a, b):
        ids = {t.track_id for t in b}
        return [t for t in a if t.track_id not in ids]
