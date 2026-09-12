"""
Centralized Configuration for the Queue Management System.
All thresholds, model settings, and ROI definitions live here.
"""

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


class ROIConfig(BaseModel):
    """Region of Interest definition for queue zones."""
    id: str = "default"
    name: str = "Queue Zone"
    points: List[List[int]] = Field(default_factory=list)  # Polygon points [[x,y], ...]


class QueueConfig(BaseModel):
    """Queue alert thresholds."""
    max_queue_length_alert: int = 10
    max_wait_time_alert: float = 300.0  # seconds
    crowd_spike_threshold: int = 15
    abandonment_time_threshold: float = 5.0  # seconds of no movement = queue member
    velocity_threshold: float = 5.0  # pixels/frame - below this = stationary


class DetectorConfig(BaseModel):
    """YOLOv8 detector settings."""
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    person_class_id: int = 0


class TrackerConfig(BaseModel):
    """ByteTrack tracker settings."""
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8


class AppConfig(BaseModel):
    """Main application configuration."""
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    rois: List[ROIConfig] = Field(default_factory=list)
    video_source: str = "0"
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30


# Global config instance
config = AppConfig()
