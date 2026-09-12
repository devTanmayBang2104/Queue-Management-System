"""
Heatmap & Flow Analysis
Generates density heatmaps showing high-traffic areas and movement patterns.
Accumulates person positions over time and applies Gaussian blur for visualization.
"""

import numpy as np
import cv2


class HeatmapGenerator:
    """Generates heatmaps from tracked person positions."""

    def __init__(self, width: int = 1280, height: int = 720, decay: float = 0.995):
        """
        Args:
            width: Frame width
            height: Frame height
            decay: Decay factor per frame (0.99 = slow fade, 0.9 = fast fade)
        """
        self.width = width
        self.height = height
        self.decay = decay
        # Accumulation buffer (downscaled for performance)
        self.scale = 4
        self.acc_w = width // self.scale
        self.acc_h = height // self.scale
        self.accumulator = np.zeros((self.acc_h, self.acc_w), dtype=np.float64)

    def update(self, tracks):
        """Add person positions from current frame to accumulator."""
        # Decay existing heat
        self.accumulator *= self.decay

        for track in tracks:
            tx, ty, tw, th = track.tlwh
            # Use bottom-center as position
            cx = int((tx + tw / 2) / self.scale)
            cy = int((ty + th) / self.scale)

            # Clamp to bounds
            cx = max(0, min(cx, self.acc_w - 1))
            cy = max(0, min(cy, self.acc_h - 1))

            # Add Gaussian splash at person position
            radius = max(int(tw / self.scale / 2), 3)
            y_min = max(0, cy - radius)
            y_max = min(self.acc_h, cy + radius + 1)
            x_min = max(0, cx - radius)
            x_max = min(self.acc_w, cx + radius + 1)

            for py in range(y_min, y_max):
                for px in range(x_min, x_max):
                    dist = np.sqrt((px - cx)**2 + (py - cy)**2)
                    if dist <= radius:
                        self.accumulator[py, px] += np.exp(-dist**2 / (2 * (radius/2)**2))

    def generate(self) -> np.ndarray:
        """
        Generate a colored heatmap image.

        Returns:
            BGR image (H, W, 3) at original resolution
        """
        # Normalize to 0-255
        heat = self.accumulator.copy()
        max_val = heat.max()
        if max_val > 0:
            heat = (heat / max_val * 255).astype(np.uint8)
        else:
            heat = np.zeros_like(heat, dtype=np.uint8)

        # Apply Gaussian blur for smooth visualization
        heat = cv2.GaussianBlur(heat, (15, 15), 0)

        # Apply colormap (JET gives classic heatmap look)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        # Resize to original resolution
        colored = cv2.resize(colored, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        return colored

    def generate_overlay(self, frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Generate heatmap overlaid on the original frame."""
        heatmap = self.generate()
        return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

    def reset(self):
        self.accumulator = np.zeros((self.acc_h, self.acc_w), dtype=np.float64)
