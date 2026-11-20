"""Virtual broadcast camera (first version).

A rectangular window that follows the ball along the pitch length and is clipped to
the pitch. Players outside the window are treated as "off-screen".

This is a deliberately simple baseline. Next steps (RESEARCH.md, section 3):
- fit window size, lag and zoom to SkillCorner visibility statistics;
- trapezoid footprint (perspective) instead of a rectangle;
- reproduce the camera settings of arXiv 2607.11548 for the R1 replication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BroadcastCamera:
    view_length: float = 40.0  # metres of pitch length visible
    view_width: float = 68.0  # metres of pitch width visible (full width by default)
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    smoothing: float = 0.0  # 0 = camera jumps to ball; closer to 1 = lazier camera

    def window(self, ball_x: float, ball_y: float) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) of the visible area, centred on the ball and clipped."""
        hl, hw = self.view_length / 2, self.view_width / 2
        cx = float(np.clip(ball_x, hl, self.pitch_length - hl))
        cy = float(np.clip(ball_y, hw, self.pitch_width - hw))
        return cx - hl, cx + hl, cy - hw, cy + hw

    def visible(self, xy: np.ndarray, ball_xy: tuple[float, float]) -> np.ndarray:
        """Boolean mask over (N, 2) player positions for a single frame."""
        x0, x1, y0, y1 = self.window(*ball_xy)
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)

    def track_ball(self, ball_xy: np.ndarray) -> np.ndarray:
        """Smoothed camera centre over a (T, 2) ball trajectory (simple exponential lag)."""
        ball_xy = np.asarray(ball_xy, dtype=np.float64)
        out = np.empty_like(ball_xy)
        out[0] = ball_xy[0]
        a = self.smoothing
        for t in range(1, len(ball_xy)):
            out[t] = a * out[t - 1] + (1 - a) * ball_xy[t]
        return out
