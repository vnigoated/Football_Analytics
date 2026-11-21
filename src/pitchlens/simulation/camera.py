"""Virtual broadcast camera.

A window that follows the ball along the pitch length and is clipped to the pitch.
Players outside the window are treated as "off-screen". Three optional effects make
it closer to a real main camera (all off by default = the Experiment 1 camera):

- ``lag_s``: the camera follows the ball with an exponential delay (seconds).
- ``perspective``: the camera stands on the near touchline (y = 0), so it sees a
  shorter stretch of the near touchline and a longer stretch of the far touchline.
  The visible half-length at height y is ``h * (1 + perspective * (2y/W - 1))``,
  so the average over the width stays ``view_length``.
- ``zoom_gain``: the camera zooms out when the ball moves fast (long passes,
  counter-attacks): ``view_length * (1 + zoom_gain * min(speed / zoom_speed_ref, 1))``,
  with ball speed smoothed by the same lag.

These are assumptions to be fitted to real broadcast visibility (SkillCorner) next.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BroadcastCamera:
    view_length: float = 40.0  # metres of pitch length visible (at mid-width)
    view_width: float = 68.0  # metres of pitch width visible (full width by default)
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    smoothing: float = 0.0  # per-frame lag factor; used only when lag_s == 0
    lag_s: float = 0.0  # camera lag time constant in seconds
    perspective: float = 0.0  # 0 = rectangle; 0.3 = near side 30% shorter, far side 30% longer
    zoom_gain: float = 0.0  # 0 = fixed zoom; 0.5 = up to 50% wider view for fast balls
    zoom_speed_ref: float = 15.0  # ball speed (m/s) at which the zoom-out is complete
    fps: float = 25.0

    @property
    def alpha(self) -> float:
        """Per-frame smoothing factor (0 = camera jumps to the ball)."""
        if self.lag_s > 0:
            return float(np.exp(-1.0 / (self.fps * self.lag_s)))
        return self.smoothing

    def window(self, ball_x: float, ball_y: float) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) of the visible area, centred on the ball and clipped."""
        hl, hw = self.view_length / 2, self.view_width / 2
        cx = float(np.clip(ball_x, hl, self.pitch_length - hl))
        cy = float(np.clip(ball_y, hw, self.pitch_width - hw))
        return cx - hl, cx + hl, cy - hw, cy + hw

    def visible(self, xy: np.ndarray, ball_xy: tuple[float, float]) -> np.ndarray:
        """Boolean mask over (N, 2) player positions for a single frame (rectangle, no lag)."""
        x0, x1, y0, y1 = self.window(*ball_xy)
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)

    def track_ball(self, ball_xy: np.ndarray) -> np.ndarray:
        """Smoothed camera centre over a (T, 2) ball trajectory (exponential lag)."""
        return _ema(np.asarray(ball_xy, dtype=np.float64), self.alpha)

    def path(self, ball_xy: np.ndarray) -> dict[str, np.ndarray]:
        """Camera state per frame for a (T, 2) ball trajectory in pitch coordinates.

        Returns ``cx``, ``cy`` (clipped centre), ``half_length`` (at mid-width, after zoom)
        and ``half_width``.
        """
        ball_xy = np.asarray(ball_xy, dtype=np.float64)
        centre = self.track_ball(ball_xy)
        view = np.full(len(ball_xy), float(self.view_length))
        if self.zoom_gain > 0 and len(ball_xy) > 1:
            speed = np.r_[0.0, np.linalg.norm(np.diff(ball_xy, axis=0), axis=1) * self.fps]
            speed = _ema(speed[:, None], self.alpha)[:, 0]
            view *= 1 + self.zoom_gain * np.clip(speed / self.zoom_speed_ref, 0, 1)
        hl = np.minimum(view, self.pitch_length) / 2
        hw = min(self.view_width, self.pitch_width) / 2
        cx = np.clip(centre[:, 0], hl, self.pitch_length - hl)
        cy = np.clip(centre[:, 1], hw, self.pitch_width - hw)
        return {"cx": cx, "cy": cy, "half_length": hl, "half_width": np.full_like(cx, hw)}


def _ema(x: np.ndarray, a: float) -> np.ndarray:
    out = np.empty_like(x)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = a * out[t - 1] + (1 - a) * x[t]
    return out
