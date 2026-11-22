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


@dataclass
class WedgeCamera:
    """Main broadcast camera as a fixed point that pans, zooms and tilts.

    Coordinates are centred on the pitch (metres, x along the length). The camera
    stands at ``(cam_x, cam_y)`` behind the near touchline. A player is visible when
    their bearing from the camera lies within ``pan +- fov/2`` and they are beyond the
    bottom image edge (``near``: distance along the viewing direction).

    - pan  = EMA_lag(pan_gain * ball_bearing + pan_gain_players * players_bearing
             + pan_offset) + noise
    - fov  = clip(fov_a + fov_b * ball_distance + fov_c * players_spread) + noise
    - near = near_a + near_b * ball_distance + noise (always in front of the ball)

    Noise is an AR(1) process with time constant ``noise_tau_s`` (``seed=None`` turns it
    off). Default values are placeholders; fit them with ``pitchlens fit-camera``.
    Angles are in radians.
    """

    cam_x: float = 0.0
    cam_y: float = -57.0
    pan_gain: float = 0.9
    pan_gain_players: float = 0.0  # weight of the players' bearing (operator frames play)
    pan_offset: float = 0.0
    lag_s: float = 0.2
    fov_a: float = 0.8
    fov_b: float = -0.005
    fov_c: float = 0.0  # zoom out when players are spread (std of their bearings, rad)
    fov_min: float = 0.3
    fov_max: float = 0.8
    near_a: float = 0.0
    near_b: float = 0.6
    pan_noise_sd: float = 0.0
    fov_noise_sd: float = 0.0
    near_noise_sd: float = 0.0
    noise_tau_s: float = 5.0
    fps: float = 25.0
    seed: int | None = None
    pitch_length: float = 105.0
    pitch_width: float = 68.0

    def _noise(self, n: int, sd: float, rng) -> np.ndarray:
        if rng is None or sd <= 0 or n == 0:
            return np.zeros(n)
        rho = float(np.exp(-1.0 / (self.fps * self.noise_tau_s)))
        e = rng.standard_normal(n) * sd * np.sqrt(1 - rho**2)
        out = np.empty(n)
        out[0] = rng.standard_normal() * sd
        for t in range(1, n):
            out[t] = rho * out[t - 1] + e[t]
        return out

    def path(
        self,
        ball_xy: np.ndarray,
        rng=None,
        players_bearing: np.ndarray | None = None,
        players_spread: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-frame ``pan``, ``half_fov`` and ``near`` for a (T, 2) centred ball trajectory.

        ``players_bearing`` / ``players_spread``: per-frame mean and std of the players'
        bearings from the camera (used when ``pan_gain_players`` / ``fov_c`` are set).
        """
        ball_xy = np.asarray(ball_xy, dtype=np.float64)
        dx, dy = ball_xy[:, 0] - self.cam_x, ball_xy[:, 1] - self.cam_y
        bearing, dist = np.arctan2(dx, dy), np.hypot(dx, dy)
        alpha = float(np.exp(-1.0 / (self.fps * self.lag_s))) if self.lag_s > 0 else 0.0
        n = len(ball_xy)
        pb = bearing if players_bearing is None else np.nan_to_num(players_bearing, nan=0.0)
        sp = np.zeros(n) if players_spread is None else np.nan_to_num(players_spread, nan=0.0)
        target = self.pan_gain * bearing + self.pan_gain_players * pb + self.pan_offset
        pan = _ema(target[:, None], alpha)[:, 0]
        pan += self._noise(n, self.pan_noise_sd, rng)
        fov = self.fov_a + self.fov_b * dist + self.fov_c * sp
        fov = np.clip(fov, self.fov_min, self.fov_max)
        fov = np.clip(fov + self._noise(n, self.fov_noise_sd, rng), 0.05, np.pi / 2)
        near = self.near_a + self.near_b * dist + self._noise(n, self.near_noise_sd, rng)
        near = np.minimum(near, dist * np.cos(bearing - pan) - 1.0)  # keep the ball on screen
        return {"pan": pan, "half_fov": fov / 2, "near": near}

    def visible(self, xy: np.ndarray, pan: float, half_fov: float, near: float) -> np.ndarray:
        """Boolean mask over (N, 2) centred positions for one frame."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        dx, dy = xy[:, 0] - self.cam_x, xy[:, 1] - self.cam_y
        rel = (np.arctan2(dx, dy) - pan + np.pi) % (2 * np.pi) - np.pi
        depth = dx * np.sin(pan) + dy * np.cos(pan)
        return (np.abs(rel) <= half_fov) & (depth >= near)
