import numpy as np
import polars as pl
import pytest

from pitchlens.simulation.camera_fit import camera_position, fit_wedge_camera


def _synthetic_frames(cam=(2.0, -60.0), n=600, seed=0):
    """Footprints of a camera that pans with the ball (gain 0.8) and zooms with distance."""
    rng = np.random.default_rng(seed)
    cx, cy = cam
    ball = np.c_[np.cumsum(rng.normal(0, 0.8, n)).clip(-50, 50), rng.uniform(-30, 30, n)]
    bearing = np.arctan2(ball[:, 0] - cx, ball[:, 1] - cy)
    dist = np.hypot(ball[:, 0] - cx, ball[:, 1] - cy)
    pan = 0.8 * bearing
    fov = 0.9 - 0.006 * dist
    near = 10 + 0.5 * dist

    def on_ray(angle, y):  # point on the ray from the camera at height y
        return cx + (y - cy) * np.tan(angle), y

    rows = {}
    for side, a in (("left", pan - fov / 2), ("right", pan + fov / 2)):
        rows[f"x_top_{side}"], rows[f"y_top_{side}"] = on_ray(a, np.full(n, 39.0))
        y_near = cy + near / np.cos(a - pan) * np.cos(a)
        rows[f"x_bottom_{side}"], rows[f"y_bottom_{side}"] = on_ray(a, y_near)
    return pl.DataFrame(
        {
            "frame": np.arange(n),
            "period": np.ones(n, dtype=int),
            "ball_x": ball[:, 0],
            "ball_y": ball[:, 1],
            "ball_detected": np.ones(n, dtype=bool),
            **rows,
        }
    )


def test_camera_position_is_recovered():
    x, y = camera_position(_synthetic_frames())
    assert x == pytest.approx(2.0, abs=0.01)
    assert y == pytest.approx(-60.0, abs=0.01)


def test_dynamics_are_recovered():
    cam, report = fit_wedge_camera([_synthetic_frames()])
    assert cam.pan_gain == pytest.approx(0.8, abs=0.01)
    assert cam.fov_b == pytest.approx(-0.006, abs=1e-4)
    assert cam.near_b == pytest.approx(0.5, abs=0.01)
    assert cam.lag_s == 0.0
    assert report["r2_fov"] > 0.99
