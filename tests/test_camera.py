import numpy as np

from pitchlens.simulation.camera import BroadcastCamera


def test_window_clipped_to_pitch():
    cam = BroadcastCamera(view_length=40)
    assert cam.window(2.0, 34.0)[:2] == (0.0, 40.0)
    assert cam.window(104.0, 34.0)[:2] == (65.0, 105.0)


def test_visibility_mask():
    cam = BroadcastCamera(view_length=40)
    players = np.array([[50.0, 10.0], [52.0, 60.0], [10.0, 34.0], [95.0, 34.0]])
    vis = cam.visible(players, ball_xy=(52.5, 34.0))
    assert vis.tolist() == [True, True, False, False]


def test_smoothing_lags_ball():
    ball = np.array([[10.0, 34.0], [90.0, 34.0], [90.0, 34.0]])
    assert np.allclose(BroadcastCamera(smoothing=0.0).track_ball(ball), ball)
    lagged = BroadcastCamera(smoothing=0.5).track_ball(ball)
    assert 10.0 < lagged[1, 0] < 90.0
