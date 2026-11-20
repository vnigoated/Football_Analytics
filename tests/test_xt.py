import numpy as np

from pitchlens.analytics.xt import ExpectedThreat


def _synthetic_actions(n=20000, seed=0):
    """Teams move the ball forward; shots happen near goal and score more when closer."""
    rng = np.random.default_rng(seed)
    sx = rng.uniform(0, 105, n)
    sy = rng.uniform(0, 68, n)
    dist_to_goal = np.hypot(105 - sx, 34 - sy)
    is_shot = (dist_to_goal < 25) & (rng.random(n) < 0.3)
    ex = np.clip(sx + rng.normal(10, 8, n), 0, 105)
    ey = np.clip(sy + rng.normal(0, 8, n), 0, 68)
    success = rng.random(n) < 0.8
    p_goal = np.clip(0.4 - dist_to_goal / 60, 0.01, None)
    is_goal = is_shot & (rng.random(n) < p_goal)
    return sx, sy, ex, ey, is_shot, success, is_goal


def test_xt_shape_and_range():
    xt = ExpectedThreat().fit(*_synthetic_actions())
    assert xt.grid.shape == (12, 16)
    assert (xt.grid >= 0).all() and (xt.grid <= 1).all()


def test_xt_higher_near_opponent_goal():
    xt = ExpectedThreat().fit(*_synthetic_actions())
    own_third = xt.grid[:, :5].mean()
    final_third = xt.grid[:, -5:].mean()
    assert final_third > 3 * own_third


def test_rate_forward_move_positive_and_failed_zero():
    xt = ExpectedThreat().fit(*_synthetic_actions())
    r = xt.rate(
        start_x=np.array([40.0, 40.0]),
        start_y=np.array([34.0, 34.0]),
        end_x=np.array([95.0, 95.0]),
        end_y=np.array([34.0, 34.0]),
        success=np.array([True, False]),
    )
    assert r[0] > 0
    assert r[1] == 0
