import numpy as np
import polars as pl
import pytest

from pitchlens.analytics.correction import evaluate_correction
from pitchlens.analytics.propagation import (
    clip_errors,
    constant,
    correlated,
    independent,
    propagate,
    summarise,
    zero,
)


def phase1_table(seed=0):
    """Synthetic broadcasts whose bias is linear in visible players (4 pp per player) plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(5):
        for m in range(4):
            v = 12 + c + rng.normal(0, 0.3)
            for team in ("A", "B"):
                gt = rng.uniform(1.0, 1.4)
                bias = 4 * v - 80 + rng.normal(0, 2)
                rows.append({"camera": f"c{c}", "match_id": f"m{m}", "pressing_team_name": team, "gt": gt,
                             "sim": gt * (1 + bias / 100), "players_visible": v, "bias": bias})
    return pl.DataFrame(rows)


def test_zero_error_reproduces_phase1_correction():
    t = phase1_table()
    res, _ = evaluate_correction(t)
    ours = propagate(t, zero(), n_draws=3)
    expected = res.sort("camera", "match_id", "pressing_team_name")["error_corrected"].to_numpy()
    got = pl.DataFrame({"camera": t["camera"], "match_id": t["match_id"], "team": t["pressing_team_name"],
                        "e": ours[0]}).sort("camera", "match_id", "team")["e"].to_numpy()
    assert np.allclose(got, expected)
    assert np.allclose(ours[0], ours[2])


def test_count_error_moves_corrected_value_by_slope():
    t = phase1_table()
    base = propagate(t, zero(), n_draws=1)[0]
    up = propagate(t, constant(1.0), n_draws=1)[0]
    # one player too many -> predicted bias about 4 pp less negative -> corrected value about 4% lower
    assert np.mean(up - base) == pytest.approx(-4 / 0.85, rel=0.25)


def test_error_models_and_summary():
    rng = np.random.default_rng(0)
    errs = np.array([-1.0, 0.0, 1.0])
    assert set(np.unique(correlated(errs)(rng, 100))) <= {-1.0, 0.0, 1.0}
    ind = independent(errs, n_clips=180)(rng, 1000)
    assert abs(ind.mean()) < 0.05 and ind.std() < 0.1
    s = summarise(np.array([[-2.0, 4.0], [6.0, -12.0]]))
    assert s["mean_abs_error"] == 6.0 and s["within_5pct"] == 0.5 and s["within_10pct"] == 0.75


def test_clip_errors_with_calibration():
    counts = pl.DataFrame({"sequence": ["a", "a", "b", "b", "c"], "part": ["held out"] * 4 + ["tune"],
                           "gt": [10, 12, 14, 16, 9], "ours": [9, 11, 13, 15, 9]})
    assert sorted(clip_errors(counts).tolist()) == [-1.0, -1.0]
    assert np.allclose(clip_errors(counts, calibration=(1.0, 1.0)), 0.0)
