from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from pitchlens.analytics.pressing import (
    drop_frames_without_carrier,
    summarise_frames,
    summarise_matches,
)


def test_summarise_matches():
    frames = pl.DataFrame(
        {
            "match_id": ["m"] * 4,
            "pressing_team_id": ["A", "A", "B", "B"],
            "pressing_team_name": ["Team A", "Team A", "Team B", "Team B"],
            "pressure_on_carrier": [0.2, 0.8, float("nan"), 0.4],
            "n_active_pressers": [1, 3, 0, 2],
        }
    )
    t = summarise_matches(frames, fps=1.0)
    a = t.filter(pl.col("pressing_team_name") == "Team A").row(0, named=True)
    b = t.filter(pl.col("pressing_team_name") == "Team B").row(0, named=True)
    assert a["mean_pressure"] == pytest.approx(0.5)
    assert a["pct_high_pressure"] == pytest.approx(50.0)
    assert b["mean_pressure"] == pytest.approx(0.4)  # NaN ignored
    assert t["pressing_team_name"].to_list()[0] == "Team A"  # sorted by pressure


def test_drop_frames_without_carrier():
    tracking = pl.DataFrame(
        {
            "period_id": [1, 1, 1, 1],
            "frame_id": [1, 1, 2, 2],
            "id": ["a1", "d1", "a1", "d1"],
            "team_id": ["A", "D", "A", "D"],
            "ball_owning_team_id": ["A", "A", "A", "A"],
            # frame 2: carrier flag sits on the defending team -> invalid
            "is_ball_carrier": [True, False, False, True],
        }
    )
    out, n_dropped = drop_frames_without_carrier(tracking)
    assert n_dropped == 1
    assert out["frame_id"].unique().to_list() == [1]


def test_summarise_frames_synthetic():
    # 2 defenders (rows) x 2 attackers (columns); attacker "a1" carries the ball.
    pti = pl.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "frame_id": [10],
            "timestamp": [timedelta(seconds=4)],
            "time_to_intercept": [[[1.0, 2.0], [3.0, 4.0]]],
            "probability_to_intercept": [[[0.6, 0.1], [0.5, 0.0]]],
            "columns": [["a1", "a2"]],
            "rows": [["d1", "d2"]],
        }
    )
    tracking = pl.DataFrame(
        {
            "period_id": [1, 1, 1, 1],
            "frame_id": [10, 10, 10, 10],
            "id": ["a1", "a2", "d1", "d2"],
            "team_id": ["A", "A", "D", "D"],
            "is_ball_carrier": [True, False, False, False],
        }
    )
    out = summarise_frames(pti, tracking, active_threshold=0.5)
    row = out.row(0, named=True)
    assert row["pressing_team_id"] == "D"
    assert row["timestamp_s"] == pytest.approx(4.0)
    assert row["pressure_total"] == pytest.approx(1.2)
    # 1 - (1-0.6)(1-0.5) = 0.8
    assert row["pressure_on_carrier"] == pytest.approx(0.8)
    assert row["n_active_pressers"] == 1


def test_transposed_orientation_is_normalised():
    """If the library returns rows = attackers, the summary must be identical."""
    pti = pl.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "frame_id": [10],
            "timestamp": [timedelta(seconds=4)],
            "time_to_intercept": [[[1.0, 3.0], [2.0, 4.0]]],
            "probability_to_intercept": [[[0.6, 0.5], [0.1, 0.0]]],
            "columns": [["d1", "d2"]],
            "rows": [["a1", "a2"]],
        }
    )
    tracking = pl.DataFrame(
        {
            "period_id": [1, 1, 1, 1],
            "frame_id": [10, 10, 10, 10],
            "id": ["a1", "a2", "d1", "d2"],
            "team_id": ["A", "A", "D", "D"],
            "is_ball_carrier": [True, False, False, False],
        }
    )
    row = summarise_frames(pti, tracking).row(0, named=True)
    assert row["pressing_team_id"] == "D"
    assert row["pressure_on_carrier"] == pytest.approx(0.8)
    assert row["n_active_pressers"] == 1


def test_no_carrier_gives_nan():
    pti = pl.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "frame_id": [1],
            "timestamp": [timedelta(seconds=0)],
            "time_to_intercept": [[[1.0]]],
            "probability_to_intercept": [[[0.3]]],
            "columns": [["a1"]],
            "rows": [["d1"]],
        }
    )
    tracking = pl.DataFrame(
        {
            "period_id": [1, 1],
            "frame_id": [1, 1],
            "id": ["a1", "d1"],
            "team_id": ["A", "D"],
            "is_ball_carrier": [False, False],
        }
    )
    row = summarise_frames(pti, tracking).row(0, named=True)
    assert np.isnan(row["pressure_on_carrier"])
