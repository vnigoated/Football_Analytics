import numpy as np
import polars as pl
import pytest

from pitchlens.analytics.robustness import (
    balanced_mean,
    correction_team_unseen,
    leave_one_team_out,
    per_team,
    team_effect,
    team_matches,
)


def table(team_extra=0.0, seed=0):
    """6 matches; team 'F' plays in 4 of them. bias = 4 * visible - 80 (+ team_extra for F) + noise."""
    rng = np.random.default_rng(seed)
    pairs = [("F", "A"), ("F", "B"), ("F", "C"), ("F", "D"), ("E", "G"), ("H", "I")]
    rows = []
    for m, teams in enumerate(pairs):
        for c in range(4):
            v = 12 + c + rng.normal(0, 0.4)
            for team in teams:
                gt = rng.uniform(1.0, 1.4)
                bias = 4 * v - 80 + (team_extra if team == "F" else 0) + rng.normal(0, 0.5)
                rows.append({"camera": f"c{c}", "match_id": f"m{m}", "pressing_team_name": team, "gt": gt,
                             "sim": gt * (1 + bias / 100), "players_visible": v, "bias": bias})
    return pl.DataFrame(rows)


def test_team_matches_and_per_team():
    t = table()
    assert team_matches(t)["F"] == {"m0", "m1", "m2", "m3"}
    p = per_team(t)
    assert p.filter(pl.col("pressing_team_name") == "F")["matches"][0] == 4


def test_balanced_mean_weights_teams_equally():
    t = table(team_extra=-10)
    est, lo, hi = balanced_mean(t, n_boot=300)
    team_means = t.group_by("pressing_team_name").agg(pl.col("bias").mean())["bias"]
    assert est == pytest.approx(float(team_means.mean()))
    assert lo <= est <= hi


def test_leave_one_team_out_drops_all_matches_of_the_team():
    lo = leave_one_team_out(table())
    row = lo.filter(pl.col("left_out") == "F")
    assert row["matches"][0] == 2 and row["slope"][0] == pytest.approx(4, abs=0.5)


def test_team_effect_finds_a_planted_effect_and_no_effect():
    assert team_effect(table(team_extra=-6), "F", n_boot=200)["extra_bias_pp"] == pytest.approx(-6, abs=0.6)
    e = team_effect(table(team_extra=0), "F", n_boot=200)
    assert e["ci_low"] <= 0 <= e["ci_high"]


def test_correction_with_team_unseen_is_accurate_when_no_team_effect():
    _, s = correction_team_unseen(table())
    assert s["mean_abs_error"] < 2.0
