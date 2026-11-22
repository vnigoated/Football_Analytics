import numpy as np
import polars as pl
import pytest

from pitchlens.analytics.uncertainty import bootstrap_bias


def _joined(factor_by_match: dict[str, float], n_frames: int = 3000, seed: int = 1):
    rng = np.random.default_rng(seed)
    parts = []
    for m, f in factor_by_match.items():
        for team in ("A", "B"):
            gt = rng.uniform(0, 1, n_frames)
            act = rng.integers(0, 4, n_frames)
            parts.append(
                pl.DataFrame(
                    {
                        "camera": "c",
                        "view_length": 40.0,
                        "match_id": m,
                        "pressing_team_name": f"{m}-{team}",
                        "period_id": 1,
                        "timestamp_s": np.arange(n_frames) * 0.2,
                        "pressure_on_carrier": gt,
                        "pressure_on_carrier_sim": gt * f,
                        "n_active_pressers": act,
                        "n_active_pressers_sim": np.round(act * f),
                    }
                )
            )
    return pl.concat(parts)


def test_constant_bias_gives_tight_interval():
    per_team, overall = bootstrap_bias(_joined({"m1": 0.8, "m2": 0.8}), n_boot=300)
    row = overall.row(0, named=True)
    assert row["pressure_bias"] == pytest.approx(-20.0)
    assert row["pressure_lo"] == pytest.approx(-20.0, abs=1e-6)
    assert row["pressure_hi"] == pytest.approx(-20.0, abs=1e-6)
    assert per_team.height == 4


def test_varying_bias_interval_covers_point_and_widens():
    _, overall = bootstrap_bias(_joined({"m1": 0.7, "m2": 0.8, "m3": 0.9}), n_boot=500)
    row = overall.row(0, named=True)
    assert row["pressure_lo"] < row["pressure_bias"] < row["pressure_hi"]
    assert row["pressure_bias"] == pytest.approx(-20.0)
    assert row["pressure_hi"] - row["pressure_lo"] > 5  # match-to-match spread shows up
    assert row["matches"] == 3 and row["team_matches"] == 6


def test_stadium_bootstrap_separates_sources():
    from pitchlens.analytics.uncertainty import stadium_bootstrap

    parts = []
    for cam, f in (("A", 0.7), ("B", 0.8), ("C", 0.9)):
        parts.append(
            _joined({"m1": f, "m2": f}).with_columns(
                pl.lit(cam).alias("camera"), pl.lit(14.0 + f).alias("players_visible")
            )
        )
    per_cam, iv = stadium_bootstrap(pl.concat(parts), n_boot=300)
    assert per_cam.height == 3
    assert per_cam.sort("camera")["pressure_bias"].to_list() == pytest.approx([-30, -20, -10])
    row = {r["source"]: r for r in iv.filter(pl.col("target") == "pressure").iter_rows(named=True)}
    assert row["combined"]["bias"] == pytest.approx(-20.0)
    # every camera has a constant ratio, so matches/blocks add (almost) nothing
    assert row["matches + within-match"]["hi"] - row["matches + within-match"]["lo"] < 0.5
    assert row["stadium camera only"]["hi"] - row["stadium camera only"]["lo"] > 5
