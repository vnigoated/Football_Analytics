import numpy as np
import polars as pl
import pytest

from pitchlens.analytics.correction import evaluate_correction


def test_linear_bias_is_corrected_out_of_sample():
    rows = []
    for c, vis in enumerate(np.linspace(12, 16, 6)):
        bias = 4.0 * vis - 80.0  # -32 % ... -16 %
        for m in range(3):
            for team in ("A", "B"):
                gt = 1.0 + 0.1 * m
                rows.append({"camera": f"c{c}", "match_id": f"m{m}", "pressing_team_name": team,
                             "gt": gt, "sim": gt * (1 + bias / 100), "players_visible": vis,
                             "bias": bias})  # fmt: skip
    res, s = evaluate_correction(pl.DataFrame(rows))
    assert s["raw_mean_abs_error"] > 15
    assert s["corrected_mean_abs_error"] == pytest.approx(0.0, abs=1e-6)
    assert s["slope_pp_per_player"] == pytest.approx(4.0)
    assert res.height == 36
