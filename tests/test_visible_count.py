import numpy as np
import polars as pl
import pytest

from pitchlens.analytics.visible_count import (
    calibrate,
    count_summary,
    error_by,
    fit_calibration,
    gt_counts,
    pitch_counts,
    predicted_counts,
)
from pitchlens.data.gsr import parse_labels


def table(seed=0, clips=6, frames=50, offset=-1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(clips):
        for f in range(frames):
            gt = int(rng.integers(8, 20))
            rows.append({"sequence": f"S{c}", "frame": f, "gt": gt, "ours": gt + offset + int(rng.integers(-1, 2)),
                         "zoom": float(rng.uniform(40, 200))})
    return pl.DataFrame(rows)


def test_summary_bias_and_implied_error():
    s = count_summary(table(), n_boot=200)
    assert s["bias"] == pytest.approx(-1.0, abs=0.15)
    assert s["bias_ci_low"] <= s["bias"] <= s["bias_ci_high"]
    assert s["implied_bias_shift_pp"] == pytest.approx(4.1 * s["bias"], abs=0.01)
    assert 0 <= s["frame_within_1"] <= 1 and s["clip_mae"] >= 0


def test_calibration_removes_offset():
    t = table(offset=-2.0)
    a, b = fit_calibration(t)
    s = count_summary(calibrate(t, a, b), n_boot=100)
    assert abs(s["bias"]) < 0.1


def test_error_by_bins():
    e = error_by(table(), "zoom", edges=[80.0, 120.0], labels=["wide", "mid", "close"])
    assert e["frames"].sum() == 300 and set(e["zoom"].to_list()) == {"wide", "mid", "close"}


def test_counts_from_tracks_labels_and_game_state():
    tracks = pl.DataFrame({"frame": [1, 1, 1, 2], "id": [1, 2, 3, 1]})
    c = predicted_counts(tracks, {1: ("player", "left"), 2: ("referee", None), 3: ("goalkeeper", "right")}, [1, 2, 3])
    assert c["ours"].to_list() == [2, 1, 0]
    pred = pl.DataFrame({"frame": [1, 1, 2], "role": ["player", "referee", "goalkeeper"]})
    assert pitch_counts(pred, [1, 2, 3])["ours"].to_list() == [1, 1, 0]
    images = [{"image_id": "1", "file_name": "000001.jpg", "has_labeled_person": True}]
    anns = [{"image_id": "1", "track_id": i, "attributes": {"role": r}, "bbox_image": {"x": 0, "y": 0, "w": 10, "h": h}}
            for i, (r, h) in enumerate([("player", 50), ("goalkeeper", 70), ("referee", 60), ("ball", 5)])]
    g = gt_counts(*parse_labels({"images": images, "annotations": anns}, "S"))
    assert g["gt"].to_list() == [2] and g["zoom"].to_list() == [60.0]


def test_error_by_the_label_count_itself_and_tied_edges():
    t = table()
    e = error_by(t, "gt", edges=[10.5, 14.5], labels=["low", "mid", "high"])
    assert e["frames"].sum() == 300 and e["gt_mean"].min() < e["gt_mean"].max()
    assert error_by(t, "zoom", edges=[80.0, 80.0, 120.0], labels=["a", "b", "c", "d"])["frames"].sum() == 300
