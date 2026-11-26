import json

import pytest

from pitchlens.analytics.broadcast_shots import (
    complete_games,
    filter_report,
    frame_truth,
    leak_shift,
    load_all,
    max_leak,
    per_game,
    time_shares,
)


def write_game(root, name, anns):
    d = root / "lg" / "2015" / name
    d.mkdir(parents=True)
    (d / "Labels-cameras.json").write_text(json.dumps({"annotations": anns}), encoding="utf-8")


def ann(half, sec, label, replay="real-time"):
    return {"gameTime": f"{half} - 00:00", "position": str(int(sec * 1000)), "label": label, "replay": replay}


def test_shares_and_completeness(tmp_path):
    # each label describes the shot ENDING at its timestamp
    write_game(tmp_path, "g1", [ann(1, 0, "Public"), ann(1, 60, "Main camera center"),
                                ann(1, 80, "Close-up player or field referee"),
                                ann(1, 100, "Main camera center", "replay"), ann(1, 900, "Main camera left"),
                                ann(1, 920, "Public"), ann(2, 0, "Other"), ann(2, 20, "Main camera center")])
    shots = load_all(tmp_path)
    assert shots.height == 6  # first label of each half dropped
    assert shots["start_s"][0] == 0 and shots["duration_s"][0] == 60 and shots["label"][0] == "Main camera center"
    assert shots["league"][0] == "lg"
    g = per_game(shots)
    total = 60 + 20 + 20 + 800 + 20 + 20
    assert g["main_live"][0] == pytest.approx((60 + 800 + 20) / total)
    assert g["main_replay"][0] == pytest.approx(20 / total)
    s = time_shares(shots)
    assert s["share"].sum() == pytest.approx(1.0)
    assert complete_games(shots, min_changes=3, max_gap_s=1000) == ["lg/2015/g1"]
    assert complete_games(shots, min_changes=3, max_gap_s=100) == []


def test_frame_truth_and_filter_report(tmp_path):
    write_game(tmp_path, "g1", [ann(1, 0, "Public"), ann(1, 10, "Main camera center"),
                                ann(1, 14, "Close-up player or field referee"), ann(1, 20, "Main camera center")])
    shots = load_all(tmp_path)
    # shots: 0-10 main, 10-14 close-up, 14-20 main (each label = the shot ending at it)
    t = frame_truth(shots, 1, [5.0, 12.0, 13.0, 19.5, 25.0])
    assert t["group"].to_list() == ["main camera", "close-up", "close-up", "main camera", None]
    import polars as pl

    frames = t.with_columns(game=pl.lit("g1"), main=pl.Series([True, False, True, True, True]),
                            people=pl.Series([14.0, 2.0, 4.0, 12.0, 0.0]))
    by_group, per_game, s = filter_report(frames)
    assert s["main_live_removed"] == 0.0
    assert s["nonmain_kept"] == pytest.approx(0.5)
    assert s["count_shift"] == pytest.approx((14 + 4 + 12) / 3 - 13)
    assert per_game.height == 1 and set(by_group["cat"]) == {"main camera, live", "close-up"}


def test_leak_tolerance():
    assert leak_shift(0.3, 0.0, 14, 3) == 0.0
    s = leak_shift(0.3, 1.0, 14, 3)  # all leaked: f = 0.3
    assert s == pytest.approx(4.1 * 0.3 * -11)
    m = max_leak(0.3, 14, 3, max_pp=1.0)
    assert abs(leak_shift(0.3, m, 14, 3)) <= 1.0 < abs(leak_shift(0.3, m + 0.01, 14, 3))
    assert max_leak(0.3, 14, 14) == 1.0
