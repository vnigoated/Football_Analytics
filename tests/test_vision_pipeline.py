import numpy as np
import polars as pl
import pytest

from pitchlens.geometry.homography import apply_homography
from pitchlens.vision.pipeline import (
    GS_SCHEMA,
    assign_roles,
    check_points,
    fill_homographies,
    game_state,
    gs_hota,
    id_sim,
    loc_sim,
    normalise_attributes,
    oracle_attributes,
    to_pitch,
    write_predictions,
)

pytest.importorskip("trackers")


def shift(dx: float) -> np.ndarray:
    """Image -> pitch: 0.05 m per pixel, then shifted by dx metres."""
    return np.array([[0.05, 0, -48 + dx], [0, 0.05, -27], [0, 0, 1]], float)


def gs_rows(rows):
    return pl.DataFrame([{**{k: None for k in GS_SCHEMA}, **r} for r in rows], schema=GS_SCHEMA)


def test_loc_sim_tolerance():
    s = loc_sim(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0], [5.0, 0.0], [3.0, 4.0], [10.0, 0.0]]))
    assert s[0, 0] == pytest.approx(1.0)
    assert s[0, 1] == pytest.approx(0.05)
    assert s[0, 2] == pytest.approx(0.05)
    assert s[0, 3] < 1e-4


def test_id_sim_official_rules():
    gt = normalise_attributes(gs_rows([{"role": "player", "team": "left", "jersey": "10"}, {"role": "referee"},
                                       {"role": "player", "team": "left"}, {"role": "goalkeeper", "team": "left",
                                                                            "jersey": "1"}]))
    pr = normalise_attributes(gs_rows([{"role": "player", "team": "left"}, {"role": "referee", "team": "right"},
                                       {"role": "player", "team": "left", "jersey": "10"},
                                       {"role": "goalkeeper", "team": "left"}]))
    s = id_sim(gt, pr, ("role", "team"))
    assert s[1, 1] == 1  # a referee's team is dropped on both sides
    full = id_sim(gt, pr, ("role", "team", "jersey"))
    assert full[0, 0] == 0  # number in the labels, none predicted
    assert full[2, 0] == 1  # no number in the labels, none predicted
    assert full[2, 2] == 0  # no number in the labels, one predicted
    assert full[3, 3] == 1  # goalkeepers' numbers are not scored


def test_write_predictions_format(tmp_path):
    import json

    images = pl.DataFrame({"frame": [1, 2], "image_id": ["1021000001", "1021000002"]})
    pred = gs_rows([{"frame": 2, "id": 5, "x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 32.0, "pitch_x": 3.5,
                     "pitch_y": -1.0, "role": "player", "team": "left", "jersey": "9"},
                    {"frame": 1, "id": 6, "x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 32.0, "role": "referee"}])
    write_predictions(pred, images, tmp_path / "SNGS-021.json")
    d = json.loads((tmp_path / "SNGS-021.json").read_text())["predictions"]
    assert len(d) == 1  # no pitch position -> not exported
    p = d[0]
    assert p["image_id"] == "1021000002" and p["track_id"] == 5 and p["supercategory"] == "object"
    assert p["attributes"] == {"role": "player", "team": "left", "jersey": "9"}
    assert p["bbox_pitch"]["x_bottom_middle"] == 3.5 and "y_bottom_left" in p["bbox_pitch"]


def test_gs_hota_perfect_and_wrong_team():
    rows = [{"frame": f, "id": i, "pitch_x": 10.0 * i + f, "pitch_y": 0.0, "role": "player",
             "team": "left" if i < 2 else "right"} for f in range(1, 6) for i in range(4)]
    gt = gs_rows(rows)
    assert gs_hota(gt, gt, range(1, 6))["HOTA"] == pytest.approx(1.0)
    wrong = gt.with_columns(team=pl.when(pl.col("team") == "left").then(pl.lit("right")).otherwise(pl.lit("left")))
    assert gs_hota(gt, wrong, range(1, 6))["HOTA"] == pytest.approx(0.0)
    assert gs_hota(gt, wrong, range(1, 6), attributes=())["HOTA"] == pytest.approx(1.0)
    far = gt.with_columns(pl.col("pitch_y") + 6.0)  # beyond the 5 m tolerance
    assert gs_hota(gt, far, range(1, 6), attributes=())["HOTA"] == pytest.approx(0.0)


def test_fill_homographies_interpolates_drops_outliers_and_holds_ends():
    frames = list(range(1, 41))
    truth = {f: shift(0.1 * f) for f in frames}
    hs = {f: truth[f] for f in frames if f >= 3 and not 10 <= f <= 14}
    hs[25] = shift(30.0)  # one wrong frame
    out, status = fill_homographies(hs, frames, max_gap=10)
    src = check_points()
    err = lambda f: np.abs(apply_homography(out[f], src) - apply_homography(truth[f], src)).max()
    assert status[12] == "filled" and err(12) < 1e-4
    assert status[25] == "filled" and err(25) < 1e-4
    assert status[1] == "held" and err(1) == pytest.approx(0.2, abs=1e-4)
    assert status[30] == "own"
    out2, status2 = fill_homographies(hs, frames, max_gap=3)
    assert status2[12] == "none" and out2[12] is None


def test_roles_teams_sides_and_goalkeeper():
    rng = np.random.default_rng(0)
    # tracks 0-9 kit A standing left, 10-19 kit B standing right, 20 odd kit in the left box, 21 odd kit at midfield
    centres = {"A": np.r_[1.0, 0, 0, 0], "B": np.r_[0, 1.0, 0, 0], "odd": np.r_[0, 0, 1.0, 0]}
    kits = ["A"] * 10 + ["B"] * 10 + ["odd", "odd"]
    xs = [-20.0] * 10 + [20.0] * 10 + [-48.0, 0.0]
    crop_ids, feats, rows = [], [], []
    for t, (kit, x) in enumerate(zip(kits, xs, strict=True)):
        for f in range(1, 6):
            crop_ids.append(t)
            feats.append(centres[kit] + rng.normal(0, 0.02, 4))
            rows.append({"frame": f, "id": t, "pitch_x": x + rng.normal(0, 1), "pitch_y": 0.0})
    tracks = pl.DataFrame(rows)
    roles = assign_roles(tracks, np.array(crop_ids), np.array(feats), k=3)
    assert all(roles[t] == ("player", "left") for t in range(10))
    assert all(roles[t] == ("player", "right") for t in range(10, 20))
    assert roles[20] == ("goalkeeper", "left")
    assert roles[21] == ("referee", None)
    flipped = assign_roles(tracks, np.array(crop_ids), np.array(feats), k=3, left_is_negative_x=False)
    assert flipped[0] == ("player", "right")
    # k = 2 with the distance rule sets the odd kits apart too
    roles2 = assign_roles(tracks, np.array(crop_ids), np.array(feats), k=2, outlier_z=3.0)
    assert roles2[21] == ("referee", None) and roles2[0][0] == "player"


def test_to_pitch_and_game_state():
    tracks = pl.DataFrame({"frame": [1, 1, 2], "id": [1, 2, 1], "x1": [950.0, 0.0, 950.0], "y1": [400.0, 0, 400],
                           "x2": [970.0, 20, 970], "y2": [540.0, 100, 540], "score": [0.9, 0.9, 0.9]})
    tp = to_pitch(tracks, {1: shift(0.0), 2: None})
    assert tp.filter((pl.col("frame") == 1) & (pl.col("id") == 1))["pitch_x"][0] == pytest.approx(0.0)
    assert tp.filter(pl.col("frame") == 2)["pitch_x"][0] is None
    gs = game_state(tp, {1: ("player", "left")}, pitch_margin=5.0)
    assert gs.height == 2  # frame 2 has no homography; track 2 maps to (-47.5, -22), on the pitch
    assert gs.filter(pl.col("id") == 2)["role"][0] == "player"  # unknown tracks default to player
    inner = game_state(tp, {1: ("player", "left")}, pitch_margin=-20.0)
    assert inner["id"].to_list() == [1]


def test_oracle_attributes_copies_majority_label():
    pred = gs_rows([{"frame": f, "id": 7, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 30.0, "pitch_x": 0.0,
                     "pitch_y": 0.0, "role": "referee"} for f in (1, 2, 3)])
    gt = gs_rows([{"frame": f, "id": 1, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 30.0, "pitch_x": 0.0,
                   "pitch_y": 0.0, "role": "player", "team": "right", "jersey": "9"} for f in (1, 2, 3)])
    out = oracle_attributes(pred, gt, ("role", "team", "jersey"))
    assert out["role"].unique().to_list() == ["player"]
    assert out["team"].unique().to_list() == ["right"] and out["jersey"].unique().to_list() == ["9"]
