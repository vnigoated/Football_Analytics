import numpy as np
import polars as pl
import pytest

from pitchlens.data.gsr_camera import clip_footprints, orient


def make_clip(H_img_to_pitch: np.ndarray, frames: int = 3):
    """People whose labelled pitch positions are the exact projection of their box bottoms."""
    rng = np.random.default_rng(0)
    img_rows, obj_rows = [], []
    for f in range(1, frames + 1):
        iid = f"i{f}"
        img_rows.append({"image_id": iid, "frame": f})
        for k in range(10):
            u, v = rng.uniform(100, 1800), rng.uniform(400, 1000)
            p = H_img_to_pitch @ np.array([u, v, 1.0])
            obj_rows.append({"image_id": iid, "role": "player", "x": u - 10, "y": v - 60, "w": 20.0, "h": 60.0,
                             "pitch_raw_x": p[0] / p[2], "pitch_raw_y": p[1] / p[2]})
        obj_rows.append({"image_id": iid, "role": "ball", "x": 900.0, "y": 700.0, "w": 5.0, "h": 5.0,
                         "pitch_raw_x": 3.0, "pitch_raw_y": -2.0})
    return pl.DataFrame(img_rows), pl.DataFrame(obj_rows)


def test_footprint_from_exact_homography_and_orientation():
    # a perspective view: image top -> far side (positive y), image left -> negative x
    src = np.array([[0, 0], [1920, 0], [0, 1080], [1920, 1080]], np.float32)
    dst = np.array([[-40, 34], [40, 34], [-15, -20], [15, -20]], np.float32)
    import cv2

    H = cv2.getPerspectiveTransform(src, dst).astype(float)
    images, objects = make_clip(H)
    fp = clip_footprints(images, objects)
    assert fp.height == 3
    assert fp["x_top_left"][0] == pytest.approx(-40, abs=0.05) and fp["y_bottom_right"][0] == pytest.approx(-20, abs=0.05)
    assert fp["ball_detected"].all() and fp["ball_x"][0] == pytest.approx(3.0)
    assert orient(fp).equals(fp)  # already in SkillCorner's convention

    mirrored = fp.with_columns([(-pl.col(c)).alias(c) for c in fp.columns if c.startswith("y_") or c == "ball_y"])
    back = orient(mirrored)  # near edge at positive y and left/right swapped -> flip y, then x
    assert (back["y_bottom_left"] < back["y_top_left"]).all()
    assert (back["x_top_left"] < back["x_top_right"]).all()
    assert orient(fp.head(0)).height == 0


def test_transfer_correction_recovers_a_shared_line():
    from pitchlens.analytics.robustness import transfer_correction

    rng = np.random.default_rng(1)

    def rows(cams, matches):
        out = []
        for c in cams:
            for m in matches:
                v = rng.uniform(12, 16)
                for team in ("A", "B"):
                    gt = rng.uniform(1, 1.4)
                    bias = 4 * v - 80 + rng.normal(0, 1)
                    out.append({"camera": c, "match_id": m, "pressing_team_name": team, "gt": gt,
                                "sim": gt * (1 + bias / 100), "players_visible": v, "bias": bias})
        return pl.DataFrame(out)

    train, test = rows([f"a{i}" for i in range(6)], ["m1", "m2", "m3"]), rows(["s1", "s2"], ["m1", "m2", "m3"])
    res, s = transfer_correction(train, test)
    assert res.height == test.height
    assert s["corrected_mean_abs_error"] < 2.0 < s["raw_mean_abs_error"]
    assert s["slope_train"] == pytest.approx(4, abs=0.5)
