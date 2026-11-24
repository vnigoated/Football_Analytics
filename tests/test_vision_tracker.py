import numpy as np
import polars as pl
import pytest

from pitchlens.data.gsr import parse_labels
from pitchlens.vision.tracker import (
    DET_SCHEMA,
    TRACK_SCHEMA,
    gt_tracks,
    interpolate_gaps,
)


def _two_players(n=40, gap=None):
    """Two people walking in opposite directions on separate rows; optional detection gap."""
    gt, dets = [], []
    for f in range(1, n + 1):
        for pid, x, y in ((1, 100 + 10 * f, 300.0), (2, 900 - 10 * f, 600.0)):
            box = (x, y, x + 30.0, y + 80.0)
            gt.append({"frame": f, "id": pid, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
            if gap and pid == 1 and gap[0] <= f <= gap[1]:
                continue
            dets.append({"frame": f, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                         "score": 0.9, "cls": "person"})
    gt_schema = {k: v for k, v in TRACK_SCHEMA.items() if k != "score"}
    return pl.DataFrame(gt, schema=gt_schema), pl.DataFrame(dets, schema=DET_SCHEMA)


def test_gt_tracks_filters_and_clips():
    data = {
        "images": [{"image_id": "1", "file_name": "000001.jpg", "width": 1920, "height": 1080,
                    "has_labeled_person": True}],
        "annotations": [
            {"image_id": "1", "track_id": 5, "attributes": {"role": "player"},
             "bbox_image": {"x": -10, "y": 100, "w": 30, "h": 80}},
            {"image_id": "1", "track_id": 6, "attributes": {"role": "ball"},
             "bbox_image": {"x": 500, "y": 500, "w": 8, "h": 8}},
            {"image_id": "1", "track_id": 7, "attributes": {"role": "referee"},
             "bbox_image": {"x": 700, "y": 100, "w": 30, "h": 80}},
        ],
    }
    gt = gt_tracks(*parse_labels(data, "SNGS-001"))
    assert gt["id"].to_list() == [5, 7]  # ball dropped
    assert gt.row(0, named=True)["x1"] == 0.0 and gt.row(0, named=True)["x2"] == 20.0


def test_interpolate_gaps():
    t = pl.DataFrame(
        [{"frame": 1, "id": 1, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "score": 1.0},
         {"frame": 5, "id": 1, "x1": 40.0, "y1": 0.0, "x2": 50.0, "y2": 10.0, "score": 1.0},
         {"frame": 30, "id": 1, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "score": 1.0}],
        schema=TRACK_SCHEMA,
    )
    out = interpolate_gaps(t, max_gap=10)
    assert out["frame"].to_list() == [1, 2, 3, 4, 5, 30]  # the 24-frame gap stays open
    assert out.filter(pl.col("frame") == 3)["x1"].item() == 20.0
    assert interpolate_gaps(t, max_gap=0).height == 3


def test_tracking_and_metrics():
    pytest.importorskip("trackers")
    from pitchlens.vision.tracker import evaluate, headline, run_tracker

    gt, dets = _two_players()
    frames = list(range(1, 41))
    perfect = headline(evaluate(gt, gt.with_columns(score=pl.lit(1.0)), frames))
    assert perfect["HOTA"] > 0.99 and perfect["IDSW"] == 0

    tracks = run_tracker(dets, frames, "bytetrack", minimum_consecutive_frames=1)
    m = headline(evaluate(gt, tracks, frames))
    assert m["IDSW"] == 0 and m["HOTA"] > 0.9
    assert tracks["id"].n_unique() == 2


def test_gap_is_bridged():
    pytest.importorskip("trackers")
    from pitchlens.vision.tracker import evaluate, headline, run_tracker

    gt, dets = _two_players(gap=(15, 19))
    frames = list(range(1, 41))
    tracks = run_tracker(dets, frames, "bytetrack", minimum_consecutive_frames=1)
    filled = interpolate_gaps(tracks, max_gap=10)
    assert tracks["id"].n_unique() == 2  # the same id before and after the gap
    assert filled.height - tracks.height == 5  # the 5 missing boxes restored
    assert headline(evaluate(gt, filled, frames))["DetA"] > headline(evaluate(gt, tracks, frames))["DetA"]
    assert np.isclose(filled.filter((pl.col("frame") == 17) & (pl.col("x1") < 500))["x1"].item(), 270.0)
