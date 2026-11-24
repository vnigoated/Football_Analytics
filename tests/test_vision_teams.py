import numpy as np
import polars as pl

from pitchlens.vision.teams import (
    assign_teams,
    assign_teams_by_track,
    colour_feature,
    evaluate_teams,
    kmeans,
    match_tracks_to_gt,
    sample_crops,
    shirt_region,
    summarise,
)

GREEN, RED, BLUE, YELLOW = (40, 140, 40), (200, 30, 30), (30, 40, 200), (230, 220, 20)


def _scene():
    """A grass image with 4 red players, 4 blue players and 1 yellow referee."""
    img = np.zeros((400, 900, 3), np.uint8)
    img[:] = GREEN
    people = []
    for i in range(9):
        colour = RED if i < 4 else BLUE if i < 8 else YELLOW
        x1, y1 = 20 + 95 * i, 100
        img[y1:y1 + 120, x1:x1 + 40] = colour
        people.append((i, (x1, y1, x1 + 40, y1 + 120)))
    return img, people


def test_colour_feature_separates_kits():
    img, people = _scene()
    f = np.stack([colour_feature(shirt_region(img, b)) for _, b in people])
    assert np.allclose(f[0], f[1]) and not np.allclose(f[0], f[4])
    assert np.isclose(np.linalg.norm(f[0]), 1.0)


def test_kmeans_two_blobs():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    lab = kmeans(x, 2)
    assert len(set(lab[:20])) == 1 and len(set(lab[20:])) == 1 and lab[0] != lab[20]


def test_assign_teams_with_referee():
    img, people = _scene()
    rng = np.random.default_rng(1)
    ids, feats = [], []
    for tid, b in people:  # 3 noisy crops per person
        for _ in range(3):
            noisy = np.clip(img.astype(int) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
            ids.append(tid)
            feats.append(colour_feature(shirt_region(noisy, b)))
    labels = assign_teams(np.array(ids), np.stack(feats), k=3)
    assert len({labels[i] for i in range(4)}) == 1 and len({labels[i] for i in range(4, 8)}) == 1
    assert labels[0] != labels[4] and {labels[0], labels[4]} == {"team_a", "team_b"}
    assert labels[8] == "other"
    by_track = assign_teams_by_track(np.array(ids), np.stack(feats), k=3)
    assert by_track[0] == by_track[3] != by_track[4] == by_track[7] and by_track[8] == "other"


def test_sample_crops_covers_every_track():
    rows = [{"frame": f, "id": 1, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 50.0} for f in range(1, 101)]
    rows += [{"frame": 3, "id": 2, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 50.0}]  # not on a sampled frame
    out = sample_crops(pl.DataFrame(rows), every=5, max_per_track=4)
    assert out.filter(pl.col("id") == 1).height == 4
    assert out.filter(pl.col("id") == 2)["frame"].to_list() == [3]


def test_evaluate_teams_maps_clusters():
    gt = pl.DataFrame({"frame": [1, 1, 1, 1], "id": [10, 11, 12, 13],
                       "role": ["player", "player", "referee", "goalkeeper"],
                       "team": ["left", "right", None, "left"]})
    matches = pl.DataFrame({"frame": [1, 1, 1, 1], "track_id": [1, 2, 3, 4], "gt_id": [10, 11, 12, 13]})
    labels = {1: "team_b", 2: "team_a", 3: "other", 4: "team_b"}  # clusters swapped vs left/right
    c = evaluate_teams(labels, matches, gt)
    assert c["player_correct"] == 2 and c["referee_other"] == 1 and c["keeper_other"] == 0
    s = summarise([c])
    assert s["player_team_accuracy"] == 1.0 and s["referee_as_other"] == 1.0


def test_match_tracks_to_gt():
    box = {"x1": [0.0, 100.0], "y1": [0.0, 0.0], "x2": [10.0, 110.0], "y2": [20.0, 20.0]}
    tracks = pl.DataFrame({"frame": [1, 1], "id": [7, 8], **box})
    gt = pl.DataFrame({"frame": [1, 1], "id": [21, 20], **{k: v[::-1] for k, v in box.items()}})
    m = match_tracks_to_gt(tracks, gt).sort("track_id")
    assert m["gt_id"].to_list() == [20, 21]
