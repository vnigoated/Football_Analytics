import cv2
import numpy as np
import polars as pl

from pitchlens.vision.shots import (
    BROADCAST,
    colour_signature,
    cut_scores,
    frame_features,
    grass_share,
    main_view,
    person_size,
)


def pitch_img(green=1.0):
    img = np.full((60, 80, 3), (40, 40, 160), np.uint8)  # red-ish (BGR)
    img[: int(60 * green)] = (40, 150, 60)  # grass green (BGR)
    return img


def test_grass_share():
    assert grass_share(pitch_img(1.0)) > 0.99
    assert grass_share(pitch_img(0.5)) == np.float64(0.5)
    assert grass_share(pitch_img(0.0)) == 0.0


def test_cut_scores_find_a_cut():
    sigs = np.stack([colour_signature(pitch_img(1.0))] * 3 + [colour_signature(pitch_img(0.0))] * 3)
    d = cut_scores(sigs)
    assert d[0] == 0 and d[1] < 0.01 and d[3] > 0.9 and d[4] < 0.01


def test_frame_features_from_files(tmp_path):
    frames = []
    for i, g in enumerate([1.0, 1.0, 0.1], 1):
        p = tmp_path / f"{i:06d}.jpg"
        cv2.imwrite(str(p), cv2.resize(pitch_img(g), (320, 240), interpolation=cv2.INTER_NEAREST))
        frames.append((i, p))
    f = frame_features(frames, workers=2)
    assert f["frame"].to_list() == [1, 2, 3]
    assert f["grass"][0] > 0.9 and f["grass"][2] < 0.2 and f["cut"][2] > 0.5


def test_main_view_with_shot_vote():
    # shot 1: frames 1-6 pitch (frame 4 a noisy low-grass frame); shot 2: frames 7-9 close-up
    feats = pl.DataFrame({"frame": list(range(1, 10)),
                          "grass": [0.8, 0.8, 0.8, 0.1, 0.8, 0.8, 0.1, 0.1, 0.1],
                          "cut": [0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.9, 0.0, 0.0],
                          "box_height": [80.0] * 6 + [400.0] * 3})
    m = main_view(feats)
    assert m["main"].to_list() == [True] * 6 + [False] * 3
    assert m["shot"].to_list() == [0] * 6 + [1] * 3
    assert not main_view(feats, shot_vote=False)["main"][3]


def test_main_view_broadcast_drops_frames_without_people():
    feats = pl.DataFrame({"frame": [1, 2, 3], "grass": [0.8, 0.5, 0.8], "cut": [0.0, 0.9, 0.9],
                          "box_height": [90.0, None, 250.0]})
    assert main_view(feats, shot_vote=False)["main"].to_list() == [True, True, True]  # B3: no people passes
    assert main_view(feats, **BROADCAST)["main"].to_list() == [True, False, False]


def test_person_size():
    tracks = pl.DataFrame({"frame": [1, 1, 2], "y1": [0.0, 0.0, 0.0], "y2": [50.0, 100.0, 300.0]})
    p = person_size(tracks).sort("frame")
    assert p["box_height"].to_list() == [75.0, 300.0] and p["people"].to_list() == [2, 1]
