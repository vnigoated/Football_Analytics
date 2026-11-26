"""Main-camera filter: drop replays and close-ups before counting visible players (Phase 2 · B3).

The pressing correction assumes the wide main-camera view. Broadcasts also show close-ups, replays and other
cameras. Three cues that need no training data mark them:

* **grass share**: the share of green pitch pixels; a close-up of a player or the crowd has little;
* **player size**: the median height of the tracked people; a close-up has a few very tall boxes;
* **shot cuts**: a large colour-histogram change between consecutive frames starts a new shot, and a whole shot is
  kept or dropped by majority, so single noisy frames do not split a view.

SN-GSR contains main-camera footage only, so on SN-GSR every frame the filter drops is a frame it should have kept:
the evaluation there measures the false-removal rate and the effect on the visible-player count.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import polars as pl


def grass_share(bgr: np.ndarray) -> float:
    """Share of pitch-green pixels (HSV hue 30-90, saturation >= 50, value >= 40; as in the A4 colour features)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(((h >= 30) & (h <= 90) & (s >= 50) & (v >= 40)).mean())


def colour_signature(bgr: np.ndarray) -> np.ndarray:
    """Normalised hue-saturation histogram (16 x 4) for shot-cut detection."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256]).ravel().astype(np.float32)
    return h / max(float(h.sum()), 1.0)


def cut_scores(signatures: np.ndarray) -> np.ndarray:
    """Per frame: Bhattacharyya distance (0 = same colours, 1 = disjoint) to the previous frame; 0 for the first."""
    s = np.sqrt(np.clip(np.asarray(signatures, np.float64), 0, None))
    d = np.zeros(len(s))
    if len(s) > 1:
        bc = (s[1:] * s[:-1]).sum(1)
        d[1:] = np.sqrt(np.clip(1 - bc, 0, 1))
    return d


def frame_features(frames: Sequence[tuple[int, Path]], workers: int = 8) -> pl.DataFrame:
    """``frame, grass, cut`` for every frame (images read at 1/4 size)."""
    def one(fp):
        img = cv2.imread(str(fp[1]), cv2.IMREAD_REDUCED_COLOR_4)
        if img is None:
            return fp[0], np.nan, np.zeros(64, np.float32)
        return fp[0], grass_share(img), colour_signature(img)

    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(one, frames))
    sig = np.stack([r[2] for r in res]) if res else np.zeros((0, 64))
    return pl.DataFrame({"frame": [r[0] for r in res], "grass": [r[1] for r in res], "cut": cut_scores(sig)},
                        schema={"frame": pl.Int64, "grass": pl.Float64, "cut": pl.Float64})


def person_size(tracks: pl.DataFrame) -> pl.DataFrame:
    """``frame, box_height, people``: median height (px) of the tracked people per frame, and how many."""
    return (tracks.with_columns(h=pl.col("y2") - pl.col("y1")).group_by("frame")
            .agg(pl.col("h").median().alias("box_height"), pl.len().alias("people")))


# Journal step 6: chosen by leave-one-game-out on 8 full SoccerNet broadcasts (every fold picked it). Per-frame
# decisions: at 2 fps a shot vote merges close-ups into the main shots around them.
BROADCAST = {"grass_min": 0.15, "box_max": 200.0, "shot_vote": False, "no_people_ok": False}


def main_view(features: pl.DataFrame, grass_min: float = 0.2, box_max: float = 300.0, cut_min: float = 0.5,
              shot_vote: bool = True, no_people_ok: bool = True) -> pl.DataFrame:
    """Add ``shot`` (a new shot starts where ``cut >= cut_min``) and ``main`` (True = keep).

    A frame looks like the main camera if its grass share is at least ``grass_min`` and the median tracked person is
    at most ``box_max`` px tall. Frames without people pass the size test when ``no_people_ok``, and fail it
    otherwise. With ``shot_vote`` a whole shot takes the majority decision of its frames.

    The defaults are B3's (main-camera clips). For full broadcasts use ``BROADCAST`` (journal step 6): on 8 SoccerNet
    broadcasts the detector often finds nobody in a close-up, so B3 kept 40% of the close-ups.

    Defaults from B3 run 1 (SN-GSR, main camera only):
    - Grass share is at least 0.21 in 99% of frames, but a sunny stadium with hard shadows and pale turf falls to
      0.2–0.3. A threshold of 0.3 wrongly dropped 2 of 107 clips; 0.2 drops none.
    - The median player is at most 255 px tall.
    - Consecutive frames never differ by more than 0.16, and a cut starts at 0.5.
    """
    f = features.sort("frame").with_columns(
        shot=(pl.col("cut") >= cut_min).cum_sum(),
        _ok=(pl.col("grass").fill_null(0) >= grass_min)
        & ((pl.col("box_height") <= box_max).fill_null(no_people_ok) if "box_height" in features.columns
           else pl.lit(True)),
    )
    if shot_vote:
        f = f.with_columns(main=pl.col("_ok").mean().over("shot") >= 0.5)
    else:
        f = f.with_columns(main=pl.col("_ok"))
    return f.drop("_ok")
