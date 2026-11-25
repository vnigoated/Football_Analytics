"""End-to-end game state from broadcast frames, and its GS-HOTA score (Phase 2 · A6).

Steps per clip (each from an earlier work package):

1. people boxes: RF-DETR (A2), cached per clip;
2. tracks: OC-SORT with camera-motion compensation (A3);
3. pitch keypoints -> one homography per frame (A5), then **over time** (new here): frames whose homography disagrees
   with its neighbours are dropped, short gaps are filled by interpolation, and an optional moving average reduces
   jitter;
4. each box's bottom middle -> pitch position in metres;
5. teams by shirt colour per track (A4), then **roles and sides** (new here): tracks that fit neither team are
   goalkeepers when they stand in a penalty area and referees otherwise; the team standing further left on average
   is "left" (SN-GSR names teams by the side of the goal they defend);
6. no jersey numbers yet (predicted as null).

GS-HOTA (Somers et al., CVPRW 2024) is HOTA with the similarity ``LocSim × IdSim``: ``LocSim = exp(ln(0.05) d² / τ²)``
with d the pitch distance and τ = 5 m (0.05 at 5 m: never matched beyond), and ``IdSim = 1`` only if role, team and
jersey are equal under the official evaluator's rules (sn-trackeval): team counts for players and goalkeepers only,
jersey for players only, and a null number matches only a null number. The ball is not scored. Scored frames:
labelled people, pitch and camera. ``write_predictions`` exports the official prediction format.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

from pitchlens.geometry.homography import apply_homography
from pitchlens.vision.coco import clip_box

GS_ROLES = ("player", "goalkeeper", "referee", "other")
TEAM_ROLES = ("player", "goalkeeper")
TAU_M = 5.0
HALF_LENGTH, HALF_WIDTH = 52.5, 34.0
PENALTY_DEPTH, PENALTY_HALF_WIDTH = 16.5, 20.16
GS_SCHEMA = {
    "frame": pl.Int64,
    "id": pl.Int64,
    **{k: pl.Float64 for k in ("x1", "y1", "x2", "y2", "pitch_x", "pitch_y")},
    "role": pl.String,
    "team": pl.String,
    "jersey": pl.String,
}


# ---------------------------------------------------------------- ground truth


def gs_ground_truth(images: pl.DataFrame, objects: pl.DataFrame, use_raw: bool = False) -> pl.DataFrame:
    """Scored ground truth of one clip: people (no ball) with a pitch position, in labelled frames.

    ``use_raw=False`` uses ``bbox_pitch`` (the field the official evaluation uses).
    """
    px, py = ("pitch_raw_x", "pitch_raw_y") if use_raw else ("pitch_x", "pitch_y")
    frames = {
        r["image_id"]: (r["frame"], r["width"] or 1920, r["height"] or 1080)
        for r in images.iter_rows(named=True)
        if r["has_labeled_person"] is not False and r["has_labeled_pitch"] is not False
        and r["has_labeled_camera"] is not False  # the frames the official evaluator scores
    }
    rows = []
    for o in objects.iter_rows(named=True):
        if o["role"] not in GS_ROLES or o[px] is None or o["image_id"] not in frames:
            continue
        frame, width, height = frames[o["image_id"]]
        box = clip_box(o["x"], o["y"], o["w"], o["h"], width, height) if o["x"] is not None else None
        x, y, w, h = box if box is not None else (math.nan,) * 4
        rows.append({"frame": frame, "id": o["track_id"], "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                     "pitch_x": o[px], "pitch_y": o[py], "role": o["role"],
                     "team": o["team"] if o["role"] in TEAM_ROLES else None, "jersey": o["jersey"]})
    return pl.DataFrame(rows, schema=GS_SCHEMA).sort("frame", "id")


def gt_homographies(images: pl.DataFrame, objects: pl.DataFrame, use_raw: bool = True) -> dict[int, np.ndarray | None]:
    """Oracle: per frame, the homography fitted to the labelled people (A5), or None."""
    from pitchlens.vision.keypoints import frame_homography

    by_image = {k[0]: g for k, g in objects.group_by("image_id")}
    out = {}
    for r in images.iter_rows(named=True):
        g = by_image.get(r["image_id"])
        out[r["frame"]] = None if g is None else frame_homography(g, use_raw=use_raw)[0]
    return out


# ---------------------------------------------------------------- homographies over time


def predict_sequence_keypoints(model: Any, frames: Sequence[tuple[int, Path]], every: int = 1, batch: int = 16,
                               device: str = "cuda", threshold: float = 0.3, workers: int = 8) -> dict[int, np.ndarray]:
    """Detected pitch keypoints (K, 3) for every ``every``-th frame; images are read in parallel."""
    import torch

    from pitchlens.vision.keypoints import decode_heatmaps, load_input

    picked = [(f, p) for i, (f, p) in enumerate(frames) if i % every == 0]
    out: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(picked), batch):
            chunk = picked[start:start + batch]
            xs, srcs = zip(*pool.map(lambda fp: load_input(str(fp[1])), chunk), strict=True)
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                hm = model(torch.from_numpy(np.stack(xs)).to(device)).float().cpu().numpy()
            for (f, _), h, s in zip(chunk, hm, srcs, strict=True):
                out[f] = decode_heatmaps(h, s, threshold)
    return out


def keypoint_homographies(keypoints: dict[int, np.ndarray], min_inliers: int = 6) -> dict[int, np.ndarray | None]:
    from pitchlens.vision.keypoints import homography_from_keypoints

    return {f: homography_from_keypoints(kp, min_inliers=min_inliers)[0] for f, kp in keypoints.items()}


def check_points(size: tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """Nine image points in the lower part of the frame (usually grass): a homography is compared, averaged and
    interpolated through where it sends these points."""
    w, h = size
    return np.array([(w * a, h * b) for b in (0.6, 0.8, 1.0) for a in (0.1, 0.5, 0.9)], float)


def _fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(dst).all():
        return None
    H, _ = cv2.findHomography(src, dst, 0)
    return H


def fill_homographies(
    hs: dict[int, np.ndarray | None],
    frames: Iterable[int],
    max_gap: int = 25,
    outlier_m: float = 5.0,
    window: int = 6,
    smooth: int = 0,
    size: tuple[int, int] = (1920, 1080),
) -> tuple[dict[int, np.ndarray | None], dict[int, str]]:
    """Homographies for every frame from per-frame estimates ``hs`` (None = no estimate).

    1. **Outliers:** an estimate is dropped when the check points it maps lie more than ``outlier_m`` metres (median)
       from the median of its neighbours' mappings within ``window`` frames (needs 2+ neighbours).
    2. **Smoothing** (``smooth`` > 0): each kept mapping is averaged with the kept ones within ``smooth`` frames.
    3. **Gaps** of at most ``max_gap`` frames between kept frames are filled by interpolating the mapped check points
       linearly; at the start and end of a clip the nearest kept homography is held for up to ``max_gap`` frames.

    Returns (frame -> H or None, frame -> "own" / "filled" / "held" / "none").
    """
    frames = sorted(frames)
    src = check_points(size)
    own = sorted(f for f in frames if hs.get(f) is not None)
    pts = {f: apply_homography(hs[f], src) for f in own}
    own = [f for f in own if np.isfinite(pts[f]).all()]
    keep = []
    for i, f in enumerate(own):
        nb = [pts[g] for g in own[max(i - window, 0):i + window + 1] if g != f and abs(g - f) <= window]
        if len(nb) >= 2:
            ref = np.median(np.stack(nb), axis=0)
            if np.median(np.linalg.norm(pts[f] - ref, axis=1)) > outlier_m:
                continue
        keep.append(f)
    if smooth > 0:
        arr = np.array(keep)
        pts = {f: np.mean([pts[g] for g in arr[np.abs(arr - f) <= smooth]], axis=0) for f in keep}
    out: dict[int, np.ndarray | None] = {}
    status: dict[int, str] = {}
    kept = np.array(keep)
    for f in frames:
        if len(kept) == 0:
            out[f], status[f] = None, "none"
            continue
        j = int(np.searchsorted(kept, f))
        if j < len(kept) and kept[j] == f:
            out[f] = hs[f] if smooth == 0 else _fit(src, pts[f])
            status[f] = "own"
        elif 0 < j < len(kept) and kept[j] - kept[j - 1] <= max_gap + 1:
            a, b = int(kept[j - 1]), int(kept[j])
            t = (f - a) / (b - a)
            out[f], status[f] = _fit(src, pts[a] * (1 - t) + pts[b] * t), "filled"
        else:
            near = int(kept[0] if j == 0 else kept[-1] if j == len(kept) else
                       kept[j - 1] if f - kept[j - 1] <= kept[j] - f else kept[j])
            if (j == 0 or j == len(kept)) and abs(f - near) <= max_gap:
                out[f], status[f] = (hs[near] if smooth == 0 else _fit(src, pts[near])), "held"
            else:
                out[f], status[f] = None, "none"
    return out, status


def homography_errors(hs: dict[int, np.ndarray | None], images: pl.DataFrame, objects: pl.DataFrame,
                      use_raw: bool = True) -> tuple[list[np.ndarray], int]:
    """Player position errors (metres) of every labelled frame that has a homography; (errors, frames with H)."""
    from pitchlens.vision.keypoints import position_errors

    by_image = {k[0]: g for k, g in objects.group_by("image_id")}
    errs, n = [], 0
    for r in images.iter_rows(named=True):
        H, g = hs.get(r["frame"]), by_image.get(r["image_id"])
        if H is None or g is None:
            continue
        n += 1
        errs.append(position_errors(H, g, use_raw))
    return errs, n


# ---------------------------------------------------------------- tracks -> game state


def to_pitch(tracks: pl.DataFrame, hs: dict[int, np.ndarray | None]) -> pl.DataFrame:
    """Add ``pitch_x, pitch_y`` (box bottom middle mapped with the frame's homography; null without one)."""
    parts = []
    for (frame,), g in tracks.group_by("frame", maintain_order=True):
        H = hs.get(frame)
        if H is None:
            xy = np.full((g.height, 2), np.nan)
        else:
            feet = np.column_stack([((g["x1"] + g["x2"]) / 2).to_numpy(), g["y2"].to_numpy()])
            xy = apply_homography(H, feet)
        parts.append(g.with_columns(pitch_x=pl.Series(xy[:, 0]), pitch_y=pl.Series(xy[:, 1])))
    if not parts:
        return tracks.with_columns(pitch_x=pl.lit(None, pl.Float64), pitch_y=pl.lit(None, pl.Float64))
    return pl.concat(parts).with_columns(pl.col("pitch_x", "pitch_y").fill_nan(None)).sort("frame", "id")


def _track_means(crop_ids: np.ndarray, features: np.ndarray, heights: np.ndarray | None,
                 min_height: float = 40.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.unique(crop_ids)
    good = np.ones(len(crop_ids), bool) if heights is None else np.asarray(heights) >= min_height
    means, w = [], []
    for t in ids:
        m = crop_ids == t
        use = m & good if (m & good).any() else m
        v = features[use].mean(0)
        means.append(v / max(np.linalg.norm(v), 1e-9))
        w.append(m.sum())
    return ids, np.stack(means), np.array(w, float)


def assign_roles(
    tracks: pl.DataFrame,
    crop_ids: np.ndarray,
    features: np.ndarray,
    heights: np.ndarray | None = None,
    k: int = 2,
    outlier_z: float | None = None,
    left_is_negative_x: bool = True,
    dims: int = 16,
) -> dict[int, tuple[str, str | None]]:
    """Track id -> (role, team) with team "left" / "right" (None for referees).

    Tracks are clustered by their averaged shirt colour (A4: ``k = 2`` best for teams). A track is "not a team
    member" when it falls in a smaller cluster (``k`` > 2) or lies more than ``outlier_z`` robust standard deviations
    (median + z · 1.4826 · MAD of the distances to the team centres) from its team centre. Such a track is a
    goalkeeper when its median pitch position is in a penalty area (team: that goal's side), otherwise a referee.
    The team whose players stand further towards negative x on average is "left" (``left_is_negative_x``).
    """
    from pitchlens.vision.teams import kmeans, pca

    ids, means, w = _track_means(np.asarray(crop_ids), np.asarray(features), heights)
    x = pca(means, dims)
    lab = kmeans(x, k, weights=w) if len(ids) > k else np.arange(len(ids)) % max(k, 1)
    sizes = np.bincount(lab, weights=w, minlength=k)
    order = np.argsort(-sizes)
    team_of = {int(order[0]): "a", int(order[1]): "b"} if k >= 2 else {int(order[0]): "a"}
    team = np.array([team_of.get(int(j)) for j in lab], dtype=object)
    if outlier_z is not None:
        in_team = team != None
        d = np.full(len(ids), np.nan)
        for j, name in team_of.items():
            m = lab == j
            if m.any():
                c = np.average(x[m], axis=0, weights=w[m])
                d[m] = np.linalg.norm(x[m] - c, axis=1)
        dd = d[in_team]
        if len(dd) >= 4:
            med = np.median(dd)
            mad = np.median(np.abs(dd - med)) * 1.4826
            team[in_team & (d > med + outlier_z * max(mad, 1e-9))] = None
    pos = (tracks.filter(pl.col("pitch_x").is_not_null()).group_by("id")
           .agg(pl.col("pitch_x").median().alias("mx"), pl.col("pitch_y").median().alias("my")))
    where = {r["id"]: (r["mx"], r["my"]) for r in pos.iter_rows(named=True)}
    sign = 1.0 if left_is_negative_x else -1.0
    mean_x = {}
    for name in ("a", "b"):
        xs = [where[int(t)][0] for t, tm in zip(ids, team, strict=True) if tm == name and int(t) in where]
        mean_x[name] = np.mean(xs) * sign if xs else 0.0
    side = {"a": "left", "b": "right"} if mean_x["a"] <= mean_x["b"] else {"a": "right", "b": "left"}
    out: dict[int, tuple[str, str | None]] = {}
    for t, tm in zip(ids, team, strict=True):
        t = int(t)
        if tm is not None:
            out[t] = ("player", side[tm])
            continue
        mx, my = where.get(t, (None, None))
        if mx is not None and abs(mx) >= HALF_LENGTH - PENALTY_DEPTH and abs(my) <= PENALTY_HALF_WIDTH:
            out[t] = ("goalkeeper", "left" if mx * sign < 0 else "right")
        else:
            out[t] = ("referee", None)
    return out


def game_state(tracks_pitch: pl.DataFrame, roles: dict[int, tuple[str, str | None]],
               pitch_margin: float | None = 5.0, jerseys: dict[int, str | None] | None = None) -> pl.DataFrame:
    """Predictions in the ``GS_SCHEMA`` layout: boxes with a pitch position (within ``pitch_margin`` metres of the
    pitch, if given), role, team and jersey number (track id -> number in ``jerseys``; null if missing)."""
    t = tracks_pitch.filter(pl.col("pitch_x").is_not_null())
    if pitch_margin is not None:
        t = t.filter((pl.col("pitch_x").abs() <= HALF_LENGTH + pitch_margin)
                     & (pl.col("pitch_y").abs() <= HALF_WIDTH + pitch_margin))
    ids = t["id"].to_list()
    rt = [roles.get(i, ("player", None)) for i in ids]
    return t.with_columns(
        role=pl.Series([r for r, _ in rt], dtype=pl.String),
        team=pl.Series([tm for _, tm in rt], dtype=pl.String),
        jersey=pl.Series([(jerseys or {}).get(i) for i in ids], dtype=pl.String),
    ).select(list(GS_SCHEMA)).cast(GS_SCHEMA)


def oracle_attributes(pred: pl.DataFrame, gt: pl.DataFrame, columns: Sequence[str] = ("role", "team"),
                      iou_thr: float = 0.5) -> pl.DataFrame:
    """Diagnostic: replace a track's ``columns`` by the majority ground-truth value of the boxes it matches
    (IoU >= ``iou_thr``); unmatched tracks keep their prediction."""
    from pitchlens.vision.teams import match_tracks_to_gt

    g = gt.filter(pl.col("x1").is_not_nan())
    m = match_tracks_to_gt(pred, g, iou_thr).join(
        g.select(pl.col("frame"), pl.col("id").alias("gt_id"), *columns), on=["frame", "gt_id"])
    out = pred
    for c in columns:
        maj = (m.group_by("track_id", c).len().sort("len", descending=True)
               .group_by("track_id", maintain_order=True).first().select(pl.col("track_id").alias("id"),
                                                                         pl.col(c).alias(f"_{c}")))
        has = m.select(pl.col("track_id").alias("id")).unique().with_columns(_hit=pl.lit(True))
        out = (out.join(maj, on="id", how="left").join(has, on="id", how="left")
               .with_columns(pl.when(pl.col("_hit")).then(pl.col(f"_{c}")).otherwise(pl.col(c)).alias(c))
               .drop(f"_{c}", "_hit"))
    return out.select(list(GS_SCHEMA))


# ---------------------------------------------------------------- GS-HOTA


def loc_sim(gt_xy: np.ndarray, pr_xy: np.ndarray, tau: float = TAU_M) -> np.ndarray:
    """``exp(ln(0.05) d² / τ²)`` for every (ground truth, prediction) pair."""
    d2 = ((gt_xy[:, None, :] - pr_xy[None, :, :]) ** 2).sum(-1)
    return np.exp(np.log(0.05) * d2 / tau**2)


def normalise_attributes(df: pl.DataFrame) -> pl.DataFrame:
    """The official evaluator's rules (sn-trackeval ``soccernet_gs.py``), applied to labels and predictions alike:
    team only for players and goalkeepers, jersey only for players; otherwise null."""
    return df.with_columns(
        team=pl.when(pl.col("role").is_in(TEAM_ROLES)).then(pl.col("team")).otherwise(pl.lit(None, pl.String)),
        jersey=pl.when(pl.col("role") == "player").then(pl.col("jersey")).otherwise(pl.lit(None, pl.String)),
    )


def id_sim(gt: pl.DataFrame, pr: pl.DataFrame, attributes: Sequence[str]) -> np.ndarray:
    """1 where every requested attribute is equal, else 0 (inputs normalised with ``normalise_attributes``).

    As in the official evaluator, null equals null only: a player whose number is null in the labels (never
    visible) matches only a prediction without a number.
    """
    s = np.ones((gt.height, pr.height), bool)
    for a in attributes:
        g = np.array(gt[a].to_list(), dtype=object)
        p = np.array(pr[a].to_list(), dtype=object)
        s &= g[:, None] == p[None, :]
    return s.astype(float)


def gs_hota(gt: pl.DataFrame, pred: pl.DataFrame, frames: Iterable[int],
            attributes: Sequence[str] = ("role", "team", "jersey"), tau: float = TAU_M) -> dict[str, Any]:
    """HOTA metric dict of one clip with the GS-HOTA similarity (``attributes=()``: position only)."""
    from trackers.eval import compute_hota_metrics

    gt, pred = normalise_attributes(gt), normalise_attributes(pred)
    gt_f = {k[0]: g for k, g in gt.group_by("frame")}
    pr_f = {k[0]: g for k, g in pred.filter(pl.col("pitch_x").is_not_null()).group_by("frame")}
    gi_all, pi_all, sims = [], [], []
    for f in frames:
        g, p = gt_f.get(f), pr_f.get(f)
        gi = g["id"].to_numpy().astype(np.int64) if g is not None else np.zeros(0, np.int64)
        pi = p["id"].to_numpy().astype(np.int64) if p is not None else np.zeros(0, np.int64)
        if len(gi) and len(pi):
            s = loc_sim(g.select("pitch_x", "pitch_y").to_numpy(), p.select("pitch_x", "pitch_y").to_numpy(), tau)
            if attributes:
                s = s * id_sim(g, p, attributes)
        else:
            s = np.zeros((len(gi), len(pi)))
        gi_all.append(gi)
        pi_all.append(pi)
        sims.append(s)
    return compute_hota_metrics(gi_all, pi_all, sims)


GS_VARIANTS = {
    "GS-HOTA": ("role", "team", "jersey"),
    "no jersey": ("role", "team"),
    "role only": ("role",),
    "position only": (),
}


def gs_evaluate(clips: dict[str, dict[str, Any]], variants: dict[str, Sequence[str]] | None = None) -> pl.DataFrame:
    """Score clips (``clips[name] = {"gt", "pred", "frames"}``); one row per variant: HOTA, DetA, AssA (in %)."""
    from trackers.eval import aggregate_hota_metrics

    rows = []
    for name, attrs in (variants or GS_VARIANTS).items():
        r = aggregate_hota_metrics([gs_hota(c["gt"], c["pred"], c["frames"], attrs) for c in clips.values()])
        rows.append({"metric": name, "HOTA": round(100 * float(r["HOTA"]), 2),
                     "DetA": round(100 * float(r["DetA"]), 2), "AssA": round(100 * float(r["AssA"]), 2)})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- CPU stage, in parallel


def _cpu_clip(job: tuple[str, str, pl.DataFrame, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Tracks (OC-SORT + camera-motion compensation), crops and their colour features for one clip."""
    from pitchlens.vision.teams import crop_features, frame_loader, sample_crops
    from pitchlens.vision.tracker import run_tracker, sequence_frames

    name, seq_dir, dets, params = job
    cv2.setNumThreads(1)
    frames = [f for f, _ in sequence_frames(seq_dir)]
    bgr = lambda f: cv2.imread(str(Path(seq_dir) / "img1" / f"{f:06d}.jpg"))
    tracks = run_tracker(dets, frames, "ocsort", frame_loader=bgr, enable_cmc=True, **params)
    crops = sample_crops(tracks)
    feats = crop_features(crops, frame_loader(seq_dir), "colour")
    return name, {"tracks": tracks, "crops": crops, "colour": feats}


def track_clips(jobs: dict[str, tuple[str | Path, pl.DataFrame]], workers: int | None = None,
                log: Callable[[str], None] | None = print,
                on_done: Callable[[str, dict[str, Any]], None] | None = None,
                **params: Any) -> dict[str, dict[str, Any]]:
    """``jobs[name] = (clip folder, detections)`` -> ``{name: {"tracks", "crops", "colour"}}``, clips in parallel
    ("spawn" workers, one thread each: forking after polars has started threads can hang)."""
    import multiprocessing as mp
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    workers = workers or os.cpu_count() or 1
    saved = {k: os.environ.get(k) for k in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS")}
    os.environ.update(POLARS_MAX_THREADS="1", OMP_NUM_THREADS="1")
    out, t0 = {}, time.time()
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            futs = [pool.submit(_cpu_clip, (n, str(s), d, params)) for n, (s, d) in jobs.items()]
            for i, fut in enumerate(as_completed(futs), 1):
                n, r = fut.result()
                out[n] = r
                if on_done:
                    on_done(n, r)  # e.g. save to Drive as each clip finishes
                if log:
                    log(f"{i}/{len(jobs)} clips tracked ({n}), {time.time() - t0:.0f} s")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return dict(sorted(out.items()))


def clip_game_state(cpu: dict[str, Any], hs: dict[int, np.ndarray | None] | None = None,
                    outlier_z: float | None = None, k: int = 2, pitch_margin: float | None = 5.0,
                    left_is_negative_x: bool = True, tracks_pitch: pl.DataFrame | None = None,
                    jerseys: dict[int, str | None] | None = None) -> pl.DataFrame:
    """Steps 4-6 for one clip from the CPU stage's output and per-frame homographies ``hs`` (or tracks already
    mapped to the pitch, ``tracks_pitch = to_pitch(cpu["tracks"], hs)``, to try role settings quickly)."""
    tp = tracks_pitch if tracks_pitch is not None else to_pitch(cpu["tracks"], hs)
    c = cpu["crops"]
    roles = assign_roles(tp, c["id"].to_numpy(), cpu["colour"], (c["y2"] - c["y1"]).to_numpy(), k=k,
                         outlier_z=outlier_z, left_is_negative_x=left_is_negative_x)
    return game_state(tp, roles, pitch_margin, jerseys)


def write_predictions(pred: pl.DataFrame, images: pl.DataFrame, path: str | Path) -> None:
    """One clip's predictions as the official SoccerNet GSR JSON (``{"predictions": [...]}``), for sn-trackeval."""
    import json

    ids = dict(zip(images["frame"].to_list(), images["image_id"].to_list(), strict=True))
    out = []
    for r in pred.filter(pl.col("pitch_x").is_not_null()).iter_rows(named=True):
        if r["frame"] not in ids:
            continue
        x, y = float(r["pitch_x"]), float(r["pitch_y"])
        out.append({
            "image_id": ids[r["frame"]], "track_id": int(r["id"]), "category_id": 1, "supercategory": "object",
            "confidence": 1.0,
            "attributes": {"role": r["role"], "team": r["team"], "jersey": r["jersey"]},
            "bbox_pitch": {"x_bottom_left": x, "y_bottom_left": y, "x_bottom_right": x, "y_bottom_right": y,
                           "x_bottom_middle": x, "y_bottom_middle": y},
            "bbox_image": {"x": r["x1"], "y": r["y1"], "w": r["x2"] - r["x1"], "h": r["y2"] - r["y1"]},
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"predictions": out}), encoding="utf-8")


# ---------------------------------------------------------------- one video, all steps


def video_to_frames(video: str | Path, out_dir: str | Path) -> Path:
    """Write a video's frames as ``out_dir/img1/000001.jpg, ...`` (the SN-GSR layout) and return ``out_dir``."""
    out = Path(out_dir)
    (out / "img1").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        i += 1
        cv2.imwrite(str(out / "img1" / f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    if i == 0:
        raise ValueError(f"No frames read from {video}")
    return out


def run_video(clip_dir: str | Path, detector: Any, keypoint_model: Any, device: str = "cuda",
              log: Callable[[str], None] | None = print) -> pl.DataFrame:
    """All steps on one clip folder (``img1/*.jpg``): the game state table (``GS_SCHEMA``)."""
    from pitchlens.vision.tracker import detect_sequence, sequence_frames

    frames = sequence_frames(clip_dir)
    if log:
        log(f"{len(frames)} frames: detecting people")
    dets = detect_sequence(detector, clip_dir)
    if log:
        log("pitch keypoints")
    hs, _ = fill_homographies(keypoint_homographies(predict_sequence_keypoints(keypoint_model, frames, device=device)),
                              [f for f, _ in frames])
    if log:
        log("tracking and teams")
    _, cpu = _cpu_clip(("clip", str(clip_dir), dets, {}))
    return clip_game_state(cpu, hs)
