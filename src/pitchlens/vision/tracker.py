"""Multi-object tracking of people on broadcast frames, and its evaluation (Phase 2 · A3).

The detector runs once per sequence and its boxes are cached (``detect_sequence``); trackers then
run on the cached boxes, which takes seconds per sequence on a CPU, so trackers and their settings
can be compared cheaply. Trackers and metrics come from Roboflow's ``trackers`` package
(Apache-2.0): ByteTrack, OC-SORT and SORT, and HOTA / CLEAR (MOTA) / Identity (IDF1) metrics ported
from TrackEval.

Tables used here (polars):

* detections: ``frame, x1, y1, x2, y2, score, cls`` (cls: "person" or "ball")
* tracks / ground truth: ``frame, id, x1, y1, x2, y2`` (+ ``score`` for tracks)

Frames are numbered as in SN-GSR (``img1/000001.jpg`` is frame 1).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pitchlens.vision.coco import clip_box

PERSON_ROLES = ("player", "goalkeeper", "referee")
FRAME_RATE = 25.0
TRACKERS = ("bytetrack", "ocsort", "sort")
DET_SCHEMA = {
    "frame": pl.Int64,
    **{k: pl.Float64 for k in ("x1", "y1", "x2", "y2", "score")},
    "cls": pl.String,
}
TRACK_SCHEMA = {
    "frame": pl.Int64,
    "id": pl.Int64,
    **{k: pl.Float64 for k in ("x1", "y1", "x2", "y2", "score")},
}


# ---------------------------------------------------------------- ground truth and detections


def gt_tracks(
    images: pl.DataFrame, objects: pl.DataFrame, roles: Iterable[str] = PERSON_ROLES
) -> pl.DataFrame:
    """Ground-truth boxes of one sequence (from ``gsr.load_sequence``), clipped to the image."""
    roles = tuple(roles)
    frames = {
        r["image_id"]: (r["frame"], r["width"] or 1920, r["height"] or 1080)
        for r in images.iter_rows(named=True)
        if r["has_labeled_person"] is not False
    }
    rows = []
    for o in objects.iter_rows(named=True):
        if o["role"] not in roles or o["x"] is None or o["image_id"] not in frames:
            continue
        frame, width, height = frames[o["image_id"]]
        box = clip_box(o["x"], o["y"], o["w"], o["h"], width, height)
        if box is None:
            continue
        x, y, w, h = box
        rows.append({"frame": frame, "id": o["track_id"], "x1": x, "y1": y, "x2": x + w, "y2": y + h})
    schema = {k: v for k, v in TRACK_SCHEMA.items() if k != "score"}
    return pl.DataFrame(rows, schema=schema).sort("frame", "id")


def labelled_frames(images: pl.DataFrame) -> list[int]:
    """Frames with person labels: the frames a tracker is scored on."""
    return sorted(images.filter(pl.col("has_labeled_person").fill_null(True))["frame"].to_list())


def sequence_frames(seq_dir: str | Path) -> list[tuple[int, Path]]:
    """(frame number, path) of every image in ``seq_dir/img1``."""
    paths = sorted((Path(seq_dir) / "img1").glob("*.jpg"))
    return [(int(p.stem), p) for p in paths if p.stem.isdigit()]


def detect_sequence(
    model: Any,
    seq_dir: str | Path,
    threshold: float = 0.05,
    batch: int = 16,
    workers: int = 8,
) -> pl.DataFrame:
    """Run an RF-DETR model (``model.predict``) on every frame of a sequence.

    The low threshold keeps weak boxes: ByteTrack uses them in its second association step.
    """
    from PIL import Image

    def load(p: Path):
        return Image.open(p).convert("RGB")

    frames = sequence_frames(seq_dir)
    rows: list[dict] = []
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(frames), batch):
            chunk = frames[start : start + batch]
            pil = list(pool.map(load, [p for _, p in chunk]))
            preds = model.predict(pil, threshold=threshold)
            preds = preds if isinstance(preds, list) else [preds]
            for (frame, _), p in zip(chunk, preds, strict=True):
                names = p.data.get("class_name")
                for (x1, y1, x2, y2), s, n in zip(p.xyxy, p.confidence, names, strict=True):
                    rows.append(
                        {"frame": frame, "x1": float(x1), "y1": float(y1), "x2": float(x2),
                         "y2": float(y2), "score": float(s), "cls": str(n)}
                    )
    return pl.DataFrame(rows, schema=DET_SCHEMA)


def gt_as_detections(gt: pl.DataFrame) -> pl.DataFrame:
    """Ground-truth boxes as perfect detections (score 1): isolates association errors."""
    return gt.select("frame", "x1", "y1", "x2", "y2").with_columns(
        score=pl.lit(1.0), cls=pl.lit("person")
    )


# ---------------------------------------------------------------- tracking


def make_tracker(name: str = "bytetrack", frame_rate: float = FRAME_RATE, **params: Any):
    """A fresh tracker from the ``trackers`` package."""
    import trackers

    cls = {
        "bytetrack": trackers.ByteTrackTracker,
        "ocsort": trackers.OCSORTTracker,
        "sort": trackers.SORTTracker,
    }[name]
    return cls(frame_rate=frame_rate, **params)


def run_tracker(
    dets: pl.DataFrame,
    frames: Iterable[int],
    name: str = "bytetrack",
    cls: str = "person",
    frame_loader: Callable[[int], np.ndarray] | None = None,
    **params: Any,
) -> pl.DataFrame:
    """Track one sequence. ``frames`` lists every frame in order (also those without detections).

    ``frame_loader(frame) -> image`` is only needed for camera-motion compensation (OC-SORT with
    ``enable_cmc=True``).
    """
    import supervision as sv

    tracker = make_tracker(name, **params)
    d = dets.filter(pl.col("cls") == cls)
    by_frame = {k[0]: g for k, g in d.group_by("frame")}
    empty = np.zeros((0, 4), dtype=np.float32)
    rows = []
    for f in frames:
        g = by_frame.get(f)
        if g is None:
            det = sv.Detections(xyxy=empty, confidence=np.zeros(0, dtype=np.float32))
        else:
            det = sv.Detections(
                xyxy=g.select("x1", "y1", "x2", "y2").to_numpy().astype(np.float32),
                confidence=g["score"].to_numpy().astype(np.float32),
            )
        frame = frame_loader(f) if frame_loader is not None else None
        out = tracker.update(det, frame) if frame is not None else tracker.update(det)
        if out.tracker_id is None:
            continue
        conf = out.confidence if out.confidence is not None else np.ones(len(out))
        for (x1, y1, x2, y2), tid, s in zip(out.xyxy, out.tracker_id, conf, strict=True):
            if tid < 0:
                continue
            rows.append({"frame": f, "id": int(tid), "x1": float(x1), "y1": float(y1),
                         "x2": float(x2), "y2": float(y2), "score": float(s)})
    return pl.DataFrame(rows, schema=TRACK_SCHEMA).sort("frame", "id")


def interpolate_gaps(tracks: pl.DataFrame, max_gap: int = 10) -> pl.DataFrame:
    """Fill gaps of at most ``max_gap`` missing frames inside each track by linear interpolation.

    A player hidden behind another for a few frames keeps their box; longer gaps stay empty.
    """
    if max_gap <= 0 or tracks.is_empty():
        return tracks
    new = []
    for (tid,), g in tracks.sort("frame").group_by("id", maintain_order=True):
        f = g["frame"].to_numpy()
        boxes = g.select("x1", "y1", "x2", "y2").to_numpy()
        for i in np.nonzero((np.diff(f) > 1) & (np.diff(f) <= max_gap + 1))[0]:
            f0, f1 = f[i], f[i + 1]
            for t in range(f0 + 1, f1):
                a = (t - f0) / (f1 - f0)
                b = boxes[i] * (1 - a) + boxes[i + 1] * a
                new.append({"frame": int(t), "id": int(tid), "x1": b[0], "y1": b[1],
                            "x2": b[2], "y2": b[3], "score": 0.0})
    if not new:
        return tracks
    return pl.concat([tracks, pl.DataFrame(new, schema=TRACK_SCHEMA)]).sort("frame", "id")


# ---------------------------------------------------------------- evaluation


def evaluate(gt: pl.DataFrame, pred: pl.DataFrame, frames: Iterable[int]) -> dict[str, Any]:
    """HOTA, CLEAR (MOTA) and Identity (IDF1) metrics of one sequence on ``frames``.

    Returns the raw metric dicts (``hota``, ``clear``, ``identity``) so sequences can be combined
    with ``aggregate``.
    """
    from trackers.eval import (
        box_iou,
        compute_clear_metrics,
        compute_hota_metrics,
        compute_identity_metrics,
    )

    gt_f = {k[0]: g for k, g in gt.group_by("frame")}
    pr_f = {k[0]: g for k, g in pred.group_by("frame")}
    gt_ids, pr_ids, sims = [], [], []
    for f in frames:
        g, p = gt_f.get(f), pr_f.get(f)
        gi = g["id"].to_numpy().astype(np.int64) if g is not None else np.zeros(0, np.int64)
        pi = p["id"].to_numpy().astype(np.int64) if p is not None else np.zeros(0, np.int64)
        if len(gi) and len(pi):
            s = box_iou(g.select("x1", "y1", "x2", "y2").to_numpy(),
                        p.select("x1", "y1", "x2", "y2").to_numpy())
        else:
            s = np.zeros((len(gi), len(pi)))
        gt_ids.append(gi)
        pr_ids.append(pi)
        sims.append(s)
    return {
        "hota": compute_hota_metrics(gt_ids, pr_ids, sims),
        "clear": compute_clear_metrics(gt_ids, pr_ids, sims),
        "identity": compute_identity_metrics(gt_ids, pr_ids, sims),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-sequence ``evaluate`` results the way TrackEval does."""
    from trackers.eval import (
        aggregate_clear_metrics,
        aggregate_hota_metrics,
        aggregate_identity_metrics,
    )

    return {
        "hota": aggregate_hota_metrics([r["hota"] for r in results]),
        "clear": aggregate_clear_metrics([r["clear"] for r in results]),
        "identity": aggregate_identity_metrics([r["identity"] for r in results]),
    }


def headline(result: dict[str, Any]) -> dict[str, float]:
    """The numbers we report, rounded."""
    h, c, i = result["hota"], result["clear"], result["identity"]
    return {
        "HOTA": round(float(h["HOTA"]), 4),
        "DetA": round(float(h["DetA"]), 4),
        "AssA": round(float(h["AssA"]), 4),
        "IDF1": round(float(i["IDF1"]), 4),
        "MOTA": round(float(c["MOTA"]), 4),
        "IDSW": int(c["IDSW"]),
        "Frag": int(c["Frag"]),
    }


_GRID_DATA: dict[str, dict[str, Any]] = {}


def _grid_init(data: dict[str, dict[str, Any]]) -> None:
    _GRID_DATA.clear()
    _GRID_DATA.update(data)


def _grid_score(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    name, params = job
    m, _ = evaluate_split(_GRID_DATA, name, **params)
    return {**params, **m}


def grid_search(
    data: dict[str, dict[str, Any]],
    grid: list[dict[str, Any]],
    name: str = "bytetrack",
    workers: int | None = None,
    log: Callable[[str], None] | None = print,
) -> pl.DataFrame:
    """Score every setting in ``grid`` on ``data`` in parallel; best HOTA first.

    Workers are started with "spawn" and one thread each: forking a process that already
    uses polars' thread pool can hang.
    """
    import multiprocessing as mp
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    workers = workers or os.cpu_count() or 1
    saved = {k: os.environ.get(k) for k in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS")}
    os.environ.update(POLARS_MAX_THREADS="1", OMP_NUM_THREADS="1")  # inherited by the workers
    rows, t0 = [], time.time()
    try:
        with ProcessPoolExecutor(
            workers, mp_context=mp.get_context("spawn"), initializer=_grid_init, initargs=(data,)
        ) as pool:
            futures = [pool.submit(_grid_score, (name, p)) for p in grid]
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if log:
                    log(f"{i}/{len(grid)} done, HOTA {rows[-1]['HOTA']:.4f}, {time.time() - t0:.0f} s")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return pl.DataFrame(rows).sort("HOTA", descending=True)


def evaluate_split(
    data: dict[str, dict[str, Any]],
    name: str = "bytetrack",
    max_gap: int = 0,
    use_gt_boxes: bool = False,
    **params: Any,
) -> tuple[dict[str, float], pl.DataFrame]:
    """Track and score every sequence in ``data``.

    ``data[seq] = {"dets": detections, "gt": gt_tracks(...), "frames": all frames,
    "eval_frames": labelled frames}``. Returns (overall headline, per-sequence table).
    """
    results, rows = [], []
    for seq, d in data.items():
        dets = gt_as_detections(d["gt"]) if use_gt_boxes else d["dets"]
        tracks = interpolate_gaps(run_tracker(dets, d["frames"], name, **params), max_gap)
        r = evaluate(d["gt"], tracks, d["eval_frames"])
        results.append(r)
        rows.append({"sequence": seq, **headline(r), "tracks": tracks["id"].n_unique(),
                     "gt_tracks": d["gt"]["id"].n_unique()})
    return headline(aggregate(results)), pl.DataFrame(rows)
