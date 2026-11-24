"""Team assignment by appearance, per clip (Phase 2 · A4).

Kit colours differ from match to match, so nothing is learned across matches: within each clip, crops of the tracked
people are described by an appearance feature and clustered. The two largest clusters are the two teams; people in
the smaller clusters (referees, goalkeepers, whose kits differ from both teams) are labelled "other". Each track
gets the majority label of its crops.

Two appearance features are compared:

* ``colour``: a histogram of the shirt area in Lab colour space, with grass pixels removed (CPU, no model);
* ``siglip``: an image embedding from a pretrained SigLIP model (GPU), as in common football pipelines.

Evaluation against SN-GSR ``team`` (left / right) labels: predicted tracks are matched to ground-truth boxes per frame
(IoU >= 0.5), and the cluster-to-team mapping is chosen per clip (cluster numbers are arbitrary).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from pitchlens.vision.coco import clip_box

TEAM_LABELS = ("team_a", "team_b", "other")
SIGLIP_MODEL = "google/siglip-base-patch16-224"  # Apache-2.0


# ---------------------------------------------------------------- ground truth


def gt_people(images: pl.DataFrame, objects: pl.DataFrame) -> pl.DataFrame:
    """Ground-truth people of one clip with role and team: ``frame, id, role, team, x1..y2``."""
    frames = {
        r["image_id"]: (r["frame"], r["width"] or 1920, r["height"] or 1080)
        for r in images.iter_rows(named=True)
        if r["has_labeled_person"] is not False
    }
    rows = []
    for o in objects.iter_rows(named=True):
        if o["role"] not in ("player", "goalkeeper", "referee") or o["x"] is None:
            continue
        if o["image_id"] not in frames:
            continue
        frame, width, height = frames[o["image_id"]]
        box = clip_box(o["x"], o["y"], o["w"], o["h"], width, height)
        if box is None:
            continue
        x, y, w, h = box
        rows.append({"frame": frame, "id": o["track_id"], "role": o["role"], "team": o["team"],
                     "x1": x, "y1": y, "x2": x + w, "y2": y + h})
    schema = {"frame": pl.Int64, "id": pl.Int64, "role": pl.String, "team": pl.String,
              **{k: pl.Float64 for k in ("x1", "y1", "x2", "y2")}}
    return pl.DataFrame(rows, schema=schema).sort("frame", "id")


# ---------------------------------------------------------------- crops and features


def sample_crops(tracks: pl.DataFrame, every: int = 5, max_per_track: int = 10,
                 min_height: float = 24.0) -> pl.DataFrame:
    """Which (track, frame) boxes to describe: every ``every``-th frame, at most ``max_per_track`` per
    track (evenly spread), boxes at least ``min_height`` px tall. Tracks with no such box get their
    tallest box, so every track gets a label."""
    t = tracks.with_columns(h=pl.col("y2") - pl.col("y1"))
    pick = t.filter(((pl.col("frame") - 1) % every == 0) & (pl.col("h") >= min_height))
    pick = pick.with_columns(
        rank=pl.col("frame").rank("ordinal").over("id"), n=pl.len().over("id")
    ).filter(
        # evenly spaced subset of each track's candidate frames
        ((pl.col("rank") - 1) * max_per_track // pl.col("n"))
        != ((pl.col("rank") - 2) * max_per_track // pl.col("n"))
    )
    missing = t.filter(~pl.col("id").is_in(pick["id"].unique().implode()))
    tallest = missing.sort("h", descending=True).group_by("id", maintain_order=True).first()
    cols = ["frame", "id", "x1", "y1", "x2", "y2"]
    return pl.concat([pick.select(cols), tallest.select(cols)]).sort("frame", "id")


def shirt_region(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """The shirt area of a person box: rows 15-50% of the height, the middle 60% of the width."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    r0, r1 = int(y1 + 0.15 * h), int(y1 + 0.50 * h) + 1
    c0, c1 = int(x1 + 0.2 * w), int(x2 - 0.2 * w) + 1
    H, W = img.shape[:2]
    return img[max(r0, 0):min(r1, H), max(c0, 0):min(c1, W)]


def upper_body(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """The top 60% of a person box (shirt, shorts start; little grass): input for SigLIP."""
    x1, y1, x2, y2 = box
    H, W = img.shape[:2]
    r1 = int(y1 + 0.6 * (y2 - y1)) + 1
    return img[max(int(y1), 0):min(r1, H), max(int(x1), 0):min(int(x2) + 1, W)]


def colour_feature(rgb: np.ndarray, bins: int = 8) -> np.ndarray:
    """Lab histogram of the non-grass pixels of an RGB crop: 2-D (a, b) colour + 4 lightness bins.

    Square-rooted and L2-normalised, so Euclidean distance compares colour distributions (Hellinger).
    """
    import cv2

    if rgb.size == 0:
        return np.zeros(bins * bins + 4, dtype=np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 50) & (hsv[:, 2] >= 40)
    keep = lab[~grass] if (~grass).sum() >= 10 else lab
    ab, _, _ = np.histogram2d(keep[:, 1], keep[:, 2], bins=bins, range=[[0, 256], [0, 256]])
    light, _ = np.histogram(keep[:, 0], bins=4, range=(0, 256))
    f = np.concatenate([ab.ravel(), light.astype(float)]).astype(np.float32)
    f = np.sqrt(f / max(f.sum(), 1.0))
    return f / max(np.linalg.norm(f), 1e-9)


def crop_features(
    crops: pl.DataFrame,
    load_frame: Callable[[int], np.ndarray],
    feature: str = "colour",
    embedder: Callable[[list[np.ndarray]], np.ndarray] | None = None,
) -> np.ndarray:
    """Feature vector per row of ``crops`` (from ``sample_crops``). ``load_frame(frame)`` returns RGB.

    ``feature="siglip"`` needs ``embedder`` (see ``siglip_embedder``).
    """
    out: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    for (frame,), g in crops.group_by("frame", maintain_order=True):
        img = load_frame(frame)
        for r in g.iter_rows(named=True):
            box = (r["x1"], r["y1"], r["x2"], r["y2"])
            if feature == "colour":
                out.append(colour_feature(shirt_region(img, box)))
            else:
                batch.append(upper_body(img, box))
    if feature == "colour":
        return np.stack(out) if out else np.zeros((0, 68), np.float32)
    if embedder is None:
        raise ValueError("feature='siglip' needs an embedder")
    return embedder(batch)


def siglip_embedder(model_name: str = SIGLIP_MODEL, device: str = "cuda", batch: int = 256):
    """A function list[RGB crop] -> L2-normalised SigLIP image embeddings."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(model_name).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(model_name)  # image side only: no tokenizer needed

    @torch.no_grad()
    def embed(crops: list[np.ndarray]) -> np.ndarray:
        outs = []
        for i in range(0, len(crops), batch):
            ims = [Image.fromarray(c) if c.size else Image.new("RGB", (8, 8)) for c in crops[i:i + batch]]
            x = proc(images=ims, return_tensors="pt").to(device)
            e = model.get_image_features(**x)
            e = getattr(e, "pooler_output", e)  # newer transformers may return an output object
            outs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 768), np.float32)

    return embed


# ---------------------------------------------------------------- clustering


def pca(x: np.ndarray, dims: int) -> np.ndarray:
    """Project onto the first ``dims`` principal components (no-op if already small)."""
    if x.shape[1] <= dims or x.shape[0] <= dims:
        return x
    xc = x - x.mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:dims].T


def kmeans(x: np.ndarray, k: int, restarts: int = 8, iters: int = 100, seed: int = 0) -> np.ndarray:
    """Plain k-means with k-means++ starts; returns a label per row (best of ``restarts``)."""
    n = len(x)
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(restarts):
        c = [x[rng.integers(n)]]
        for _ in range(1, k):
            d = np.min(((x[:, None] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(x[rng.choice(n, p=d / d.sum()) if d.sum() > 0 else rng.integers(n)])
        c = np.array(c)
        for _ in range(iters):
            d = ((x[:, None] - c[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = d[np.arange(n), lab].sum()
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    return best


def assign_teams(crop_ids: np.ndarray, features: np.ndarray, k: int = 3, dims: int = 16) -> dict[int, str]:
    """Track id -> "team_a" / "team_b" / "other".

    Crops are clustered into ``k`` groups; the two largest are the teams (team_a the larger), the rest
    "other". ``k = 2`` puts everyone in a team. Each track takes its crops' majority label.
    """
    lab = kmeans(pca(features, dims), k)
    sizes = np.bincount(lab, minlength=k)
    order = np.argsort(-sizes)
    name = {int(order[0]): "team_a", int(order[1]): "team_b"} if k >= 2 else {int(order[0]): "team_a"}
    crop_label = np.array([name.get(int(j), "other") for j in lab])
    out = {}
    for tid in np.unique(crop_ids):
        vals, counts = np.unique(crop_label[crop_ids == tid], return_counts=True)
        out[int(tid)] = str(vals[counts.argmax()])
    return out


# ---------------------------------------------------------------- evaluation


def match_tracks_to_gt(tracks: pl.DataFrame, gt: pl.DataFrame, iou_thr: float = 0.5) -> pl.DataFrame:
    """Per frame, one-to-one matches (Hungarian on IoU >= ``iou_thr``): ``frame, track_id, gt_id``."""
    from scipy.optimize import linear_sum_assignment

    from pitchlens.vision.metrics import box_iou

    gt_f = {k[0]: g for k, g in gt.group_by("frame")}
    rows = []
    for (frame,), p in tracks.group_by("frame"):
        g = gt_f.get(frame)
        if g is None:
            continue
        iou = box_iou(p.select("x1", "y1", "x2", "y2").to_numpy(), g.select("x1", "y1", "x2", "y2").to_numpy())
        pid, gid = p["id"].to_numpy(), g["id"].to_numpy()
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c, strict=True):
            if iou[i, j] >= iou_thr:
                rows.append({"frame": int(frame), "track_id": int(pid[i]), "gt_id": int(gid[j])})
    return pl.DataFrame(rows, schema={"frame": pl.Int64, "track_id": pl.Int64, "gt_id": pl.Int64})


def evaluate_teams(labels: dict[int, str], matches: pl.DataFrame, gt: pl.DataFrame) -> dict[str, int]:
    """Box-level counts for one clip; ``team_a`` / ``team_b`` mapped to left / right the best way.

    * players: matched player boxes and how many got their own team;
    * keepers / referees: matched boxes and how many were labelled "other".
    """
    m = matches.join(
        gt.select(pl.col("frame"), pl.col("id").alias("gt_id"), "role", "team"), on=["frame", "gt_id"]
    ).with_columns(pred=pl.col("track_id").replace_strict(labels, default="other"))
    players = m.filter((pl.col("role") == "player") & pl.col("team").is_not_null())
    best = 0
    for a, b in (("left", "right"), ("right", "left")):
        ok = ((players["pred"] == "team_a") & (players["team"] == a)) | (
            (players["pred"] == "team_b") & (players["team"] == b)
        )
        best = max(best, int(ok.sum()))
    gk = m.filter(pl.col("role") == "goalkeeper")
    ref = m.filter(pl.col("role") == "referee")
    return {
        "player_boxes": players.height,
        "player_correct": best,
        "player_other": int((players["pred"] == "other").sum()),
        "keeper_boxes": gk.height,
        "keeper_other": int((gk["pred"] == "other").sum()),
        "referee_boxes": ref.height,
        "referee_other": int((ref["pred"] == "other").sum()),
    }


def summarise(counts: list[dict[str, int]]) -> dict[str, float]:
    """Totals over clips -> the shares we report."""
    t = {k: sum(c[k] for c in counts) for k in counts[0]}
    per_clip = [c["player_correct"] / c["player_boxes"] for c in counts if c["player_boxes"]]
    return {
        "player_team_accuracy": round(t["player_correct"] / max(t["player_boxes"], 1), 4),
        "player_as_other": round(t["player_other"] / max(t["player_boxes"], 1), 4),
        "worst_clip_accuracy": round(min(per_clip), 4) if per_clip else float("nan"),
        "clips_below_90pct": sum(a < 0.9 for a in per_clip),
        "referee_as_other": round(t["referee_other"] / max(t["referee_boxes"], 1), 4),
        "keeper_as_other": round(t["keeper_other"] / max(t["keeper_boxes"], 1), 4),
    }


def frame_loader(seq_dir: str | Path) -> Callable[[int], np.ndarray]:
    """Frame number -> RGB image of one SN-GSR clip."""
    import cv2

    seq_dir = Path(seq_dir)

    def load(frame: int) -> np.ndarray:
        img = cv2.imread(str(seq_dir / "img1" / f"{frame:06d}.jpg"))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return load
