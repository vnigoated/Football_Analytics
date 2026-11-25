"""Jersey numbers from player crops, voted per track (Phase 2 · A7).

SN-GSR labels each player track with its shirt number when the number is visible at least once in the clip (87% of
player boxes have one). Those labels train a crop classifier directly: a ResNet-34 with two heads, the tens digit
(0 = none, i.e. a one-digit number) and the units digit, so every number 1-99 is covered even if rare. The labels are
per track, so many training crops do not show the number (player facing the camera, motion blur); the network
learns to be unsure on those, and the vote per track uses only confident crops.

Why the vote may say "no number": in GS-HOTA (official rules) a player whose number was never visible has a null
number, and only a null prediction matches it. A track therefore gets a number only when enough crops agree.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

CROP_SIZE = (96, 128)  # network input (w, h)
NUM_NUMBERS = 100  # 0..99; "0" itself does not occur, one-digit numbers have tens digit 0


def number_to_digits(number: str | int | None) -> tuple[int, int] | None:
    """"7" -> (0, 7), "23" -> (2, 3); None for missing or out-of-range numbers."""
    try:
        n = int(number)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not 1 <= n <= 99:
        return None
    return n // 10, n % 10


def jersey_crop(img: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """The torso of a person box (rows 5-65% of the height, the full width plus 10% on each side), resized to
    ``CROP_SIZE``: front and back numbers sit there."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    r0, r1 = int(max(y1 + 0.05 * h, 0)), int(min(y1 + 0.65 * h, H))
    c0, c1 = int(max(x1 - 0.1 * w, 0)), int(min(x2 + 0.1 * w, W))
    if r1 <= r0 or c1 <= c0:
        return np.zeros((CROP_SIZE[1], CROP_SIZE[0], 3), np.uint8)
    return cv2.resize(img[r0:r1, c0:c1], CROP_SIZE, interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- training data from SN-GSR labels


def build_jersey_items(
    seq_dirs: Sequence[str | Path],
    out_dir: str | Path,
    every: int = 5,
    min_height: float = 50.0,
    workers: int = 8,
    log: Callable[[str], None] | None = print,
) -> list[dict[str, Any]]:
    """Save torso crops of labelled players and goalkeepers with a number (every ``every``-th frame, boxes at least
    ``min_height`` px tall) as JPEGs in ``out_dir``; returns ``{"image", "number", "sequence", "track"}`` items."""
    from pitchlens.data.gsr import load_sequence

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []

    def one_frame(args):
        seq_dir, frame, rows = args
        img = cv2.imread(str(Path(seq_dir) / "img1" / f"{frame:06d}.jpg"))
        res = []
        if img is None:
            return res
        for r in rows:
            name = out / f"{Path(seq_dir).name}_{frame:06d}_{r['track_id']}.jpg"
            if not name.exists():
                cv2.imwrite(str(name), jersey_crop(img, (r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"])),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            res.append({"image": str(name), "number": str(int(r["jersey"])), "sequence": Path(seq_dir).name,
                        "track": int(r["track_id"])})
        return res

    for i, seq in enumerate(seq_dirs, 1):
        images, objects = load_sequence(seq)
        frame_of = dict(zip(images["image_id"].to_list(), images["frame"].to_list(), strict=True))
        o = objects.filter(pl.col("role").is_in(["player", "goalkeeper"]) & pl.col("jersey").is_not_null()
                           & (pl.col("h") >= min_height)).with_columns(
            frame=pl.col("image_id").replace_strict(frame_of, default=None, return_dtype=pl.Int64))
        o = o.filter(pl.col("frame").is_not_null() & ((pl.col("frame") - 1) % every == 0))
        o = o.filter(pl.col("jersey").map_elements(lambda j: number_to_digits(j) is not None, return_dtype=pl.Boolean))
        jobs = [(seq, int(f), g.to_dicts()) for (f,), g in o.group_by("frame")]
        with ThreadPoolExecutor(workers) as pool:
            for res in pool.map(one_frame, jobs):
                items.extend(res)
        if log and (i % 10 == 0 or i == len(seq_dirs)):
            log(f"{i}/{len(seq_dirs)} clips, {len(items)} crops")
    return items


# ---------------------------------------------------------------- network (PyTorch)


def build_model(pretrained: bool = True):
    """ResNet-34 (ImageNet) with one 20-way linear layer: tens digit (10, 0 = none) + units digit (10)."""
    import torchvision
    from torch import nn

    weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    r = torchvision.models.resnet34(weights=weights)
    r.fc = nn.Linear(r.fc.in_features, 20)
    return r


_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_input(rgb: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """(3, h, w) normalised input from an RGB crop of ``CROP_SIZE``; ``rng`` adds colour jitter and a small shift /
    zoom (no flips: mirrored digits are other digits)."""
    x = rgb.astype(np.float32) / 255.0
    if rng is not None:
        w, h = CROP_SIZE
        s = rng.uniform(0.9, 1.1)
        m = np.array([[s, 0, (1 - s) * w / 2 + rng.uniform(-0.06, 0.06) * w],
                      [0, s, (1 - s) * h / 2 + rng.uniform(-0.06, 0.06) * h]], np.float32)
        x = cv2.warpAffine(x, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        x = x * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)
        grey = x.mean(2, keepdims=True)
        x = np.clip(grey + (x - grey) * rng.uniform(0.6, 1.4), 0, 1)
    return ((x - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)


def make_dataset(items: list[dict[str, Any]], augment: bool):
    import torch

    class JerseyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            it = items[i]
            rgb = cv2.cvtColor(cv2.imread(it["image"]), cv2.COLOR_BGR2RGB)
            x = to_input(rgb, np.random.default_rng() if augment else None)
            t, u = number_to_digits(it["number"])
            return torch.from_numpy(x), torch.tensor([t, u])

    return JerseyDataset()


def digit_loss(logits, target):
    import torch.nn.functional as F

    return F.cross_entropy(logits[:, :10], target[:, 0]) + F.cross_entropy(logits[:, 10:], target[:, 1])


def train_jersey(
    train_items: list[dict[str, Any]],
    valid_items: list[dict[str, Any]],
    out_dir: str | Path,
    epochs: int = 12,
    batch: int = 256,
    lr: float = 2e-3,
    workers: int = 8,
    device: str = "cuda",
    log: Callable[[str], None] = print,
) -> Path:
    """Train the digit classifier; ``last.pt`` (resumable) and ``best.pt`` (best valid crop accuracy) in ``out_dir``.

    Crop accuracy is low by nature (many crops do not show the number); the track vote is what counts.
    """
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(len(train_items) // batch, 1)
    start, best = 0, -1.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["epoch"] + 1, ck["best"]
        log(f"resuming at epoch {start}")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(epochs, start + 1) * steps,
                                                pct_start=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(start * steps):
            sched.step()
    dl = torch.utils.data.DataLoader(make_dataset(train_items, True), batch_size=batch, shuffle=True,
                                     num_workers=workers, drop_last=True, pin_memory=True,
                                     persistent_workers=workers > 0)
    vl = torch.utils.data.DataLoader(make_dataset(valid_items, False), batch_size=batch, num_workers=workers)
    for epoch in range(start, epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(x)
            loss = digit_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        ok = n = 0
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for x, y in vl:
                lg = model(x.to(device)).float()
                pred = torch.stack([lg[:, :10].argmax(1), lg[:, 10:].argmax(1)], 1).cpu()
                ok += int((pred == y).all(1).sum())
                n += len(y)
        acc = ok / max(n, 1)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best": best},
                   out / "last.pt")
        log(f"epoch {epoch + 1}/{epochs}: train loss {tot / len(dl):.4f}, valid crop accuracy {acc:.4f}"
            f"{' (best)' if acc >= best else ''}")
    return out / "best.pt"


def load_model(path: str | Path, device: str = "cuda"):
    import torch

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def number_probs(logits: np.ndarray) -> np.ndarray:
    """(N, 20) digit logits -> (N, 100) probabilities of the numbers 0..99 (index = number)."""
    def softmax(z):
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    t, u = softmax(logits[:, :10]), softmax(logits[:, 10:])
    return (t[:, :, None] * u[:, None, :]).reshape(len(logits), NUM_NUMBERS)


def predict_crops(model: Any, crops: list[np.ndarray], device: str = "cuda", batch: int = 512) -> np.ndarray:
    """(N, 100) number probabilities for RGB crops of ``CROP_SIZE``."""
    import torch

    outs = []
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(np.stack([to_input(c) for c in crops[i:i + batch]])).to(device)
            outs.append(model(x).float().cpu().numpy())
    return number_probs(np.concatenate(outs)) if outs else np.zeros((0, NUM_NUMBERS))


# ---------------------------------------------------------------- per track


def sample_jersey_crops(tracks: pl.DataFrame, every: int = 2, max_per_track: int = 40,
                        min_height: float = 40.0) -> pl.DataFrame:
    """Which boxes of each track to read: every ``every``-th frame, at least ``min_height`` px tall, at most
    ``max_per_track`` (the tallest)."""
    t = tracks.with_columns(h=pl.col("y2") - pl.col("y1"))
    t = t.filter(((pl.col("frame") - 1) % every == 0) & (pl.col("h") >= min_height))
    t = t.sort("h", descending=True).group_by("id", maintain_order=True).head(max_per_track)
    return t.select("frame", "id", "x1", "y1", "x2", "y2").sort("frame", "id")


def crops_for(boxes: pl.DataFrame, load_frame: Callable[[int], np.ndarray], workers: int = 8) -> list[np.ndarray]:
    """RGB crops for the rows of ``boxes`` (``frame, x1, y1, x2, y2``), in row order; frames read in parallel."""
    out: list[np.ndarray | None] = [None] * boxes.height
    groups = [(int(f), g) for (f,), g in boxes.with_row_index("_i").group_by("frame", maintain_order=True)]

    def one(fg):
        frame, g = fg
        img = load_frame(frame)
        return [(r["_i"], jersey_crop(img, (r["x1"], r["y1"], r["x2"], r["y2"]))) for r in g.iter_rows(named=True)]

    with ThreadPoolExecutor(workers) as pool:
        for res in pool.map(one, groups):
            for i, c in res:
                out[i] = c
    return out  # type: ignore[return-value]


def vote_numbers(track_ids: np.ndarray, probs: np.ndarray, min_conf: float = 0.5,
                 min_votes: int = 1) -> dict[int, str | None]:
    """Track id -> number (string) or None.

    Crops whose best number has probability >= ``min_conf`` vote with their probabilities; the track gets the number
    with the largest total if at least ``min_votes`` crops voted, otherwise None.
    """
    out: dict[int, str | None] = {}
    conf = probs.max(1) if len(probs) else np.zeros(0)
    for t in np.unique(track_ids):
        m = (track_ids == t) & (conf >= min_conf)
        if m.sum() < max(min_votes, 1):
            out[int(t)] = None
            continue
        score = probs[m].sum(0)
        score[0] = 0  # "0" is not a shirt number
        out[int(t)] = str(int(score.argmax()))
    return out


def track_accuracy(pred: dict[int, str | None], truth: dict[int, str | None]) -> dict[str, float]:
    """Per track: share right (null = null counts), share of numbered tracks read correctly, wrong-number share."""
    keys = [k for k in truth if k in pred]
    if not keys:
        return {"tracks": 0, "correct": float("nan"), "numbered_read": float("nan"), "wrong_number": float("nan")}
    numbered = [k for k in keys if truth[k] is not None]
    return {
        "tracks": len(keys),
        "correct": round(float(np.mean([pred[k] == truth[k] for k in keys])), 4),
        "numbered_read": round(float(np.mean([pred[k] == truth[k] for k in numbered])), 4) if numbered else float("nan"),
        "wrong_number": round(float(np.mean([pred[k] is not None and pred[k] != truth[k] for k in keys])), 4),
    }
