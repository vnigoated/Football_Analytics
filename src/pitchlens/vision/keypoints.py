"""Pitch keypoints -> image-to-pitch homography (Phase 2 · A5).

Training labels come for free from SN-GSR: every labelled person has a pitch position (metres) for the bottom
middle of their box, computed by the annotators' camera model. From 4+ such people in a frame a ground-truth
homography is fitted (``frame_homography``); projecting known pitch landmarks through its inverse gives their
pixel positions (``project_keypoints``). A heatmap network then learns to find those landmarks in the image, and
at test time a RANSAC homography is fitted to the detected ones (``homography_from_keypoints``).

Coordinates: SN-GSR pitch metres, origin at the centre, x along the length (±52.5), y along the width (±34).
The main score is what matters downstream: **player position error in metres** (``position_errors``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl

from pitchlens.geometry.homography import apply_homography
from pitchlens.geometry.pitch import Pitch

PITCH = Pitch()
_HALF = np.array([PITCH.length / 2, PITCH.width / 2])


def _keypoints() -> tuple[list[str], np.ndarray]:
    lm = dict(PITCH.landmarks())
    # where the penalty arcs ("D") meet the penalty-area line: useful when only the box is visible
    dy = float(np.sqrt(PITCH.centre_circle_radius**2 - (PITCH.penalty_area_depth - PITCH.penalty_spot_distance) ** 2))
    cy = PITCH.width / 2
    lm["left_arc_bottom"] = (PITCH.penalty_area_depth, cy - dy)
    lm["left_arc_top"] = (PITCH.penalty_area_depth, cy + dy)
    lm["right_arc_bottom"] = (PITCH.length - PITCH.penalty_area_depth, cy - dy)
    lm["right_arc_top"] = (PITCH.length - PITCH.penalty_area_depth, cy + dy)
    names = list(lm)
    return names, np.array([lm[n] for n in names]) - _HALF  # centre origin, as in SN-GSR


KEYPOINT_NAMES, KEYPOINTS_M = _keypoints()
NUM_KEYPOINTS = len(KEYPOINT_NAMES)  # 33
INPUT_SIZE = (960, 544)  # network input (w, h); divisible by 32
STRIDE = 4  # heatmap resolution: 240 x 136


# ---------------------------------------------------------------- ground-truth homography and keypoints


def frame_homography(
    objects: pl.DataFrame, use_raw: bool = True, min_points: int = 6, ransac_m: float = 0.5
) -> tuple[np.ndarray | None, int, float]:
    """Image -> pitch homography of one frame from people's box bottoms and their labelled positions.

    Returns (H or None, number of inlier people, mean inlier residual in metres).
    """
    px, py = ("pitch_raw_x", "pitch_raw_y") if use_raw else ("pitch_x", "pitch_y")
    o = objects.filter(
        pl.col("role").is_in(["player", "goalkeeper", "referee"])
        & pl.col(px).is_not_null() & pl.col("x").is_not_null()
        & (pl.col(px).abs() < 70) & (pl.col(py).abs() < 50)
    )
    if o.height < min_points:
        return None, o.height, float("nan")
    img = np.column_stack([(o["x"] + o["w"] / 2).to_numpy(), (o["y"] + o["h"]).to_numpy()])
    pit = o.select(px, py).to_numpy()
    H, mask = cv2.findHomography(img, pit, cv2.RANSAC, ransac_m)
    if H is None:
        return None, 0, float("nan")
    inl = mask.ravel().astype(bool)
    if inl.sum() < 4:
        return None, int(inl.sum()), float("nan")
    err = np.linalg.norm(apply_homography(H, img[inl]) - pit[inl], axis=1)
    return H, int(inl.sum()), float(err.mean())


def project_keypoints(H: np.ndarray, width: int, height: int, margin: float = 0.0) -> np.ndarray:
    """(K, 3) pixel positions of the pitch keypoints and a visible flag (inside the image, in front of the camera)."""
    Hinv = np.linalg.inv(H)
    pts = np.hstack([KEYPOINTS_M, np.ones((NUM_KEYPOINTS, 1))]) @ Hinv.T
    # the side of the horizon the visible pitch is on: that of the image's bottom centre
    ref = np.array([width / 2, height - 1.0, 1.0])
    ref_pitch = H @ ref
    w_ref = (Hinv @ (ref_pitch / ref_pitch[2]))[2]
    front = np.sign(pts[:, 2]) == np.sign(w_ref)
    uv = pts[:, :2] / pts[:, 2:3]
    inside = (uv[:, 0] >= -margin) & (uv[:, 0] < width + margin) & (uv[:, 1] >= -margin) & (uv[:, 1] < height + margin)
    return np.column_stack([uv, (front & inside).astype(float)])


def build_keypoint_items(
    seq_dirs: list[Path], every: int = 10, use_raw: bool = True, min_visible: int = 0
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Training / evaluation items ``{"image", "keypoints" (K, 3), "H", "sequence", "frame"}`` for every
    ``every``-th frame with a ground-truth homography, plus statistics."""
    from pitchlens.data.gsr import load_sequence

    items, n_frames, n_no_h, residuals = [], 0, 0, []
    for seq in seq_dirs:
        images, objects = load_sequence(seq)
        by_img = {k[0]: g for k, g in objects.group_by("image_id")}
        for im in images.filter(((pl.col("frame") - 1) % every) == 0).iter_rows(named=True):
            n_frames += 1
            o = by_img.get(im["image_id"])
            H, _, res = frame_homography(o, use_raw) if o is not None else (None, 0, float("nan"))
            if H is None:
                n_no_h += 1
                continue
            kp = project_keypoints(H, im["width"] or 1920, im["height"] or 1080)
            if kp[:, 2].sum() < min_visible:
                continue
            residuals.append(res)
            items.append({"image": str(seq / "img1" / im["file_name"]), "keypoints": kp, "H": H,
                          "sequence": seq.name, "frame": im["frame"]})
    vis = np.array([it["keypoints"][:, 2].sum() for it in items]) if items else np.zeros(1)
    stats = {"frames": n_frames, "items": len(items), "no_homography": n_no_h,
             "residual_m_median": round(float(np.median(residuals)), 3) if residuals else float("nan"),
             "visible_keypoints_mean": round(float(vis.mean()), 2),
             "frames_with_4plus_keypoints": round(float((vis >= 4).mean()), 3)}
    return items, stats


def save_items(items: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text(json.dumps([{**it, "keypoints": it["keypoints"].tolist(), "H": it["H"].tolist()}
                                      for it in items]), encoding="utf-8")


def load_items(path: str | Path) -> list[dict[str, Any]]:
    return [{**it, "keypoints": np.array(it["keypoints"]), "H": np.array(it["H"])}
            for it in json.loads(Path(path).read_text(encoding="utf-8"))]


# ---------------------------------------------------------------- heatmaps


def heatmap_targets(keypoints: np.ndarray, src_size: tuple[int, int] = (1920, 1080),
                    sigma: float = 2.0) -> np.ndarray:
    """(K, h, w) Gaussian heatmaps (peak 1) at the network's output resolution; empty for invisible keypoints."""
    out_w, out_h = INPUT_SIZE[0] // STRIDE, INPUT_SIZE[1] // STRIDE
    sx, sy = out_w / src_size[0], out_h / src_size[1]
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    hm = np.zeros((len(keypoints), out_h, out_w), np.float32)
    for k, (u, v, vis) in enumerate(keypoints):
        if vis > 0:
            cx, cy = u * sx, v * sy
            hm[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    return hm


def decode_heatmaps(hm: np.ndarray, src_size: tuple[int, int] = (1920, 1080),
                    threshold: float = 0.3) -> np.ndarray:
    """(K, 3) keypoints in source pixels from (K, h, w) heatmaps: peak position (3 × 3 weighted refinement),
    peak value as score; score 0 below ``threshold``."""
    k, h, w = hm.shape
    out = np.zeros((k, 3))
    sx, sy = src_size[0] / w, src_size[1] / h
    for i in range(k):
        j = int(hm[i].argmax())
        y, x = divmod(j, w)
        score = float(hm[i, y, x])
        if score < threshold:
            continue
        y0, y1, x0, x1 = max(y - 1, 0), min(y + 2, h), max(x - 1, 0), min(x + 2, w)
        patch = np.clip(hm[i, y0:y1, x0:x1], 0, None)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cy, cx = (patch * yy).sum() / patch.sum(), (patch * xx).sum() / patch.sum()
        out[i] = [cx * sx, cy * sy, score]  # same convention as heatmap_targets
    return out


def homography_from_keypoints(kp: np.ndarray, min_points: int = 4,
                              ransac_m: float = 1.0) -> tuple[np.ndarray | None, int]:
    """Image -> pitch homography from detected keypoints (score > 0); (None, n) if too few or degenerate."""
    ok = kp[:, 2] > 0
    if ok.sum() < min_points:
        return None, int(ok.sum())
    H, mask = cv2.findHomography(kp[ok, :2], KEYPOINTS_M[ok], cv2.RANSAC, ransac_m)
    if H is None or mask.sum() < min_points:
        return None, int(ok.sum())
    return H, int(mask.sum())


def position_errors(H: np.ndarray, objects: pl.DataFrame, use_raw: bool = True) -> np.ndarray:
    """Pitch-position error (metres) of each labelled person when their box bottom is mapped with ``H``."""
    px, py = ("pitch_raw_x", "pitch_raw_y") if use_raw else ("pitch_x", "pitch_y")
    o = objects.filter(pl.col("role").is_in(["player", "goalkeeper", "referee"])
                       & pl.col(px).is_not_null() & pl.col("x").is_not_null())
    if o.is_empty():
        return np.zeros(0)
    img = np.column_stack([(o["x"] + o["w"] / 2).to_numpy(), (o["y"] + o["h"]).to_numpy()])
    return np.linalg.norm(apply_homography(H, img) - o.select(px, py).to_numpy(), axis=1)


def summarise_errors(errors: list[np.ndarray], frames_total: int, frames_with_h: int) -> dict[str, float]:
    e = np.concatenate(errors) if errors else np.zeros(0)
    return {
        "frames": frames_total,
        "frames_with_homography": round(frames_with_h / max(frames_total, 1), 3),
        "people": len(e),
        "error_m_median": round(float(np.median(e)), 2) if len(e) else float("nan"),
        "error_m_mean": round(float(e.mean()), 2) if len(e) else float("nan"),
        "error_m_p90": round(float(np.quantile(e, 0.9)), 2) if len(e) else float("nan"),
        "within_1m": round(float((e <= 1).mean()), 3) if len(e) else float("nan"),
        "within_2m": round(float((e <= 2).mean()), 3) if len(e) else float("nan"),
    }


# ---------------------------------------------------------------- network (PyTorch)


def build_model(num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = True):
    """ResNet-34 (ImageNet, BSD) + 3 upsampling stages -> heatmaps at stride 4 ("simple baseline" design)."""
    import torchvision
    from torch import nn

    weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    r = torchvision.models.resnet34(weights=weights)
    layers: list[nn.Module] = []
    ch = 512
    for out in (256, 256, 256):
        layers += [nn.ConvTranspose2d(ch, out, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(out),
                   nn.ReLU(inplace=True)]
        ch = out
    return nn.Sequential(
        r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4,  # stride 32
        *layers,  # stride 4
        nn.Conv2d(ch, num_keypoints, 1),
    )


_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_input(path: str, rng: np.random.Generator | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    """(3, h, w) normalised network input from an image file, and the source size. ``rng`` adds colour jitter."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    src = (img.shape[1], img.shape[0])
    x = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if rng is not None:  # brightness, contrast, saturation
        x = x * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)
        grey = x.mean(2, keepdims=True)
        x = grey + (x - grey) * rng.uniform(0.7, 1.3)
        x = np.clip(x, 0, 1)
    return ((x - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32), src


def make_dataset(items: list[dict[str, Any]], augment: bool):
    import torch

    class KeypointDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            it = items[i]
            rng = np.random.default_rng() if augment else None
            x, src = load_input(it["image"], rng)
            return torch.from_numpy(x), torch.from_numpy(heatmap_targets(it["keypoints"], src))

    return KeypointDataset()


def heatmap_loss(pred, target):
    """MSE with extra weight near keypoints (most heatmap pixels are empty)."""
    return (((pred - target) ** 2) * (1 + 20 * target)).mean()


def train_keypoints(
    train_items: list[dict[str, Any]],
    valid_items: list[dict[str, Any]],
    out_dir: str | Path,
    epochs: int = 30,
    batch: int = 16,
    lr: float = 1e-3,
    workers: int = 8,
    device: str = "cuda",
    log=print,
) -> Path:
    """Train the heatmap network; keeps ``last.pt`` (resumable) and ``best.pt`` (lowest valid loss) in ``out_dir``."""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(len(train_items) // batch, 1)
    start, best = 0, float("inf")
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["epoch"] + 1, ck["best"]
        log(f"resuming at epoch {start}")
    # the schedule is rebuilt for this call's epochs and fast-forwarded, so a resume may change `epochs`
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(epochs, start + 1) * steps_per_epoch,
                                                pct_start=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "scheduler.step() before optimizer.step()"
        for _ in range(start * steps_per_epoch):
            sched.step()
    dl = torch.utils.data.DataLoader(make_dataset(train_items, True), batch_size=batch, shuffle=True,
                                     num_workers=workers, drop_last=True, pin_memory=True, persistent_workers=workers > 0)
    vl = torch.utils.data.DataLoader(make_dataset(valid_items, False), batch_size=batch, num_workers=workers)
    for epoch in range(start, epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16):
                p = model(x)
            loss = heatmap_loss(p.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        vtot = 0.0
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for x, y in vl:
                vtot += heatmap_loss(model(x.to(device)).float(), y.to(device)).item()
        tr, va = tot / len(dl), vtot / max(len(vl), 1)
        if va < best:
            best = va
            torch.save(model.state_dict(), out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best": best},
                   out / "last.pt")
        log(f"epoch {epoch + 1}/{epochs}: train {tr:.5f}, valid {va:.5f}{' (best)' if va <= best else ''}")
    return out / "best.pt"


def load_model(path: str | Path, device: str = "cuda"):
    import torch

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def predict_keypoints(model, paths: list[str], device: str = "cuda", threshold: float = 0.3) -> list[np.ndarray]:
    """Detected keypoints (K, 3) per image path (source pixels, score)."""
    import torch

    xs, srcs = zip(*(load_input(p) for p in paths), strict=True)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        hm = model(torch.from_numpy(np.stack(xs)).to(device)).float().cpu().numpy()
    return [decode_heatmaps(h, s, threshold) for h, s in zip(hm, srcs, strict=True)]
