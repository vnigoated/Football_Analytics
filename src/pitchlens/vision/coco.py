"""Convert SoccerNet-GSR labels to a COCO detection dataset (the format RF-DETR trains on).

Output layout (one folder per split)::

    out_dir/_annotations.coco.json
    out_dir/SNGS-060_000001.jpg   (symlink to the original frame, or a copy)

Consecutive frames at 25 fps are nearly identical, so only every ``every``-th frame is kept.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from pitchlens.data.gsr import find_sequences, load_sequence

CLASSES: tuple[str, ...] = ("player", "goalkeeper", "referee", "ball")
CATEGORY_IDS: dict[str, int] = {name: i + 1 for i, name in enumerate(CLASSES)}
ANNOTATION_FILE = "_annotations.coco.json"


def _place(src: Path, dst: Path, link: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if link:
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass  # e.g. Windows without symlink rights
    shutil.copy2(src, dst)


def clip_box(
    x: float, y: float, w: float, h: float, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Clip an (x, y, w, h) box to the image; None if less than 1 px remains."""
    x0, y0 = max(x, 0.0), max(y, 0.0)
    x1, y1 = min(x + w, float(width)), min(y + h, float(height))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return x0, y0, x1 - x0, y1 - y0


def gsr_to_coco(
    split_root: str | Path,
    out_dir: str | Path,
    every: int = 10,
    link: bool = True,
    copy_images: bool = True,
) -> dict[str, int]:
    """Write a COCO dataset for one SN-GSR split. Returns counts per class and totals."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    counts = dict.fromkeys(CLASSES, 0)
    for seq_dir in find_sequences(split_root):
        imgs, objs = load_sequence(seq_dir)
        imgs = imgs.filter(((imgs["frame"] - 1) % every) == 0)
        by_image = {}
        for o in objs.iter_rows(named=True):
            if o["role"] in CATEGORY_IDS and o["x"] is not None:
                by_image.setdefault(o["image_id"], []).append(o)
        for im in imgs.iter_rows(named=True):
            width, height = im["width"] or 1920, im["height"] or 1080
            name = f"{seq_dir.name}_{im['frame']:06d}.jpg"
            src = seq_dir / "img1" / im["file_name"]
            if copy_images:
                _place(src, out / name, link)
            img_id = len(images) + 1
            images.append({"id": img_id, "file_name": name, "width": width, "height": height})
            for o in by_image.get(im["image_id"], []):
                box = clip_box(o["x"], o["y"], o["w"], o["h"], width, height)
                if box is None:
                    continue
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": img_id,
                        "category_id": CATEGORY_IDS[o["role"]],
                        "bbox": [round(v, 2) for v in box],
                        "area": round(box[2] * box[3], 2),
                        "iscrowd": 0,
                    }
                )
                counts[o["role"]] += 1
    coco = {
        "info": {"description": f"SoccerNet-GSR {Path(split_root).name}, every {every}th frame"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": n, "supercategory": "object"} for n, i in CATEGORY_IDS.items()
        ],
    }
    (out / ANNOTATION_FILE).write_text(json.dumps(coco), encoding="utf-8")
    return {"images": len(images), "annotations": len(annotations), **counts}
