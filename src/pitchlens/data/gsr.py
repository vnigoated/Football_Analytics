"""Read SoccerNet Game State Reconstruction (SN-GSR) labels.

Each sequence folder (e.g. ``valid/SNGS-021``) holds ``img1/*.jpg`` (30-s clip, 25 fps) and
``Labels-GameState.json`` with three lists: ``images``, ``annotations`` and ``categories``.
Object annotations (``supercategory == "object"``) carry a ``track_id``, ``attributes``
(``role``: player / goalkeeper / referee / ball / other, ``team``: left / right, ``jersey``),
an image box ``bbox_image`` and, when known, the pitch position of the box's bottom middle
``bbox_pitch`` in metres (origin at the pitch centre).

The parser is lenient: missing fields become nulls, so a format change shows up as nulls in
the statistics rather than a crash.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

LABEL_FILE = "Labels-GameState.json"
PLAYER_ROLES = ("player", "goalkeeper")


def find_sequences(root: str | Path) -> list[Path]:
    """All sequence folders under ``root`` that contain a labels file, sorted by path."""
    return sorted(p.parent for p in Path(root).rglob(LABEL_FILE))


def _num(d: dict | None, key: str) -> float | None:
    if not d:
        return None
    v = d.get(key)
    return float(v) if isinstance(v, int | float) else None


def parse_labels(data: dict, sequence: str = "") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Turn one ``Labels-GameState.json`` dict into (images, objects) tables."""
    img_rows = []
    for im in data.get("images", []):
        name = str(im.get("file_name", ""))
        stem = Path(name).stem
        img_rows.append(
            {
                "sequence": sequence,
                "image_id": str(im.get("image_id")),
                "frame": int(stem) if stem.isdigit() else None,
                "file_name": name,
                "width": im.get("width"),
                "height": im.get("height"),
                "has_labeled_person": im.get("has_labeled_person"),
                "has_labeled_pitch": im.get("has_labeled_pitch"),
                "has_labeled_camera": im.get("has_labeled_camera"),
            }
        )
    obj_rows = []
    for a in data.get("annotations", []):
        if a.get("supercategory", "object") != "object":
            continue
        att = a.get("attributes") or {}
        box = a.get("bbox_image") or {}
        pitch = a.get("bbox_pitch") or {}
        obj_rows.append(
            {
                "sequence": sequence,
                "image_id": str(a.get("image_id")),
                "track_id": a.get("track_id"),
                "role": att.get("role"),
                "team": att.get("team"),
                "jersey": None if att.get("jersey") is None else str(att.get("jersey")),
                "x": _num(box, "x"),
                "y": _num(box, "y"),
                "w": _num(box, "w"),
                "h": _num(box, "h"),
                "pitch_x": _num(pitch, "x_bottom_middle"),
                "pitch_y": _num(pitch, "y_bottom_middle"),
            }
        )
    img_schema = {
        "sequence": pl.String,
        "image_id": pl.String,
        "frame": pl.Int64,
        "file_name": pl.String,
        "width": pl.Int64,
        "height": pl.Int64,
        "has_labeled_person": pl.Boolean,
        "has_labeled_pitch": pl.Boolean,
        "has_labeled_camera": pl.Boolean,
    }
    obj_schema = {
        "sequence": pl.String,
        "image_id": pl.String,
        "track_id": pl.Int64,
        "role": pl.String,
        "team": pl.String,
        "jersey": pl.String,
        **{k: pl.Float64 for k in ("x", "y", "w", "h", "pitch_x", "pitch_y")},
    }
    return (
        pl.DataFrame(img_rows, schema=img_schema),
        pl.DataFrame(obj_rows, schema=obj_schema),
    )


def load_sequence(seq_dir: str | Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    seq_dir = Path(seq_dir)
    data = json.loads((seq_dir / LABEL_FILE).read_text(encoding="utf-8"))
    return parse_labels(data, sequence=seq_dir.name)


def load_split(root: str | Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Concatenate all sequences found under ``root``."""
    parts = [load_sequence(s) for s in find_sequences(root)]
    if not parts:
        raise FileNotFoundError(f"No {LABEL_FILE} under {root}")
    return pl.concat([p[0] for p in parts]), pl.concat([p[1] for p in parts])


def visible_per_frame(images: pl.DataFrame, objects: pl.DataFrame) -> pl.DataFrame:
    """Per labelled frame: visible players (incl. goalkeepers), per team, referees, ball."""
    labelled = images.filter(pl.col("has_labeled_person").fill_null(True))
    counts = objects.group_by("sequence", "image_id").agg(
        players=pl.col("role").is_in(PLAYER_ROLES).sum(),
        left=(pl.col("role").is_in(PLAYER_ROLES) & (pl.col("team") == "left")).sum(),
        right=(pl.col("role").is_in(PLAYER_ROLES) & (pl.col("team") == "right")).sum(),
        referees=(pl.col("role") == "referee").sum(),
        ball=(pl.col("role") == "ball").any(),
        players_on_pitch=(pl.col("role").is_in(PLAYER_ROLES) & pl.col("pitch_x").is_not_null()).sum(),
    )
    out = labelled.select("sequence", "image_id", "frame").join(
        counts, on=["sequence", "image_id"], how="left"
    )
    return out.with_columns(
        pl.col("players", "left", "right", "referees", "players_on_pitch").fill_null(0),
        pl.col("ball").fill_null(False),
    ).sort("sequence", "frame")


def split_summary(images: pl.DataFrame, objects: pl.DataFrame) -> dict[str, float]:
    """Headline numbers for one split."""
    v = visible_per_frame(images, objects)
    p = v["players"]
    return {
        "sequences": images["sequence"].n_unique(),
        "frames": images.height,
        "labelled_frames": v.height,
        "objects": objects.height,
        "tracks": objects.select("sequence", "track_id").unique().height,
        "visible_players_mean": round(float(p.mean()), 2),
        "visible_players_median": float(p.median()),
        "visible_players_p10": float(p.quantile(0.1)),
        "visible_players_p90": float(p.quantile(0.9)),
        "ball_visible_share": round(float(v["ball"].mean()), 3),
        "players_with_pitch_pos_share": round(
            float(v["players_on_pitch"].sum() / max(int(p.sum()), 1)), 3
        ),
    }
