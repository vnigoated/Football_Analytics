"""SkillCorner open data: real broadcast tracking (A-League 2024/25, 10 fps).

Each frame has the camera footprint on the pitch (``image_corners_projection``) and,
for every player, whether they were actually on screen (``is_detected``) or
extrapolated. We use it to fit and validate the virtual broadcast camera.

Licence: MIT (SkillCorner/opendata, checked 2026-11-22). Please credit SkillCorner.
Files are downloaded at run time into the git-ignored ``data/`` folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"
CORNERS = (
    "x_bottom_left", "y_bottom_left", "x_top_left", "y_top_left",
    "x_bottom_right", "y_bottom_right", "x_top_right", "y_top_right",
)  # fmt: skip


def _fetch(url: str, path: Path) -> Path:
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def skillcorner_match_ids(dest: str | Path = "data/skillcorner") -> list[int]:
    path = _fetch(f"{_RAW}/matches.json", Path(dest) / "matches.json")
    return [m["id"] for m in json.loads(path.read_text(encoding="utf-8"))]


def load_skillcorner(
    match_id: int, dest: str | Path = "data/skillcorner"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(frames, players)`` for one match, cached as Parquet next to the raw files.

    frames:  frame, period, ball_x, ball_y, ball_detected, possession, 8 corner columns,
             pitch_length, pitch_width (only frames in play with a footprint and a ball)
    players: frame, player_id, x, y, is_detected
    """
    dest = Path(dest)
    f_cache, p_cache = dest / f"{match_id}_frames.parquet", dest / f"{match_id}_players.parquet"
    if f_cache.exists() and p_cache.exists():
        return pl.read_parquet(f_cache), pl.read_parquet(p_cache)

    meta = json.loads(
        _fetch(f"{_RAW}/matches/{match_id}/{match_id}_match.json", dest / f"{match_id}_match.json")
        .read_text(encoding="utf-8")
    )
    raw = _fetch(
        f"{_LFS}/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl",
        dest / f"{match_id}_tracking_extrapolated.jsonl",
    )
    frames, players = [], {"frame": [], "player_id": [], "x": [], "y": [], "is_detected": []}
    with raw.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            c, b = r["image_corners_projection"], r["ball_data"]
            if r["period"] is None or c["x_top_left"] is None or b["x"] is None:
                continue
            frames.append(
                {
                    "frame": r["frame"],
                    "period": r["period"],
                    "ball_x": b["x"],
                    "ball_y": b["y"],
                    "ball_detected": bool(b["is_detected"]),
                    "possession": r["possession"]["group"],
                    **{k: c[k] for k in CORNERS},
                }
            )
            for p in r["player_data"]:
                players["frame"].append(r["frame"])
                players["player_id"].append(p["player_id"])
                players["x"].append(p["x"])
                players["y"].append(p["y"])
                players["is_detected"].append(bool(p["is_detected"]))
    fdf = pl.DataFrame(frames, infer_schema_length=None).with_columns(
        pl.lit(float(meta.get("pitch_length") or 105)).alias("pitch_length"),
        pl.lit(float(meta.get("pitch_width") or 68)).alias("pitch_width"),
        pl.lit(match_id).alias("match_id"),
    )
    pdf = pl.DataFrame(players).with_columns(pl.lit(match_id).alias("match_id"))
    fdf.write_parquet(f_cache)
    pdf.write_parquet(p_cache)
    return fdf, pdf
