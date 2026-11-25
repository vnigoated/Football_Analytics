"""SportsMOT football clips as an independent test of the visible-player count (journal revision, step 4).

SN-Tracking-2023's labelled clips (106) are the same clips as SoccerNet-GSR, so they add no new data. SportsMOT
(Cui et al., ICCV 2023; Hugging Face ``MCG-NJU/SportsMOT``, CC BY-NC 4.0) has 30 labelled football clips (15 train,
15 val) from other broadcasts, in MOTChallenge layout:
``<split>/<clip>/img1/000001.jpg``, ``gt/gt.txt`` and ``seqinfo.ini``.

Only players on the pitch are labelled (both teams, goalkeepers included; no referees, coaches or crowd), which is
the definition of "visible players" used by the correction.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Iterable
from pathlib import Path

import polars as pl

GT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]


def read_gt(seq_dir: str | Path) -> pl.DataFrame:
    """``gt/gt.txt`` as a table: frame, id, x, y, w, h (pixels)."""
    p = Path(seq_dir) / "gt" / "gt.txt"
    text = p.read_text().replace(" ", "").replace("\t", "")  # SportsMOT writes ", " between values
    n_cols = len(text.split("\n", 1)[0].split(","))
    df = pl.read_csv(io.StringIO(text), has_header=False, new_columns=GT_COLUMNS[:n_cols],
                     schema_overrides=dict.fromkeys(GT_COLUMNS[:n_cols], pl.Float64))
    return df.select(pl.col("frame").cast(pl.Int64), pl.col("id").cast(pl.Int64),
                     *[pl.col(c).cast(pl.Float64) for c in ("x", "y", "w", "h")])


def gt_counts(seq_dir: str | Path, frames: Iterable[int] | None = None) -> pl.DataFrame:
    """Labelled players per frame and a zoom proxy (median box height, px), in the B1 table layout:
    ``sequence, frame, gt, zoom``. ``frames`` (e.g. every image of the clip) adds frames with no labels as 0."""
    seq_dir = Path(seq_dir)
    g = read_gt(seq_dir).group_by("frame").agg(gt=pl.len().cast(pl.Int64), zoom=pl.col("h").median())
    if frames is not None:
        g = (pl.DataFrame({"frame": list(frames)}, schema={"frame": pl.Int64}).join(g, on="frame", how="left")
             .with_columns(pl.col("gt").fill_null(0)))
    return g.sort("frame").with_columns(sequence=pl.lit(seq_dir.name)).select("sequence", "frame", "gt", "zoom")


def read_split_list(path: str | Path) -> set[str]:
    return {line.strip() for line in Path(path).read_text().split() if line.strip()}


def extract_clips(tar_path: str | Path, clips: set[str], dest: str | Path) -> list[str]:
    """Stream ``tar_path`` and extract only members of the given clips into ``dest`` (paths are checked to stay
    inside ``dest``). Returns the clips found."""
    dest = Path(dest).resolve()
    found = set()
    with tarfile.open(tar_path, "r|*") as tar:
        for m in tar:
            parts = Path(m.name).parts
            clip = next((p for p in parts if p in clips), None)
            if clip is None or not (m.isfile() or m.isdir()):
                continue
            out = (dest / Path(*parts[parts.index(clip):])).resolve()
            if not out.is_relative_to(dest):
                continue
            found.add(clip)
            if m.isdir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is not None:
                out.write_bytes(src.read())
    return sorted(found)
