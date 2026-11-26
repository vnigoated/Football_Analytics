"""How much of a real broadcast is not the live main camera? (journal revision, step 5)

SoccerNet-v2's camera-shot labels (Hugging Face ``SoccerNet/SN-Labels``, ``Labels-cameras.json``; 500 games from 6
competitions) give every camera change with the new shot's camera type and whether it is live or a replay. About 200
games are labelled throughout; the rest only around some moments (gaps of several minutes), so shares are computed on
the complete games only.

Each label describes the shot that **ends** at its timestamp: at shots on target and goals the label is a live main
camera 99% of the time read this way and 1-2% read as the shot starting there; at substitutions it is a close-up
(97%). A shot therefore lasts from the previous change to its own; the first label of each half is dropped (its
start is unknown).

Leak tolerance: if a fraction ``leak`` of the non-main frames slips through the main-camera filter (B3), the visible
count averages main frames (``v_main`` players) with leaked frames (``v_leak``). The count then shifts by
``f · (v_leak − v_main)``, with ``f`` the leaked share of the kept frames. That moves the correction by ``slope`` pp
per player.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

MAIN_LIVE = ("Main camera center", "Main camera left", "Main camera right")
GROUPS = {
    "Main camera center": "main camera", "Main camera left": "main camera", "Main camera right": "main camera",
    "Main behind the goal": "behind the goal", "Close-up behind the goal": "behind the goal",
    "Close-up player or field referee": "close-up", "Close-up side staff": "close-up", "Close-up corner": "close-up",
    "Public": "crowd", "Spider camera": "other", "Inside the goal": "other", "Goal line technology camera": "other",
    "Other": "other", "": "other",
}


def load_camera_labels(path: str | Path, root: str | Path | None = None) -> pl.DataFrame:
    """One row per shot: ``game, league, half, start_s, duration_s, label, group, replay, main_live``."""
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    rel = path.parent.relative_to(root) if root else path.parent
    game, league = rel.as_posix(), rel.parts[0] if root else ""
    rows = [{"half": int(a["gameTime"].split(" - ")[0]), "start_s": int(a["position"]) / 1000,
             "label": a.get("label", ""), "replay": a.get("replay", "")} for a in d.get("annotations", [])]
    schema = {"half": pl.Int64, "start_s": pl.Float64, "label": pl.String, "replay": pl.String}
    df = pl.DataFrame(rows, schema=schema).sort("half", "start_s")
    # the label describes the shot ENDING at its timestamp: it starts at the previous change of the half
    df = df.rename({"start_s": "end_s"}).with_columns(start_s=pl.col("end_s").shift(1).over("half")).drop_nulls(
        "start_s").with_columns(duration_s=pl.col("end_s") - pl.col("start_s"))
    return df.with_columns(
        game=pl.lit(game), league=pl.lit(league),
        group=pl.col("label").replace_strict(GROUPS, default="other"),
        main_live=pl.col("label").is_in(MAIN_LIVE) & (pl.col("replay") == "real-time"),
    ).select("game", "league", "half", "start_s", "duration_s", "label", "group", "replay", "main_live")


def load_all(root: str | Path) -> pl.DataFrame:
    root = Path(root)
    return pl.concat([load_camera_labels(p, root) for p in sorted(root.rglob("Labels-cameras.json"))])


def complete_games(shots: pl.DataFrame, min_changes: int = 300, max_gap_s: float = 180.0) -> list[str]:
    """Games labelled throughout: many changes and no shot longer than ``max_gap_s``."""
    g = shots.group_by("game").agg(n=pl.len(), longest=pl.col("duration_s").max())
    return sorted(g.filter((pl.col("n") >= min_changes) & (pl.col("longest") <= max_gap_s))["game"].to_list())


def time_shares(shots: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    """Share of broadcast time per camera group × live/replay (columns ``group, replay, share``), optionally per
    ``by`` (e.g. league)."""
    by = by or []
    t = shots.group_by(*by, "group", "replay").agg(seconds=pl.col("duration_s").sum())
    total = t.group_by(*by).agg(total=pl.col("seconds").sum()) if by else None
    t = t.join(total, on=by) if by else t.with_columns(total=pl.col("seconds").sum())
    return t.with_columns(share=pl.col("seconds") / pl.col("total")).drop("total").sort(*by, "share",
                                                                                      descending=[False] * len(by)
                                                                                      + [True])


def per_game(shots: pl.DataFrame) -> pl.DataFrame:
    """Per game: main-camera live share, replay share, close-up share and the median non-main shot length."""
    return shots.group_by("game", "league").agg(
        minutes=pl.col("duration_s").sum() / 60,
        main_live=(pl.col("duration_s") * pl.col("main_live")).sum() / pl.col("duration_s").sum(),
        replay=(pl.col("duration_s") * (pl.col("replay") == "replay")).sum() / pl.col("duration_s").sum(),
        main_replay=(pl.col("duration_s") * ((pl.col("group") == "main camera") & (pl.col("replay") == "replay")))
        .sum() / pl.col("duration_s").sum(),
        close_up=(pl.col("duration_s") * (pl.col("group") == "close-up")).sum() / pl.col("duration_s").sum(),
        nonmain_shot_s=pl.col("duration_s").filter(~pl.col("main_live")).median(),
    ).sort("game")


def frame_truth(shots: pl.DataFrame, half: int, times_s) -> pl.DataFrame:
    """Camera label of each frame time (seconds into the half) from one game's shots: ``t, label, group, replay,
    main_live``. Times outside every labelled shot get nulls."""
    s = shots.filter(pl.col("half") == half).sort("start_s")
    t = pl.DataFrame({"t": np.asarray(times_s, float)}).sort("t")
    j = t.join_asof(s.select("start_s", "duration_s", "label", "group", "replay", "main_live"),
                    left_on="t", right_on="start_s", strategy="backward")
    inside = pl.col("t") < pl.col("start_s") + pl.col("duration_s")
    return j.with_columns([pl.when(inside).then(pl.col(c)).otherwise(None).alias(c)
                           for c in ("label", "group", "replay", "main_live")]).drop("start_s", "duration_s")


def filter_report(frames: pl.DataFrame, slope: float = 4.1) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float]]:
    """How well a main-camera filter separates the live main camera from the rest of a broadcast.

    ``frames``: one row per frame with ``game, group, replay, main_live, main`` (True = kept by the filter) and
    ``people`` (detected people). Returns a table per camera group (share removed), a table per game, and a summary:

    * ``main_live_removed``: live main-camera frames wrongly removed;
    * ``nonmain_kept``: frames that are not the main camera (live or replay) but were kept: the leak;
    * ``count_shift``: mean people on kept frames minus on live main-camera frames, and ``shift_pp`` = slope x it,
      the move of the corrected pressing value.
    """
    f = frames.drop_nulls(["group", "main"]).with_columns(
        cat=pl.when(pl.col("main_live")).then(pl.lit("main camera, live"))
        .when(pl.col("group") == "main camera").then(pl.lit("main camera, replay"))
        .otherwise(pl.col("group")))
    by_group = f.group_by("cat").agg(frames=pl.len(), removed=(~pl.col("main")).mean()).with_columns(
        share=pl.col("frames") / pl.col("frames").sum()).sort("frames", descending=True)

    def game_stats(g: pl.DataFrame) -> dict[str, float]:
        live = g.filter(pl.col("main_live"))
        nonmain = g.filter(pl.col("group") != "main camera")
        kept = g.filter(pl.col("main"))
        shift = float(kept["people"].mean() - live["people"].mean()) if kept.height and live.height else float("nan")
        return {"frames": float(g.height),
                "main_live_removed": float((~live["main"]).mean()) if live.height else float("nan"),
                "nonmain_kept": float(nonmain["main"].mean()) if nonmain.height else float("nan"),
                "nonmain_kept_share_of_kept": float((kept["group"] != "main camera").mean()) if kept.height else 0.0,
                "count_shift": shift, "shift_pp": slope * shift}

    per_game = pl.DataFrame([{"game": k[0], **game_stats(g)} for k, g in f.group_by("game")]).sort("game")
    return by_group, per_game, game_stats(f)


def tune_filter_logo(frames: pl.DataFrame, settings: list[dict], apply, max_live_removed: float = 0.05,
                     slope: float = 4.1) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Choose filter settings leaving one game out (journal step 6).

    ``apply(frames, **setting)`` must return ``frames`` with a ``main`` column. For each held-out game, the setting
    with the smallest mean |shift_pp| over the other games is chosen, among those that remove at most
    ``max_live_removed`` of their live main-camera frames, and scored on the held-out game. Returns the grid
    (all games) and the held-out table.
    """
    res = []
    for kw in settings:
        _, pg, s = filter_report(apply(frames, **kw), slope)
        res.append((kw, pg, s))
    keys = ("main_live_removed", "nonmain_kept", "count_shift", "shift_pp")
    grid = pl.DataFrame([{**kw, **{k: s[k] for k in keys}} for kw, _, s in res])
    rows = []
    for test in sorted(frames["game"].unique()):
        best, score = None, float("inf")
        for kw, pg, _ in res:
            tr = pg.filter(pl.col("game") != test)
            if tr["main_live_removed"].mean() > max_live_removed:
                continue
            sc = tr["shift_pp"].abs().mean()
            if sc < score:
                best, score = (kw, pg), sc
        if best is None:
            raise ValueError("no setting keeps enough live main-camera frames")
        te = best[1].filter(pl.col("game") == test).row(0, named=True)
        rows.append({"game": test, **best[0], **{k: te[k] for k in keys}})
    return grid, pl.DataFrame(rows)


def leak_shift(nonmain_share: float, leak: float, v_main: float, v_leak: float, slope: float = 4.1) -> float:
    """Shift of the corrected pressing value (pp) when a fraction ``leak`` of the non-main frames is kept."""
    kept_leak = leak * nonmain_share
    f = kept_leak / ((1 - nonmain_share) + kept_leak) if kept_leak > 0 else 0.0
    return slope * f * (v_leak - v_main)


def max_leak(nonmain_share: float, v_main: float, v_leak: float, max_pp: float = 1.0, slope: float = 4.1) -> float:
    """Largest leak fraction (0-1) that keeps ``|leak_shift|`` within ``max_pp``."""
    if abs(v_leak - v_main) < 1e-9:
        return 1.0
    leaks = np.linspace(0, 1, 10001)
    ok = [lk for lk in leaks if abs(leak_shift(nonmain_share, lk, v_main, v_leak, slope)) <= max_pp]
    return float(max(ok))
