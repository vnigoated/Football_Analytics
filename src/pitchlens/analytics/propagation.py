"""How the video count error widens the corrected pressing estimate (Phase 2 · B2).

Paper v1 corrects a broadcast's active-presser value with ``bias = k · visible + b`` (fitted leaving the broadcast's
stadium camera and match out) and ``corrected = broadcast / (1 + bias / 100)``; with the *true* visible-player mean
the corrected error is 3.1% on average. On real video ``visible`` is measured by the pipeline with an error ``e``
(B1). Here ``e`` is drawn from the errors B1 measured on held-out SN-GSR clips and pushed through the same
correction, broadcast by broadcast:

* **correlated**: the whole broadcast has the error of one 30-s clip (worst case: every clip of a match is off the
  same way, e.g. one camera style);
* **independent**: the broadcast mean averages ``n_clips`` clip errors (a 90-min match is about 180 clips);
* **constant**: a fixed error, for the sensitivity curve.

Both pressing teams of a broadcast share the same draw of ``e`` (same video).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

ErrorModel = Callable[[np.random.Generator, int], np.ndarray]


def clip_errors(counts: pl.DataFrame, parts: tuple[str, ...] = ("held out", "test"),
                calibration: tuple[float, float] | None = None) -> np.ndarray:
    """Mean (ours − labels) per clip from B1's per-frame table (``sequence, part, gt, ours``), optionally after the
    calibration ``labels ≈ a · ours + b``."""
    d = counts.filter(pl.col("part").is_in(parts))
    ours = pl.col("ours").cast(pl.Float64)
    if calibration is not None:
        ours = ours * calibration[0] + calibration[1]
    d = d.with_columns(diff=ours - pl.col("gt").cast(pl.Float64))
    return d.group_by("sequence").agg(pl.col("diff").mean())["diff"].to_numpy()


def zero() -> ErrorModel:
    return lambda rng, n: np.zeros(n)


def constant(e: float) -> ErrorModel:
    return lambda rng, n: np.full(n, float(e))


def correlated(errors: np.ndarray) -> ErrorModel:
    """The broadcast mean is off by the error of one clip, drawn from ``errors``."""
    errors = np.asarray(errors, float)
    return lambda rng, n: rng.choice(errors, n)


def independent(errors: np.ndarray, n_clips: int = 180) -> ErrorModel:
    """The broadcast mean is off by the mean of ``n_clips`` clip errors drawn from ``errors``."""
    errors = np.asarray(errors, float)
    return lambda rng, n: rng.choice(errors, (n, n_clips)).mean(1)


def propagate(table: pl.DataFrame, model: ErrorModel, n_draws: int = 200, seed: int = 0) -> np.ndarray:
    """Corrected errors (%) of every (broadcast, pressing team) row for ``n_draws`` draws of the count error:
    array (n_draws, rows). ``table``: Phase 1 ``correction_per_broadcast.csv`` (``camera, match_id, gt, sim,
    players_visible``). The line is fitted as in paper v1, without the row's stadium camera and match."""
    rng = np.random.default_rng(seed)
    t = table.with_row_index("_row")
    out = np.full((n_draws, t.height), np.nan)
    for (cam, match), test in t.group_by("camera", "match_id", maintain_order=True):
        train = t.filter((pl.col("camera") != cam) & (pl.col("match_id") != match))
        k, b = np.polyfit(train["players_visible"].to_numpy(), train["bias"].to_numpy(), 1)
        e = model(rng, n_draws)[:, None]  # one draw per broadcast, shared by both teams
        v = test["players_visible"].to_numpy()[None, :] + e
        corrected = test["sim"].to_numpy()[None, :] / (1 + (k * v + b) / 100)
        out[:, test["_row"].to_numpy()] = (corrected / test["gt"].to_numpy()[None, :] - 1) * 100
    return out


def summarise(errors: np.ndarray) -> dict[str, float]:
    """Over all broadcasts and draws: mean error, mean absolute error, 95th percentile of |error|, shares within
    ±5% and ±10% (all in %)."""
    e = np.asarray(errors, float).ravel()
    e = e[np.isfinite(e)]
    return {
        "mean_error": round(float(e.mean()), 2),
        "mean_abs_error": round(float(np.abs(e).mean()), 2),
        "p95_abs_error": round(float(np.quantile(np.abs(e), 0.95)), 2),
        "within_5pct": round(float((np.abs(e) <= 5).mean()), 3),
        "within_10pct": round(float((np.abs(e) <= 10).mean()), 3),
    }
