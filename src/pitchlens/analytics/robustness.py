"""Is the pressing bias driven by one team? (journal revision, step 1)

The IDSSE ground truth has 7 matches, and one team (Fortuna Düsseldorf) plays in 5 of them. These checks use paper
v1's per-broadcast table (``correction_per_broadcast.csv``: one row per stadium camera × match × pressing team with
the relative bias of active pressers and the number of visible players):

* per-team bias;
* a **team-balanced** mean (every team weighs the same), with an interval from resampling teams;
* **leave one team out**: the mean bias and the bias-per-visible-player slope without every match of that team;
* a **team effect at equal visibility**: is the team's bias different once the number of visible players is
  accounted for (match-level bootstrap)?
* the correction evaluated with the team **unseen**: the line is fitted without the broadcast's camera and without
  every match of its pressing team.
"""

from __future__ import annotations

import numpy as np
import polars as pl

DATASET = pl.when(pl.col("match_id").str.starts_with("metrica")).then(pl.lit("Metrica")).otherwise(pl.lit("IDSSE"))


def with_dataset(t: pl.DataFrame) -> pl.DataFrame:
    return t.with_columns(dataset=DATASET) if "dataset" not in t.columns else t


def team_matches(t: pl.DataFrame) -> dict[str, set[str]]:
    """Team -> the matches it plays in (as pressing team or opponent)."""
    out: dict[str, set[str]] = {}
    for r in t.select("pressing_team_name", "match_id").unique().iter_rows(named=True):
        out.setdefault(r["pressing_team_name"], set()).add(r["match_id"])
    return out


def per_team(t: pl.DataFrame) -> pl.DataFrame:
    """Mean, min and max bias and mean visible players per pressing team."""
    return (with_dataset(t).group_by("dataset", "pressing_team_name").agg(
        matches=pl.col("match_id").n_unique(), rows=pl.len(), bias=pl.col("bias").mean().round(2),
        bias_min=pl.col("bias").min().round(1), bias_max=pl.col("bias").max().round(1),
        visible=pl.col("players_visible").mean().round(2)).sort("dataset", "bias"))


def balanced_mean(t: pl.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of the per-team mean biases (every team weighs the same); 95% interval from resampling teams."""
    team_means = t.group_by("pressing_team_name").agg(pl.col("bias").mean())["bias"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(team_means, (n_boot, len(team_means))).mean(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(team_means.mean()), float(lo), float(hi)


def slope(t: pl.DataFrame) -> float:
    """Percentage points of bias per visible player (least squares over rows)."""
    return float(np.polyfit(t["players_visible"].to_numpy(), t["bias"].to_numpy(), 1)[0])


def leave_one_team_out(t: pl.DataFrame) -> pl.DataFrame:
    """For every team: mean bias and slope over the matches that do not involve it."""
    tm = team_matches(t)
    rows = [{"left_out": "none", "matches": t["match_id"].n_unique(), "rows": t.height,
             "bias": round(float(t["bias"].mean()), 2), "slope": round(slope(t), 2)}]
    for team, matches in sorted(tm.items()):
        rest = t.filter(~pl.col("match_id").is_in(list(matches)))
        if rest["match_id"].n_unique() < 2:
            continue
        rows.append({"left_out": team, "matches": rest["match_id"].n_unique(), "rows": rest.height,
                     "bias": round(float(rest["bias"].mean()), 2), "slope": round(slope(rest), 2)})
    return pl.DataFrame(rows)


def team_effect(t: pl.DataFrame, team: str, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Least squares ``bias = a + b · visible + c · [pressing team == team]``: c is the team's extra bias at equal
    visibility (percentage points), with a 95% interval from resampling matches."""
    def fit(d: pl.DataFrame) -> tuple[float, float]:
        x = np.column_stack([np.ones(d.height), d["players_visible"].to_numpy(),
                             (d["pressing_team_name"] == team).to_numpy().astype(float)])
        coef, *_ = np.linalg.lstsq(x, d["bias"].to_numpy(), rcond=None)
        return float(coef[2]), float(coef[1])

    c, b = fit(t)
    parts = {k[0]: g for k, g in t.group_by("match_id")}
    names = list(parts)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        d = pl.concat([parts[names[i]] for i in rng.integers(len(names), size=len(names))])
        if (d["pressing_team_name"] == team).any() and (d["pressing_team_name"] != team).any():
            boots.append(fit(d)[0])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"team": team, "extra_bias_pp": round(c, 2), "ci_low": round(float(lo), 2), "ci_high": round(float(hi), 2),
            "slope_pp_per_player": round(b, 2)}


def correction_team_unseen(t: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    """Paper v1's correction, but the line for each row is fitted without its stadium camera and without every match
    of its pressing team (so the team itself is never seen)."""
    tm = team_matches(t)
    out = []
    for r in t.iter_rows(named=True):
        train = t.filter((pl.col("camera") != r["camera"])
                         & ~pl.col("match_id").is_in(list(tm[r["pressing_team_name"]])))
        k, b = np.polyfit(train["players_visible"].to_numpy(), train["bias"].to_numpy(), 1)
        pred = k * r["players_visible"] + b
        out.append({**r, "pred_team_unseen": pred,
                    "error_team_unseen": (r["sim"] / (1 + pred / 100) / r["gt"] - 1) * 100})
    res = pl.DataFrame(out)
    e = res["error_team_unseen"].to_numpy()
    return res, {"rows": float(len(e)), "mean_error": float(e.mean()), "mean_abs_error": float(np.abs(e).mean()),
                 "p95_abs_error": float(np.quantile(np.abs(e), 0.95))}


def transfer_correction(train: pl.DataFrame, test: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    """Correction line fitted on ``train`` (e.g. the A-League stadium cameras) and applied to ``test`` (e.g. cameras
    fitted to another league); for each test row the line leaves out the row's match, so the match is unseen too.
    Returns per-row errors and a summary (raw and corrected errors, %)."""
    out = []
    for (match,), g in test.group_by("match_id"):
        tr = train.filter(pl.col("match_id") != match)
        k, b = np.polyfit(tr["players_visible"].to_numpy(), tr["bias"].to_numpy(), 1)
        pred = k * g["players_visible"].to_numpy() + b
        cor = g["sim"].to_numpy() / (1 + pred / 100)
        out.append(g.with_columns(pl.Series("predicted_bias", pred),
                                  pl.Series("error_transfer", (cor / g["gt"].to_numpy() - 1) * 100)))
    res = pl.concat(out)
    raw, cor = res["bias"].to_numpy(), res["error_transfer"].to_numpy()
    ks, bs = np.polyfit(res["players_visible"].to_numpy(), raw, 1)
    kt, bt = np.polyfit(train["players_visible"].to_numpy(), train["bias"].to_numpy(), 1)
    return res, {
        "rows": float(res.height),
        "raw_mean_error": float(raw.mean()), "raw_mean_abs_error": float(np.abs(raw).mean()),
        "corrected_mean_error": float(cor.mean()), "corrected_mean_abs_error": float(np.abs(cor).mean()),
        "corrected_p95_abs_error": float(np.quantile(np.abs(cor), 0.95)),
        "slope_train": float(kt), "intercept_train": float(bt),
        "slope_test": float(ks), "intercept_test": float(bs),
    }
