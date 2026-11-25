"""Command-line interface: ``pitchlens <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _cmd_download_gsr(args: argparse.Namespace) -> None:
    from pitchlens.data.datasets import download_sn_gsr

    path = download_sn_gsr(args.dest, tuple(args.splits), args.revision)
    print(f"Downloaded {args.splits} to {path}")


def _cmd_pressing_baseline(args: argparse.Namespace) -> None:
    from pitchlens.analytics.pressing import compute_pressing_intensity, summarise_frames
    from pitchlens.data.datasets import load_tracking

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading match {args.match} (limit={args.limit}) ...")
    ds = load_tracking(args.match, limit=args.limit)
    print("Computing pressing intensity ...")
    pti, tracking = compute_pressing_intensity(ds, speed_threshold=args.speed_threshold)
    team_names = {t.team_id: t.name for t in ds.metadata.teams}
    summary = summarise_frames(pti, tracking).with_columns(
        pl.lit(args.match).alias("match_id"),
        pl.col("pressing_team_id").replace_strict(team_names, default=None).alias(
            "pressing_team_name"
        ),
    )
    summary_path = out / f"pressing_{args.match}.parquet"
    summary.write_parquet(summary_path)
    print(f"Saved {len(summary)} frames -> {summary_path}")

    if not args.no_plot:
        from pitchlens.viz import plot_pressing_timeline

        fig_path = out / f"pressing_{args.match}.png"
        plot_pressing_timeline(summary, team_names, fig_path)
        print(f"Saved plot -> {fig_path}")


def _cmd_pressing_summary(args: argparse.Namespace) -> None:
    from pitchlens.analytics.pressing import summarise_matches

    files = sorted(Path(args.dir).glob("pressing_*.parquet"))
    if not files:
        raise SystemExit(f"No pressing_*.parquet files in {args.dir}; run pressing-baseline first")
    table = summarise_matches(pl.concat([pl.read_parquet(f) for f in files], how="diagonal"))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True):
        print(table)
    out = Path(args.dir) / "summary.csv"
    table.write_csv(out)
    print(f"Saved -> {out}")


def _cmd_visibility_bias(args: argparse.Namespace) -> None:
    """Experiment 1: pressing on full vs. virtual-broadcast tracking."""
    from pitchlens.analytics.compare import join_frames
    from pitchlens.analytics.pressing import prepare_tracking, run_pressing, summarise_frames
    from pitchlens.data.datasets import load_tracking
    from pitchlens.simulation.camera import BroadcastCamera, WedgeCamera
    from pitchlens.simulation.occlusion import apply_broadcast_view, visibility_stats

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt_path = Path(args.gt_dir) / f"pressing_{args.match}.parquet"
    if not gt_path.exists():
        raise SystemExit(
            f"Need full-match ground truth at {gt_path}: "
            f"run `pitchlens pressing-baseline --match {args.match}` first"
        )
    gt = pl.read_parquet(gt_path)
    ds = load_tracking(args.match)
    dims = ds.metadata.pitch_dimensions
    team_names = {t.team_id: t.name for t in ds.metadata.teams}

    fps = ds.metadata.frame_rate or 25.0
    runs = []  # (view_length, camera, tag)
    if args.camera_json:
        import json

        paths = []
        for c in args.camera_json:
            c = Path(c)
            paths += sorted(c.glob("*.json")) if c.is_dir() else [c]
        for path in paths:
            params = json.loads(path.read_text())
            params |= {"fps": fps, "pitch_length": dims.pitch_length, "pitch_width": dims.pitch_width}
            base = args.tag if len(paths) == 1 else f"{args.tag}_{path.stem}"
            for seed in args.seeds or [None]:
                tag = base if seed is None else f"{base}_s{seed}"
                runs.append((0.0, WedgeCamera(**(params | {"seed": seed})), tag))
    else:
        for view in args.view_lengths:
            cam = BroadcastCamera(
                view_length=min(view, dims.pitch_length),
                pitch_length=dims.pitch_length,
                pitch_width=dims.pitch_width,
                lag_s=args.lag_s,
                perspective=args.perspective,
                zoom_gain=args.zoom_gain,
                fps=fps,
            )
            runs.append((view, cam, args.tag))

    prepared = prepare_tracking(ds)  # once per match, reused for every camera
    for view, cam, tag in runs:
        stats: dict[str, float] = {}

        def mask(df: pl.DataFrame, cam=cam, stats=stats) -> pl.DataFrame:
            masked = apply_broadcast_view(df, cam)  # camera path needs every frame
            if args.sample_every > 1:
                keep = pl.col("frame_id") % args.sample_every == 0
                df, masked = df.filter(keep), masked.filter(keep)
            stats.update(visibility_stats(df, masked))
            return masked

        print(f"[{args.match}] camera={tag} view_length={view} m ...")
        pti, tracking = run_pressing(
            prepared, speed_threshold=args.speed_threshold, transform=mask
        )
        sim = summarise_frames(pti, tracking)
        gt_part = gt.join(sim.select("period_id", "frame_id"), on=["period_id", "frame_id"], how="semi")
        joined = join_frames(gt_part, sim).with_columns(
            pl.lit(float(view)).alias("view_length"),
            pl.lit(args.lag_s).alias("lag_s"),
            pl.lit(args.perspective).alias("perspective"),
            pl.lit(args.zoom_gain).alias("zoom_gain"),
            pl.lit(tag).alias("camera"),
            pl.lit(stats["players_per_frame_visible"]).alias("players_visible"),
        )
        if "pressing_team_name" not in joined.columns:
            joined = joined.with_columns(
                pl.col("pressing_team_id").replace_strict(team_names, default=None)
                .alias("pressing_team_name")
            )
        path = out / f"vis_{args.match}_v{view:g}_{tag}.parquet"
        joined.write_parquet(path)
        print(f"  visible players/frame = {stats['players_per_frame_visible']:.2f} -> {path}")


def _cmd_visibility_summary(args: argparse.Namespace) -> None:
    from pitchlens.analytics.compare import bias_table

    joined = _load_visibility(args.dir)
    per_team = bias_table(joined)
    per_team.write_csv(Path(args.dir) / "bias_per_team.csv")

    overall = (
        per_team.group_by("camera", "view_length")
        .agg(
            pl.len().alias("team_matches"),
            pl.col("rel_bias_pressure_pct").mean().alias("bias_pressure_pct"),
            pl.col("rel_bias_pressure_pct").std().alias("bias_pressure_sd"),
            pl.col("frame_mae").mean(),
            pl.col("frame_corr").mean(),
            pl.col("rel_bias_active_pct").mean().alias("bias_active_pct"),
            pl.col("rel_bias_total_pct").mean().alias("bias_total_pct"),
        )
        .join(
            joined.group_by("camera", "view_length").agg(pl.col("players_visible").mean()),
            on=["camera", "view_length"],
        )
        .sort("camera", "view_length")
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3):
        print(overall)
    overall.write_csv(Path(args.dir) / "bias_overall.csv")
    print(f"Saved -> {Path(args.dir) / 'bias_overall.csv'} and bias_per_team.csv")


def _load_visibility(dir_: str) -> pl.DataFrame:
    files = sorted(Path(dir_).glob("vis_*.parquet"))
    if not files:
        raise SystemExit(f"No vis_*.parquet in {dir_}; run visibility-bias first")
    joined = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
    # Experiment 1 files predate the camera column: they used the simple camera
    if "camera" not in joined.columns:
        joined = joined.with_columns(pl.lit("simple").alias("camera"))
    return joined.with_columns(pl.col("camera").fill_null("simple"))


def _cmd_visibility_ci(args: argparse.Namespace) -> None:
    """RQ2: block-bootstrap confidence intervals for the visibility bias."""
    from pitchlens.analytics.uncertainty import bootstrap_bias

    joined = _load_visibility(args.dir)
    per_team, overall = bootstrap_bias(
        joined, block_s=args.block_s, n_boot=args.n_boot, level=args.level, seed=args.seed
    )
    out = Path(args.dir)
    per_team.write_csv(out / "ci_per_team.csv")
    overall = overall.sort("camera", "view_length")
    overall.write_csv(out / "ci_overall.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=2):
        print(overall)
    print(f"Saved -> {out / 'ci_overall.csv'} and ci_per_team.csv")


def _fit_per_match(ids: list[int], args: argparse.Namespace) -> None:
    """One WedgeCamera per SkillCorner match: the spread across stadiums (RQ2 part 2)."""
    import dataclasses
    import json

    from pitchlens.data.skillcorner import load_skillcorner
    from pitchlens.simulation.camera_fit import evaluate_camera, fit_wedge_camera

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in ids:
        frames, players = load_skillcorner(m, args.data)
        cam, report = fit_wedge_camera([frames], [players])
        cam.fps = 10.0
        ev = evaluate_camera(cam, frames, players)
        params = dataclasses.asdict(cam) | {"fps": 25.0, "seed": None}
        (out / f"{m}.json").write_text(json.dumps(params, indent=2))
        rows.append(
            {
                "match_id": m,
                "cam_x": cam.cam_x,
                "cam_y": cam.cam_y,
                "fov_median_deg": report["fov_deg_p10_p50_p90"][1],
                "lag_s": cam.lag_s,
                "r2_pan": report["r2_pan"],
                **ev,
            }
        )
        print(f"  {m}: camera y={cam.cam_y:.1f} m, visible {ev['visible_per_frame_pred']:.2f} "
              f"(real {ev['visible_per_frame_true']:.2f}), agreement {ev['accuracy']:.3f}")
    table = pl.DataFrame(rows)
    table.write_csv(out / "stadium_cameras.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=2):
        print(table)
    print(f"Saved {len(rows)} cameras -> {out}")


def _cmd_fit_camera(args: argparse.Namespace) -> None:
    """Fit the WedgeCamera on SkillCorner broadcast footprints; validate on held-out matches."""
    import dataclasses
    import json

    from pitchlens.data.skillcorner import load_skillcorner, skillcorner_match_ids
    from pitchlens.simulation.camera import BroadcastCamera
    from pitchlens.simulation.camera_fit import (
        evaluate_camera,
        fit_wedge_camera,
        footprint_visibility,
        skillcorner_long,
    )

    ids = sorted(skillcorner_match_ids(args.data))[: args.max_matches]
    if args.per_match:
        _fit_per_match(ids, args)
        return
    fit_ids, test_ids = ids[0::2], ids[1::2]
    print(f"Fitting on {len(fit_ids)} matches, testing on {len(test_ids)}")
    fit_data = [load_skillcorner(m, args.data) for m in fit_ids]
    ball_cam, _ = fit_wedge_camera([f for f, _ in fit_data])
    camera, report = fit_wedge_camera([f for f, _ in fit_data], [p for _, p in fit_data])
    del fit_data
    ball_cam.fps = camera.fps = 10.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = dataclasses.asdict(camera) | {"fps": 25.0, "seed": None}
    (out / "wedge_camera.json").write_text(json.dumps(params, indent=2))
    (out / "fit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(params, indent=2))
    print(json.dumps(report, indent=2))

    cameras = {
        "wedge_ball_only": ball_cam,
        "wedge_fit": camera,
        "wedge_fit_noise": dataclasses.replace(camera, seed=0),
        "rect_simple_40m": BroadcastCamera(view_length=40, fps=10),
        "rect_realistic_32m": BroadcastCamera(
            view_length=32, lag_s=0.5, perspective=0.3, zoom_gain=0.5, fps=10
        ),
    }
    rows = []
    for m in test_ids:
        frames, players = load_skillcorner(m, args.data)
        long = skillcorner_long(frames, players)
        inside = footprint_visibility(frames, long)
        det = long.filter(pl.col("id") != "ball")["is_detected"].to_numpy()
        rows.append({"match_id": m, "camera": "real_footprint (upper bound)",
                     "accuracy": float((inside == det).mean())})  # fmt: skip
        for name, cam in cameras.items():
            rows.append({"match_id": m, "camera": name, **evaluate_camera(cam, frames, players)})
        print(f"  evaluated {m}")
    table = pl.DataFrame(rows, infer_schema_length=None)
    table.write_csv(out / "validation_per_match.csv")
    summary = table.group_by("camera").agg(pl.exclude("match_id").mean()).sort("camera")
    summary.write_csv(out / "validation.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print(summary)
    print(f"Saved -> {out}")


def _cmd_stadium_summary(args: argparse.Namespace) -> None:
    """RQ2 part 2: bias across stadium cameras, with intervals per source of uncertainty."""
    import numpy as np

    from pitchlens.analytics.uncertainty import stadium_bootstrap

    joined = _load_visibility(args.dir)
    per_cam, intervals = stadium_bootstrap(joined, n_boot=args.n_boot, seed=args.seed)
    slope, _ = np.polyfit(per_cam["players_visible"], per_cam["active_bias"], 1)
    out = Path(args.dir)
    per_cam.write_csv(out / "stadium_per_camera.csv")
    intervals.write_csv(out / "stadium_intervals.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=2):
        print(per_cam)
        print(intervals)
    print(f"Active-presser bias changes by {slope:+.2f} percentage points per extra visible player "
          f"(across {per_cam.height} stadium cameras)")
    print(f"Saved -> {out / 'stadium_intervals.csv'} and stadium_per_camera.csv")


def _cmd_bias_correction(args: argparse.Namespace) -> None:
    """RQ3: correct active pressers from the number of visible players (strict evaluation)."""
    import json

    from pitchlens.analytics.correction import broadcast_table, evaluate_correction

    table = pl.concat(
        [broadcast_table(_load_visibility(d)) for d in args.dirs], how="vertical_relaxed"
    )
    res, summary = evaluate_correction(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.write_csv(out / "correction_per_broadcast.csv")
    (out / "correction_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {out}")


def _cmd_count_error(args: argparse.Namespace) -> None:
    """B2: carry the pipeline's visible-player count error (B1) into the pressing correction (paper v1)."""
    import json

    import numpy as np

    from pitchlens.analytics.propagation import (
        clip_errors,
        constant,
        correlated,
        independent,
        propagate,
        summarise,
        zero,
    )
    from pitchlens.analytics.visible_count import fit_calibration

    counts = pl.read_parquet(args.counts)
    table = pl.read_csv(args.table).select("camera", "match_id", "pressing_team_name", "gt", "sim",
                                           "players_visible", "bias")
    cal = fit_calibration(counts.filter(pl.col("part") == "tune"))
    raw_e, cal_e = clip_errors(counts), clip_errors(counts, calibration=cal)
    n = args.draws
    errs = {
        "no correction": table["bias"].to_numpy()[None, :],
        "corrected, true count (paper v1)": propagate(table, zero(), 1),
        "corrected, video count raw, whole match": propagate(table, independent(raw_e, args.clips), n, args.seed),
        "corrected, video count raw, one clip": propagate(table, correlated(raw_e), n, args.seed),
        "corrected, video count calibrated, whole match": propagate(table, independent(cal_e, args.clips), n,
                                                                    args.seed),
        "corrected, video count calibrated, one clip": propagate(table, correlated(cal_e), n, args.seed),
    }
    summary = pl.DataFrame([{"scenario": k, **summarise(v)} for k, v in errs.items()])
    sens = pl.DataFrame([{"count_error": float(e), **summarise(propagate(table, constant(e), 1))}
                         for e in np.arange(-2.0, 2.01, 0.5)])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.write_csv(out / "b2_scenarios.csv")
    sens.write_csv(out / "b2_sensitivity.csv")
    (out / "b2_settings.json").write_text(json.dumps({
        "calibration_a_b": cal, "broadcasts_rows": table.height, "draws": n, "clips_per_match": args.clips,
        "seed": args.seed, "clips": len(raw_e),
        "clip_error_raw": {"mean": float(raw_e.mean()), "sd": float(raw_e.std(ddof=1))},
        "clip_error_calibrated": {"mean": float(cal_e.mean()), "sd": float(cal_e.std(ddof=1))}}, indent=2))
    print(f"{len(raw_e)} held-out / test clips; count error per clip raw {raw_e.mean():+.2f} ± {raw_e.std(ddof=1):.2f},"
          f" calibrated {cal_e.mean():+.2f} ± {cal_e.std(ddof=1):.2f} (calibration a={cal[0]:.3f}, b={cal[1]:.2f})")
    with pl.Config(tbl_rows=20, tbl_cols=10, fmt_str_lengths=60, tbl_width_chars=200):
        print(summary)
        print(sens)
    if not args.no_plot:
        _plot_b2(errs, out)
    print(f"Saved -> {out}")

def _plot_b2(errs: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    show = [("no correction", "#9aa0a6"), ("corrected, true count (paper v1)", "#2a6fdb"),
            ("corrected, video count calibrated, whole match", "#1e8e3e"),
            ("corrected, video count calibrated, one clip", "#e07b1a")]
    labels = ["no correction", "corrected,\ntrue count", "corrected, video count\n(whole match)",
              "corrected, video count\n(single clip)"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    data = [np.asarray(errs[k]).ravel() for k, _ in show]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, (_, c) in zip(parts["bodies"], show, strict=True):
        body.set_facecolor(c)
        body.set_alpha(0.6)
    for i, d in enumerate(data, 1):
        lo, med, hi = np.quantile(d, [0.025, 0.5, 0.975])
        ax.plot([i, i], [lo, hi], c="#333", lw=1.2)
        ax.scatter([i], [med], c="#333", s=14, zorder=3)
    ax.axhline(0, c="#888", lw=1, ls="--")
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("error in active pressers (%)")
    ax.set_title("Pressing correction with the visible-player count from real video")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "b2_propagation.png", dpi=150)
    fig.savefig(out / "b2_propagation.pdf")
    plt.close(fig)


def _cmd_team_robustness(args: argparse.Namespace) -> None:
    """Journal step 1: is the pressing bias driven by one team (Fortuna Duesseldorf in 5 of 7 IDSSE matches)?"""
    import json

    from pitchlens.analytics.robustness import (
        balanced_mean,
        correction_team_unseen,
        leave_one_team_out,
        per_team,
        team_effect,
        with_dataset,
    )

    t = with_dataset(pl.read_csv(args.table))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    with pl.Config(tbl_rows=30, tbl_cols=12, tbl_width_chars=200, fmt_str_lengths=30):
        print(per_team(t))
        for name in ["IDSSE", "Metrica"]:
            d = t.filter(pl.col("dataset") == name)
            m, lo, hi = balanced_mean(d, n_boot=args.n_boot, seed=args.seed)
            summary[name] = {"rows": d.height, "mean_bias": round(float(d["bias"].mean()), 2),
                             "team_balanced_bias": round(m, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2)}
        print(json.dumps(summary, indent=1))
        idsse = t.filter(pl.col("dataset") == "IDSSE")
        loo = leave_one_team_out(idsse)
        loo.write_csv(out / "leave_one_team_out_idsse.csv")
        print(loo)
        eff = team_effect(idsse, args.team, n_boot=args.n_boot, seed=args.seed)
        print("team effect at equal visibility:", eff)
        res, cor = correction_team_unseen(t)
        res.write_csv(out / "correction_team_unseen.csv")
        print("correction with the pressing team unseen:", {k: round(v, 2) for k, v in cor.items()})
    summary.update({"team_effect": eff, "correction_team_unseen": cor})
    (out / "team_robustness.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved -> {out}")


def _cmd_fit_camera_gsr(args: argparse.Namespace) -> None:
    """Journal step 3: fit one WedgeCamera per SoccerNet-GSR game (Swiss Super League) from real footprints."""
    import dataclasses
    import json
    import zipfile
    from collections import defaultdict

    import numpy as np

    from pitchlens.data.gsr import parse_labels
    from pitchlens.data.gsr_camera import clip_footprints, orient, visible_players
    from pitchlens.simulation.camera_fit import fit_wedge_camera

    games: dict[str, list] = defaultdict(list)
    for zp in args.zips:
        z = zipfile.ZipFile(zp)
        split = Path(zp).stem
        for n in z.namelist():
            if n.endswith("Labels-GameState.json"):
                d = json.loads(z.read(n))
                games[f"{split}_{d['info']['game_id']}"].append(parse_labels(d, n.split("/")[0]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for g, clips in sorted(games.items()):
        frames = [orient(clip_footprints(im, ob)).with_columns(period=pl.lit(i + 1)) for i, (im, ob) in enumerate(clips)]
        frames = [f for f in frames if f.height >= 50]
        cam, rep = fit_wedge_camera(frames, None, fps=25.0)
        (out / f"gsr_{g}.json").write_text(json.dumps(dataclasses.asdict(cam) | {"fps": 25.0, "seed": None}, indent=2))
        vis = float(np.mean([visible_players(im, ob) for im, ob in clips]))
        rows.append({"game": g, "clips": len(frames), "frames": rep["frames"], "cam_x": round(cam.cam_x, 1),
                     "cam_y": round(cam.cam_y, 1), "fov_median_deg": rep["fov_deg_p10_p50_p90"][1],
                     "fov_p10_p90": rep["fov_deg_p10_p50_p90"][::2], "lag_s": cam.lag_s, "pan_gain": round(cam.pan_gain, 2),
                     "r2_pan": round(rep["r2_pan"], 3), "r2_fov": round(rep["r2_fov"], 3), "r2_near": round(rep["r2_near"], 3),
                     "real_visible": round(vis, 2)})
    table = pl.DataFrame(rows)
    table.drop("fov_p10_p90").write_csv(out / "gsr_cameras.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220):
        print(table)
    print(f"Saved {len(rows)} cameras -> {out}")


def _cmd_cross_league(args: argparse.Namespace) -> None:
    """Journal step 3: correction fitted on the A-League stadium cameras, applied to the Swiss (SN-GSR) cameras."""
    import json

    from pitchlens.analytics.correction import broadcast_table
    from pitchlens.analytics.robustness import transfer_correction

    train = pl.read_csv(args.train)
    test = pl.concat([broadcast_table(_load_visibility(d)) for d in args.dirs], how="vertical_relaxed")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res, summary = transfer_correction(train, test)
    res.write_csv(out / "cross_league_per_broadcast.csv")
    per_cam = res.group_by("camera").agg(pl.col("players_visible").mean().round(2), pl.col("bias").mean().round(2),
                                         pl.col("error_transfer").abs().mean().round(2).alias("corrected_mae")).sort("players_visible")
    per_cam.write_csv(out / "cross_league_per_camera.csv")
    with pl.Config(tbl_rows=-1):
        print(per_cam)
    (out / "cross_league_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _cmd_replay_share(args: argparse.Namespace) -> None:
    """Journal step 5: share of broadcast time that is not the live main camera, and the leak tolerance."""
    import json

    import numpy as np

    from pitchlens.analytics.broadcast_shots import (
        complete_games,
        leak_shift,
        load_all,
        max_leak,
        per_game,
        time_shares,
    )

    shots_all = load_all(args.labels)
    games = complete_games(shots_all)
    shots = shots_all.filter(pl.col("game").is_in(games))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{shots_all['game'].n_unique()} games with camera labels, {len(games)} labelled throughout")
    shares = time_shares(shots)
    shares.write_csv(out / "time_shares.csv")
    league = time_shares(shots, by=["league"])
    league.write_csv(out / "time_shares_by_league.csv")
    pg = per_game(shots)
    pg.write_csv(out / "per_game.csv")
    nonmain = 1 - pg["main_live"].to_numpy()
    q = {k: float(np.quantile(nonmain, p)) for k, p in [("median", 0.5), ("p90", 0.9)]}
    rows = []
    for what, v_leak in [("close-up (0 players)", 0.0), ("close-up (3 players)", 3.0),
                         ("wide non-main (8 players)", 8.0), ("main-camera replay (= live)", args.v_main)]:
        for game, s in q.items():
            rows.append({"leaked frames": what, "game": game, "non-main share": round(s, 3),
                         "shift if all leak (pp)": round(leak_shift(s, 1.0, args.v_main, v_leak), 2),
                         "shift at 10% leak (pp)": round(leak_shift(s, 0.1, args.v_main, v_leak), 2),
                         f"max leak for {args.max_pp} pp": round(max_leak(s, args.v_main, v_leak, args.max_pp), 3)})
    tol = pl.DataFrame(rows)
    tol.write_csv(out / "leak_tolerance.csv")
    summary = {
        "games_with_labels": shots_all["game"].n_unique(), "games_complete": len(games),
        "per_league_games": dict(pg.group_by("league").len().iter_rows()),
        **{f"{c}_{s}": float(getattr(pg[c], s)()) for c in ("main_live", "replay", "main_replay", "close_up")
           for s in ("mean", "median", "min", "max")},
        "nonmain_shot_s_median": float(pg["nonmain_shot_s"].median()),
        "v_main": args.v_main, "max_pp": args.max_pp,
    }
    (out / "replay_share_summary.json").write_text(json.dumps(summary, indent=2))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, fmt_str_lengths=40):
        print(shares.with_columns(pl.col("share").round(4), (pl.col("seconds") / 3600).round(1).alias("hours"))
              .drop("seconds"))
        print(pg.group_by("league").agg(games=pl.len(), main_live=pl.col("main_live").mean().round(3),
                                        replay=pl.col("replay").mean().round(3),
                                        main_replay=pl.col("main_replay").mean().round(3),
                                        close_up=pl.col("close_up").mean().round(3)).sort("league"))
        print(tol)
    print(json.dumps(summary, indent=2))


def _cmd_video(args: argparse.Namespace) -> None:
    """Broadcast clip -> per-frame pitch positions, roles and teams (CSV)."""
    import tempfile
    from pathlib import Path

    from rfdetr import RFDETRMedium

    from pitchlens.vision.keypoints import load_model
    from pitchlens.vision.pipeline import run_video, video_to_frames

    src = Path(args.input)
    with tempfile.TemporaryDirectory() as tmp:
        clip = src if src.is_dir() else video_to_frames(src, tmp)
        det = RFDETRMedium(pretrain_weights=args.detector, resolution=args.resolution, num_classes=2)
        kp = load_model(args.keypoints, args.device)
        gs = run_video(clip, det, kp, device=args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    gs.write_csv(args.out)
    print(f"{gs.height} rows, {gs['id'].n_unique()} tracks -> {args.out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pitchlens", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download-gsr", help="Download SoccerNet GSR zips from Hugging Face")
    d.add_argument("--dest", default="data/SN-GSR-2025")
    d.add_argument("--splits", nargs="+", default=["train", "valid"])
    d.add_argument("--revision", default="main", help="Pin a commit hash for reproducibility")
    d.set_defaults(func=_cmd_download_gsr)

    b = sub.add_parser("pressing-baseline", help="Pressing intensity on full IDSSE tracking")
    b.add_argument("--match", default="J03WPY")
    b.add_argument("--limit", type=int, default=None, help="Only the first N frames (quick test)")
    b.add_argument("--speed-threshold", type=float, default=2.0)
    b.add_argument("--out", default="outputs/pressing_baseline")
    b.add_argument("--no-plot", action="store_true")
    b.set_defaults(func=_cmd_pressing_baseline)

    s = sub.add_parser("pressing-summary", help="Per-match/team table from baseline outputs")
    s.add_argument("--dir", default="outputs/pressing_baseline")
    s.set_defaults(func=_cmd_pressing_summary)

    v = sub.add_parser("visibility-bias", help="Experiment 1: pressing under a virtual camera")
    v.add_argument("--match", default="J03WPY")
    v.add_argument("--view-lengths", type=float, nargs="+", default=[30, 40, 50, 70, 105])
    v.add_argument("--lag-s", type=float, default=0.0, help="Camera lag (seconds)")
    v.add_argument("--perspective", type=float, default=0.0, help="Trapezoid strength, e.g. 0.3")
    v.add_argument("--zoom-gain", type=float, default=0.0, help="Zoom-out for fast balls, e.g. 0.5")
    v.add_argument("--tag", default="simple", help="Camera name used in file names and tables")
    v.add_argument("--camera-json", nargs="+", default=None,
                   help="Fitted WedgeCamera JSON file(s) or folder(s) (from fit-camera)")
    v.add_argument("--sample-every", type=int, default=1,
                   help="Evaluate every N-th frame only (camera still follows every frame)")
    v.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="With --camera-json: one run per seed, with the fitted framing noise")
    v.add_argument("--speed-threshold", type=float, default=2.0)
    v.add_argument("--gt-dir", default="outputs/pressing_baseline")
    v.add_argument("--out", default="outputs/visibility_bias")
    v.set_defaults(func=_cmd_visibility_bias)

    vs = sub.add_parser("visibility-summary", help="Bias tables from visibility-bias outputs")
    vs.add_argument("--dir", default="outputs/visibility_bias")
    vs.set_defaults(func=_cmd_visibility_summary)

    ci = sub.add_parser("visibility-ci", help="RQ2: bootstrap confidence intervals for the bias")
    ci.add_argument("--dir", default="outputs/visibility_bias")
    ci.add_argument("--block-s", type=float, default=60.0, help="Block length in seconds")
    ci.add_argument("--n-boot", type=int, default=2000)
    ci.add_argument("--level", type=float, default=0.95)
    ci.add_argument("--seed", type=int, default=0)
    ci.set_defaults(func=_cmd_visibility_ci)

    bc = sub.add_parser("bias-correction", help="RQ3: correct pressing bias from visible players")
    bc.add_argument("--dirs", nargs="+", default=["outputs/stadiums/idsse", "outputs/stadiums/metrica"])
    bc.add_argument("--out", default="outputs/correction")
    bc.set_defaults(func=_cmd_bias_correction)

    ss = sub.add_parser("stadium-summary", help="RQ2 part 2: bias across stadium cameras")
    ss.add_argument("--dir", default="outputs/stadiums/idsse")
    ss.add_argument("--n-boot", type=int, default=2000)
    ss.add_argument("--seed", type=int, default=0)
    ss.set_defaults(func=_cmd_stadium_summary)

    fc = sub.add_parser("fit-camera", help="Fit the broadcast camera to SkillCorner footprints")
    fc.add_argument("--data", default="data/skillcorner")
    fc.add_argument("--out", default="outputs/camera_fit")
    fc.add_argument("--max-matches", type=int, default=None)
    fc.add_argument("--per-match", action="store_true",
                    help="Fit one camera per match (stadium) instead of one shared camera")
    fc.set_defaults(func=_cmd_fit_camera)

    ce = sub.add_parser("count-error-propagation", help="B2: count error from video -> corrected pressing error")
    ce.add_argument("--counts", default="outputs/b1_visible/counts_tracks.parquet", help="B1 frame counts")
    ce.add_argument("--table", default="outputs/correction/correction_per_broadcast.csv", help="Phase 1 broadcasts")
    ce.add_argument("--draws", type=int, default=500, help="Draws of the count error per broadcast")
    ce.add_argument("--clips", type=int, default=180, help="30-s clips per match (whole-match averaging)")
    ce.add_argument("--seed", type=int, default=0)
    ce.add_argument("--out", default="outputs/b2_propagation")
    ce.add_argument("--no-plot", action="store_true")
    ce.set_defaults(func=_cmd_count_error)

    fg = sub.add_parser("fit-camera-gsr", help="Journal step 3: one WedgeCamera per SoccerNet-GSR game")
    fg.add_argument("--zips", nargs="+", default=["data/SN-GSR-2025/valid.zip", "data/SN-GSR-2025/test.zip"])
    fg.add_argument("--out", default="configs/cameras/gsr")
    fg.set_defaults(func=_cmd_fit_camera_gsr)

    cl = sub.add_parser("cross-league", help="Journal step 3: A-League correction applied to Swiss cameras")
    cl.add_argument("--train", default="outputs/correction/correction_per_broadcast.csv")
    cl.add_argument("--dirs", nargs="+", default=["outputs/stadiums_gsr/idsse", "outputs/stadiums_gsr/metrica"])
    cl.add_argument("--out", default="outputs/cross_league")
    cl.set_defaults(func=_cmd_cross_league)

    rs = sub.add_parser("replay-share", help="Journal step 5: non-main-camera share of broadcasts, leak tolerance")
    rs.add_argument("--labels", default="data/SN-Labels")
    rs.add_argument("--v-main", type=float, default=14.0, help="Visible players on live main-camera frames")
    rs.add_argument("--max-pp", type=float, default=1.0, help="Tolerated shift of the corrected value (pp)")
    rs.add_argument("--out", default="outputs/s5_replays")
    rs.set_defaults(func=_cmd_replay_share)

    tr = sub.add_parser("team-robustness", help="Journal step 1: is the pressing bias driven by one team?")
    tr.add_argument("--table", default="outputs/correction/correction_per_broadcast.csv")
    tr.add_argument("--team", default="Fortuna Düsseldorf")
    tr.add_argument("--n-boot", type=int, default=2000)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", default="outputs/robustness")
    tr.set_defaults(func=_cmd_team_robustness)

    vd = sub.add_parser("video", help="Broadcast clip -> pitch positions, roles and teams per frame")
    vd.add_argument("input", help="Video file, or a folder with img1/000001.jpg, ...")
    vd.add_argument("--detector", required=True, help="RF-DETR checkpoint (A2)")
    vd.add_argument("--keypoints", required=True, help="Pitch keypoint network best.pt (A5)")
    vd.add_argument("--resolution", type=int, default=1120)
    vd.add_argument("--device", default="cuda")
    vd.add_argument("--out", default="outputs/video/game_state.csv")
    vd.set_defaults(func=_cmd_video)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
