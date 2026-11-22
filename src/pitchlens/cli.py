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
    from pitchlens.analytics.pressing import compute_pressing_intensity, summarise_frames
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

        params = json.loads(Path(args.camera_json).read_text())
        params |= {"fps": fps, "pitch_length": dims.pitch_length, "pitch_width": dims.pitch_width}
        for seed in args.seeds or [None]:
            tag = args.tag if seed is None else f"{args.tag}_s{seed}"
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

    for view, cam, tag in runs:
        stats: dict[str, float] = {}

        def mask(df: pl.DataFrame, cam=cam, stats=stats) -> pl.DataFrame:
            masked = apply_broadcast_view(df, cam)
            stats.update(visibility_stats(df, masked))
            return masked

        print(f"[{args.match}] camera={tag} view_length={view} m ...")
        pti, tracking = compute_pressing_intensity(
            ds, speed_threshold=args.speed_threshold, transform=mask
        )
        joined = join_frames(gt, summarise_frames(pti, tracking)).with_columns(
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
    v.add_argument("--camera-json", default=None,
                   help="Use a fitted WedgeCamera (from fit-camera) instead of the rectangle")
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

    fc = sub.add_parser("fit-camera", help="Fit the broadcast camera to SkillCorner footprints")
    fc.add_argument("--data", default="data/skillcorner")
    fc.add_argument("--out", default="outputs/camera_fit")
    fc.add_argument("--max-matches", type=int, default=None)
    fc.set_defaults(func=_cmd_fit_camera)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
