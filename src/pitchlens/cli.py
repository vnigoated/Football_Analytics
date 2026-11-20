"""Command-line interface: ``pitchlens <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_download_gsr(args: argparse.Namespace) -> None:
    from pitchlens.data.datasets import download_sn_gsr

    path = download_sn_gsr(args.dest, tuple(args.splits), args.revision)
    print(f"Downloaded {args.splits} to {path}")


def _cmd_pressing_baseline(args: argparse.Namespace) -> None:
    from pitchlens.analytics.pressing import compute_pressing_intensity, summarise_frames
    from pitchlens.data.datasets import load_idsse_tracking

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading IDSSE match {args.match} (limit={args.limit}) ...")
    ds = load_idsse_tracking(args.match, limit=args.limit)
    print("Computing pressing intensity ...")
    pti, tracking = compute_pressing_intensity(ds, speed_threshold=args.speed_threshold)
    summary = summarise_frames(pti, tracking)
    summary_path = out / f"pressing_{args.match}.parquet"
    summary.write_parquet(summary_path)
    print(f"Saved {len(summary)} frames -> {summary_path}")

    if not args.no_plot:
        from pitchlens.viz import plot_pressing_timeline

        team_names = {t.team_id: t.name for t in ds.metadata.teams}
        fig_path = out / f"pressing_{args.match}.png"
        plot_pressing_timeline(summary, team_names, fig_path)
        print(f"Saved plot -> {fig_path}")


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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
