"""Metrica Sports sample data with possession rebuilt from the event file.

Metrica's open tracking data (games 1 and 2, CSV) has all 22 players + ball at 25 fps but
**no possession or ball-in-play flag**, which the pressing model needs. The matching
event file has frame numbers, so we rebuild both, frame by frame:

- **owner**: team of the most recent PASS, RECOVERY, SHOT, SET PIECE or BALL LOST event;
- **dead ball**: from a BALL OUT, a FAULT RECEIVED, or a SHOT that goes out or scores,
  until the next owner event (the restart); also before the first and after the last
  event of each period.

This is an approximation (events are hand-annotated), so results on Metrica are a
replication check, not a new ground truth.

Licence: the Metrica sample data has no licence file. Its README asks users to be
responsible and to acknowledge the source in public work. We download it at run time
and never redistribute it.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

METRICA_MATCH_IDS = ("metrica-1", "metrica-2")
_EVENTS_URL = (
    "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
    "Sample_Game_{n}/Sample_Game_{n}_RawEventsData.csv"
)
OWNER_TYPES = ("PASS", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST")


def _game_number(match_id: str) -> int:
    if match_id not in METRICA_MATCH_IDS:
        raise ValueError(f"Unknown Metrica match {match_id!r}; expected one of {METRICA_MATCH_IDS}")
    return int(match_id.split("-")[1])


def load_metrica_events(match_id: str, cache_dir: str | Path = "data/metrica") -> pl.DataFrame:
    """Download (once) and read the raw event CSV of a Metrica sample game."""
    n = _game_number(match_id)
    path = Path(cache_dir) / f"Sample_Game_{n}_RawEventsData.csv"
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_EVENTS_URL.format(n=n), path)
    return pl.read_csv(path)


def possession_from_events(
    events: pl.DataFrame, frame_ids: np.ndarray
) -> tuple[list[str | None], np.ndarray]:
    """Per tracking frame: owning team (lower-case 'home'/'away') and ball-alive flag."""
    frame_ids = np.asarray(frame_ids)
    ev = events.with_columns(
        pl.col("Team").str.to_lowercase(),
        pl.col("Subtype").fill_null(""),
        pl.when(pl.col("End Frame") > 0)
        .then(pl.col("End Frame"))
        .otherwise(pl.col("Start Frame"))
        .alias("end"),
    ).sort("Start Frame")

    own = ev.filter(pl.col("Type").is_in(OWNER_TYPES))
    starts = own["Start Frame"].to_numpy()
    teams = own["Team"].to_list()
    idx = np.searchsorted(starts, frame_ids, side="right") - 1
    owner = [teams[i] if i >= 0 else None for i in idx]

    alive = np.ones(len(frame_ids), dtype=bool)

    def kill(a: int, b: int) -> None:  # frames a <= f < b are dead
        alive[(frame_ids >= a) & (frame_ids < b)] = False

    stops = ev.filter(
        pl.col("Type").is_in(["BALL OUT", "FAULT RECEIVED"])
        | ((pl.col("Type") == "SHOT") & pl.col("Subtype").str.contains("OUT|GOAL"))
    )
    for typ, start, end in stops.select("Type", "Start Frame", "end").iter_rows():
        dead_from = start if typ == "FAULT RECEIVED" else end
        nxt = starts[starts > dead_from]
        kill(dead_from, nxt[0] if len(nxt) else frame_ids.max() + 1)

    periods = ev.group_by("Period").agg(pl.col("Start Frame").min(), pl.col("end").max()).sort("Period")
    bounds = periods.rows()
    kill(frame_ids.min(), bounds[0][1])
    for (_, _, end_p), (_, start_next, _) in pairwise(bounds):
        kill(end_p + 1, start_next)
    kill(bounds[-1][2] + 1, frame_ids.max() + 1)
    return owner, alive


def load_metrica_tracking(match_id: str = "metrica-1", limit: int | None = None):
    """Metrica tracking (kloppy) with ``ball_owning_team`` and ``ball_state`` filled in.

    Only frames with the ball in play are kept (as for IDSSE with ``only_alive=True``).
    """
    from kloppy import metrica
    from kloppy.domain import BallState

    n = _game_number(match_id)
    ds = metrica.load_open_data(match_id=n, limit=limit)
    owner, alive = possession_from_events(
        load_metrica_events(match_id), np.array([f.frame_id for f in ds.frames])
    )
    team_by_id = {t.team_id: t for t in ds.metadata.teams}
    for frame, team, is_alive in zip(ds.frames, owner, alive):
        frame.ball_owning_team = team_by_id.get(team) if team else None
        frame.ball_state = BallState.ALIVE if is_alive else BallState.DEAD
    return ds.filter(lambda f: f.ball_state == BallState.ALIVE and f.ball_owning_team is not None)
