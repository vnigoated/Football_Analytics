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

METRICA_MATCH_IDS = ("metrica-1", "metrica-2", "metrica-3")
_EVENTS_URL = (
    "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
    "Sample_Game_{n}/Sample_Game_{n}_RawEventsData.csv"
)
# Game 3 is in the newer format: FIFA EPTS tracking (read by kloppy) and events as JSON.
_EVENTS_JSON_URL = (
    "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
    "Sample_Game_{n}/Sample_Game_{n}_events.json"
)
OWNER_TYPES = ("PASS", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST")


def _game_number(match_id: str) -> int:
    if match_id not in METRICA_MATCH_IDS:
        raise ValueError(f"Unknown Metrica match {match_id!r}; expected one of {METRICA_MATCH_IDS}")
    return int(match_id.split("-")[1])


def events_from_json(data: dict) -> pl.DataFrame:
    """Metrica's JSON events (game 3) as the columns of the raw event CSV (games 1-2): ``Team`` (team id), ``Type``,
    ``Subtype`` (subtype names joined with "-"), ``Period``, ``Start Frame``, ``End Frame``."""
    rows = []
    for e in data["data"]:
        subs = e.get("subtypes") or []
        subs = subs if isinstance(subs, list) else [subs]
        end = (e.get("end") or {}).get("frame")
        rows.append({
            "Team": (e.get("team") or {}).get("id"),
            "Type": (e.get("type") or {}).get("name"),
            "Subtype": "-".join(s["name"] for s in subs if s and s.get("name")) or None,
            "Period": e.get("period"),
            "Start Frame": e["start"]["frame"],
            "End Frame": end if end is not None else e["start"]["frame"],
        })
    schema = {"Team": pl.String, "Type": pl.String, "Subtype": pl.String, "Period": pl.Int64,
              "Start Frame": pl.Int64, "End Frame": pl.Int64}
    return pl.DataFrame(rows, schema=schema)


def load_metrica_events(match_id: str, cache_dir: str | Path = "data/metrica") -> pl.DataFrame:
    """Download (once) and read the events of a Metrica sample game (CSV for games 1-2, JSON for game 3)."""
    import json
    import urllib.request

    n = _game_number(match_id)
    as_json = n >= 3
    path = Path(cache_dir) / (f"Sample_Game_{n}_events.json" if as_json else f"Sample_Game_{n}_RawEventsData.csv")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve((_EVENTS_JSON_URL if as_json else _EVENTS_URL).format(n=n), path)
    if as_json:
        return events_from_json(json.loads(path.read_text(encoding="utf-8")))
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
    team_by_id = {t.team_id.lower(): t for t in ds.metadata.teams}  # "home" / "away" (1-2), "fifatma" (3)
    for frame, team, is_alive in zip(ds.frames, owner, alive):
        frame.ball_owning_team = team_by_id.get(team) if team else None
        frame.ball_state = BallState.ALIVE if is_alive else BallState.DEAD
    return ds.filter(lambda f: f.ball_state == BallState.ALIVE and f.ball_owning_team is not None)
