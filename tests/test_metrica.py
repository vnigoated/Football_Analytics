import numpy as np
import polars as pl

from pitchlens.data.metrica import possession_from_events


def _events(rows):
    cols = ["Team", "Type", "Subtype", "Period", "Start Frame", "End Frame"]
    return pl.DataFrame(rows, schema=cols, orient="row")


def test_possession_and_dead_ball():
    ev = _events(
        [
            ("Away", "SET PIECE", "KICK OFF", 1, 1, 0),
            ("Away", "PASS", None, 1, 1, 10),
            ("Away", "BALL LOST", None, 1, 20, 25),
            ("Home", "RECOVERY", None, 1, 26, 0),
            ("Home", "BALL OUT", None, 1, 30, 35),
            ("Away", "SET PIECE", "THROW IN", 1, 40, 0),
            ("Away", "SHOT", "ON TARGET-GOAL", 1, 50, 55),
            ("Home", "SET PIECE", "KICK OFF", 1, 70, 0),
            ("Home", "PASS", None, 1, 70, 80),
        ]
    )
    frames = np.arange(1, 91)
    owner, alive = possession_from_events(ev, frames)
    at = dict(zip(frames.tolist(), zip(owner, alive.tolist())))
    assert at[5] == ("away", True)
    assert at[27] == ("home", True)
    assert at[37][1] is False  # ball out (35) until throw-in (40)
    assert at[40] == ("away", True)
    assert at[60][1] is False  # goal (55) until kick-off (70)
    assert at[75] == ("home", True)
    assert at[85][1] is False  # after the last event
