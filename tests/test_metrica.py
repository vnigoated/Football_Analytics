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


def test_events_from_json_game3_format():
    from pitchlens.data.metrica import events_from_json, possession_from_events

    data = {"data": [
        {"team": {"id": "FIFATMA"}, "type": {"name": "SET PIECE"}, "subtypes": {"name": "KICK OFF"},
         "start": {"frame": 10}, "end": {"frame": 10}, "period": 1},
        {"team": {"id": "FIFATMA"}, "type": {"name": "PASS"}, "subtypes": None,
         "start": {"frame": 20}, "end": {"frame": 30}, "period": 1},
        {"team": {"id": "FIFATMB"}, "type": {"name": "SHOT"},
         "subtypes": [{"name": "OFF TARGET"}, {"name": "OUT"}], "start": {"frame": 50}, "end": {"frame": 60},
         "period": 1},
        {"team": {"id": "FIFATMA"}, "type": {"name": "SET PIECE"}, "subtypes": {"name": "GOAL KICK"},
         "start": {"frame": 80}, "end": {"frame": None}, "period": 1},
    ]}
    ev = events_from_json(data)
    assert ev.columns == ["Team", "Type", "Subtype", "Period", "Start Frame", "End Frame"]
    assert ev["Subtype"].to_list() == ["KICK OFF", None, "OFF TARGET-OUT", "GOAL KICK"]
    assert ev["End Frame"].to_list() == [10, 30, 60, 80]
    import numpy as np

    owner, alive = possession_from_events(ev, np.arange(10, 90))
    assert owner[25 - 10] == "fifatma" and owner[55 - 10] == "fifatmb"
    assert not alive[70 - 10] and alive[80 - 10]  # dead after the shot goes out, alive again at the goal kick
