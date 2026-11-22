import polars as pl
import pytest

from pitchlens.analytics.compare import bias_table, join_frames
from pitchlens.simulation.camera import BroadcastCamera
from pitchlens.simulation.occlusion import apply_broadcast_view, visibility_stats


def _tracking():
    # one frame, centred coordinates; ball at the centre spot
    return pl.DataFrame(
        {
            "period_id": [1] * 4,
            "frame_id": [1] * 4,
            "id": ["ball", "near", "far", "carrier_far"],
            "x": [0.0, 10.0, 40.0, -45.0],
            "y": [0.0, 5.0, 0.0, 0.0],
            "is_ball_carrier": [False, False, False, True],
        }
    )


def test_apply_broadcast_view_hides_far_players():
    cam = BroadcastCamera(view_length=40)
    out = apply_broadcast_view(_tracking(), cam, keep_carrier=False)
    assert sorted(out["id"].to_list()) == ["ball", "near"]
    out = apply_broadcast_view(_tracking(), cam, keep_carrier=True)
    assert sorted(out["id"].to_list()) == ["ball", "carrier_far", "near"]


def test_full_pitch_view_keeps_everyone():
    t = _tracking()
    out = apply_broadcast_view(t, BroadcastCamera(view_length=105), keep_carrier=False)
    assert out.height == t.height
    assert visibility_stats(t, out)["share_visible"] == pytest.approx(1.0)


def test_join_and_bias():
    gt = pl.DataFrame(
        {
            "period_id": [1, 1],
            "frame_id": [1, 2],
            "match_id": ["m", "m"],
            "pressing_team_id": ["D", "D"],
            "pressing_team_name": ["Def", "Def"],
            "pressure_on_carrier": [0.8, 0.4],
            "pressure_total": [2.0, 1.0],
            "n_active_pressers": [2, 1],
        }
    )
    # frame 2 missing from the simulation (no defender on screen) -> counts as 0
    sim = pl.DataFrame(
        {
            "period_id": [1],
            "frame_id": [1],
            "pressing_team_id": ["D"],
            "pressure_on_carrier": [0.6],
            "pressure_total": [1.0],
            "n_active_pressers": [1],
        }
    )
    j = join_frames(gt, sim)
    assert j["pressure_on_carrier_sim"].to_list() == pytest.approx([0.6, 0.0])
    row = bias_table(j).row(0, named=True)
    assert row["mean_pressure_gt"] == pytest.approx(0.6)
    assert row["mean_pressure_sim"] == pytest.approx(0.3)
    assert row["rel_bias_pressure_pct"] == pytest.approx(-50.0)
    assert row["frame_mae"] == pytest.approx(0.3)


def test_players_beyond_the_line_stay_visible_at_the_edge():
    # ball near the left goal line; keeper 1 m behind it, player beyond the touchline
    t = pl.DataFrame(
        {
            "period_id": [1, 1, 1],
            "frame_id": [1, 1, 1],
            "id": ["ball", "keeper", "thrower"],
            "x": [-50.0, -53.5, -40.0],
            "y": [0.0, 0.0, 35.0],
            "is_ball_carrier": [False, False, False],
        }
    )
    out = apply_broadcast_view(t, BroadcastCamera(view_length=40), keep_carrier=False)
    assert out.height == 3


def test_perspective_far_side_sees_more():
    # ball at the centre; two players 23 m along the pitch, one near, one far touchline
    t = pl.DataFrame(
        {
            "period_id": [1] * 3,
            "frame_id": [1] * 3,
            "id": ["ball", "near", "far"],
            "x": [0.0, 23.0, 23.0],
            "y": [0.0, -30.0, 30.0],
            "is_ball_carrier": [False, False, False],
        }
    )
    flat = apply_broadcast_view(t, BroadcastCamera(view_length=40), keep_carrier=False)
    assert sorted(flat["id"].to_list()) == ["ball"]
    persp = apply_broadcast_view(
        t, BroadcastCamera(view_length=40, perspective=0.3), keep_carrier=False
    )
    assert sorted(persp["id"].to_list()) == ["ball", "far"]


def test_wedge_camera_pans_with_ball():
    from pitchlens.simulation.camera import WedgeCamera

    cam = WedgeCamera(cam_x=0, cam_y=-57, pan_gain=1.0, lag_s=0, fov_a=0.5, fov_b=0.0,
                      fov_min=0.1, fov_max=1.0, near_a=0.0, near_b=0.0)  # fmt: skip
    t = pl.DataFrame(
        {
            "period_id": [1] * 4,
            "frame_id": [1] * 4,
            "id": ["ball", "with_ball", "other_wing", "behind_camera_line"],
            "x": [30.0, 32.0, -40.0, 30.0],
            "y": [0.0, 5.0, 0.0, -60.0],
            "is_ball_carrier": [False] * 4,
        }
    )
    out = apply_broadcast_view(t, cam, keep_carrier=False)
    assert sorted(out["id"].to_list()) == ["ball", "with_ball"]
