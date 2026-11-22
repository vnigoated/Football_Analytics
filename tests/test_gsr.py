import json

from pitchlens.data.gsr import find_sequences, load_split, split_summary, visible_per_frame


def _labels():
    def obj(i, img, role, team=None, pitch=True):
        return {
            "id": str(i),
            "image_id": img,
            "track_id": i,
            "supercategory": "object",
            "attributes": {"role": role, "team": team, "jersey": "7"},
            "bbox_image": {"x": 10, "y": 20, "w": 30, "h": 60},
            "bbox_pitch": {"x_bottom_middle": 1.5, "y_bottom_middle": -3.0} if pitch else None,
        }

    return {
        "images": [
            {"image_id": "1001", "file_name": "000001.jpg", "has_labeled_person": True},
            {"image_id": "1002", "file_name": "000002.jpg", "has_labeled_person": True},
        ],
        "annotations": [
            obj(1, "1001", "player", "left"),
            obj(2, "1001", "player", "right"),
            obj(3, "1001", "goalkeeper", "left", pitch=False),
            obj(4, "1001", "referee"),
            obj(5, "1001", "ball"),
            obj(1, "1002", "player", "left"),
            {"id": "p1", "image_id": "1001", "supercategory": "pitch", "lines": {}},
        ],
    }


def test_parse_and_count(tmp_path):
    seq = tmp_path / "valid" / "SNGS-001"
    seq.mkdir(parents=True)
    (seq / "Labels-GameState.json").write_text(json.dumps(_labels()))
    assert find_sequences(tmp_path) == [seq]

    images, objects = load_split(tmp_path)
    assert images.height == 2 and objects.height == 6  # pitch annotation skipped
    v = visible_per_frame(images, objects)
    assert v["players"].to_list() == [3, 1]
    assert v["left"].to_list() == [2, 1] and v["right"].to_list() == [1, 0]
    assert v["ball"].to_list() == [True, False]

    s = split_summary(images, objects)
    assert s["sequences"] == 1 and s["tracks"] == 5
    assert s["visible_players_mean"] == 2.0
    assert s["players_with_pitch_pos_share"] == 0.75
    assert s["implausible_pos_share"] == 0.0


def test_implausible_positions_are_flagged(tmp_path):
    labels = _labels()
    labels["annotations"][1]["bbox_pitch"] = {"x_bottom_middle": 4886.0, "y_bottom_middle": 0.0}
    seq = tmp_path / "SNGS-002"
    seq.mkdir()
    (seq / "Labels-GameState.json").write_text(json.dumps(labels))
    images, objects = load_split(tmp_path)
    s = split_summary(images, objects)
    assert s["implausible_pos_share"] == 0.2  # 1 of 5 objects with a position
    assert s["frames_with_implausible_pos"] == 1
    assert s["players_with_pitch_pos_share"] == 0.5  # 2 of 4 players
