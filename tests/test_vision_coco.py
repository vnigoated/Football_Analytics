import json

import numpy as np

from pitchlens.vision.coco import ANNOTATION_FILE, CATEGORY_IDS, clip_box, gsr_to_coco
from pitchlens.vision.metrics import DetectionCounter, box_iou, match_count


def _sequence(root, name="SNGS-001", n_frames=20):
    seq = root / name
    (seq / "img1").mkdir(parents=True)
    images, anns = [], []
    for f in range(1, n_frames + 1):
        img_id = f"9{f:06d}"
        (seq / "img1" / f"{f:06d}.jpg").write_bytes(b"jpg")
        images.append(
            {"image_id": img_id, "file_name": f"{f:06d}.jpg", "width": 1920, "height": 1080}
        )
        for role, box in [
            ("player", {"x": 100, "y": 200, "w": 30, "h": 80}),
            ("ball", {"x": 500, "y": 600, "w": 8, "h": 8}),
            ("other", {"x": 10, "y": 10, "w": 20, "h": 50}),
            ("referee", {"x": -20, "y": 100, "w": 30, "h": 80}),  # partly outside
        ]:
            anns.append(
                {
                    "image_id": img_id,
                    "track_id": len(anns),
                    "supercategory": "object",
                    "attributes": {"role": role},
                    "bbox_image": box,
                }
            )
    (seq / "Labels-GameState.json").write_text(json.dumps({"images": images, "annotations": anns}))


def test_gsr_to_coco(tmp_path):
    _sequence(tmp_path / "valid")
    out = tmp_path / "coco"
    stats = gsr_to_coco(tmp_path / "valid", out, every=10, link=False)
    assert stats["images"] == 2  # frames 1 and 11
    assert stats["annotations"] == 6 and stats["ball"] == 2  # "other" dropped
    coco = json.loads((out / ANNOTATION_FILE).read_text())
    assert {c["name"] for c in coco["categories"]} == set(CATEGORY_IDS)
    assert coco["images"][1]["file_name"] == "SNGS-001_000011.jpg"
    assert (out / "SNGS-001_000011.jpg").exists()
    ref = next(a for a in coco["annotations"] if a["category_id"] == CATEGORY_IDS["referee"])
    assert ref["bbox"] == [0.0, 100.0, 10.0, 80.0]  # clipped at the left edge


def test_gsr_to_coco_person_ball(tmp_path):
    _sequence(tmp_path / "valid")
    out = tmp_path / "coco_pb"
    stats = gsr_to_coco(tmp_path / "valid", out, every=10, link=False, class_set="person_ball")
    assert stats == {"images": 2, "annotations": 6, "person": 4, "ball": 2}
    coco = json.loads((out / ANNOTATION_FILE).read_text())
    assert [c["name"] for c in coco["categories"]] == ["person", "ball"]
    assert {a["category_id"] for a in coco["annotations"]} == {1, 2}


def test_clip_box_outside():
    assert clip_box(1930, 10, 20, 20, 1920, 1080) is None


def test_iou_and_matching():
    a = np.array([[0, 0, 10, 10]])
    assert box_iou(a, a)[0, 0] == 1.0
    assert box_iou(a, [[5, 0, 15, 10]])[0, 0] == 5 * 10 / 150
    gt = np.array([[100, 100, 106, 106]])
    pred = np.array([[103, 103, 109, 109]])  # IoU 9/63, centres 4.2 px apart
    assert match_count(pred, gt) == 0
    assert match_count(pred, gt, max_center_dist=5) == 1
    # two predictions on one ground truth: one match only
    assert match_count(np.vstack([gt, gt]), gt, scores=np.array([0.2, 0.9])) == 1


def test_detection_counter():
    c = DetectionCounter({1: "player", 4: "ball"}, center_dist={4: 10})
    gt = [[0, 0, 10, 20], [50, 50, 60, 70], [200, 200, 206, 206]]
    pred = [[0, 0, 10, 20], [203, 203, 209, 209], [300, 300, 310, 320]]
    c.update(pred, [1, 4, 1], [0.9, 0.8, 0.5], gt, [1, 1, 4])
    rows = {r["class"]: r for r in c.summary()}
    assert rows["player"]["recall"] == 0.5 and rows["player"]["precision"] == 0.5
    assert rows["ball"]["recall"] == 1.0
