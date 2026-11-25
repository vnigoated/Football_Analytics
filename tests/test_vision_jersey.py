import numpy as np
import polars as pl
import pytest

from pitchlens.vision.jersey import (
    CROP_SIZE,
    crops_for,
    jersey_crop,
    number_probs,
    number_to_digits,
    sample_jersey_crops,
    track_accuracy,
    vote_numbers,
)


def test_number_to_digits():
    assert number_to_digits("7") == (0, 7)
    assert number_to_digits(23) == (2, 3)
    assert number_to_digits("99") == (9, 9)
    assert number_to_digits(None) is None
    assert number_to_digits("0") is None
    assert number_to_digits("123") is None
    assert number_to_digits("x") is None


def test_number_probs_combines_digits():
    logits = np.full((1, 20), -10.0)
    logits[0, 2] = 10  # tens 2
    logits[0, 10 + 3] = 10  # units 3
    p = number_probs(logits)
    assert p.shape == (1, 100)
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 23
    logits[0, 2], logits[0, 0] = -10, 10  # tens "none" -> one-digit number
    assert number_probs(logits).argmax() == 3


def test_vote_numbers_uses_confident_crops_only():
    def onehot(n, conf):
        p = np.full(100, (1 - conf) / 99)
        p[n] = conf
        return p

    ids = np.array([1, 1, 1, 2, 2, 3])
    probs = np.stack([onehot(10, 0.9), onehot(10, 0.8), onehot(7, 0.3),  # track 1: two confident 10s
                      onehot(5, 0.2), onehot(6, 0.2),  # track 2: nothing confident
                      onehot(9, 0.95)])
    v = vote_numbers(ids, probs, min_conf=0.5)
    assert v == {1: "10", 2: None, 3: "9"}
    assert vote_numbers(ids, probs, min_conf=0.5, min_votes=2)[3] is None


def test_crops_and_sampling():
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[100:200, 100:150] = 255
    c = jersey_crop(img, (100, 100, 150, 200))
    assert c.shape == (CROP_SIZE[1], CROP_SIZE[0], 3)
    assert jersey_crop(img, (5000, 5000, 5010, 5010)).shape == (CROP_SIZE[1], CROP_SIZE[0], 3)
    tracks = pl.DataFrame({"frame": [1, 2, 3, 5], "id": [1, 1, 1, 2], "x1": [0.0, 0, 0, 0], "y1": [0.0, 0, 0, 0],
                           "x2": [30.0, 30, 30, 30], "y2": [60.0, 80, 70, 20]})
    s = sample_jersey_crops(tracks, every=2, max_per_track=1, min_height=40)
    assert s["frame"].to_list() == [3]  # frames 1 and 3 qualify; the taller (frame 3) is kept; track 2 too small
    crops = crops_for(s, lambda f: img)
    assert len(crops) == 1 and crops[0].shape == (CROP_SIZE[1], CROP_SIZE[0], 3)


def test_track_accuracy():
    a = track_accuracy({1: "10", 2: None, 3: "4", 4: None}, {1: "10", 2: None, 3: "5", 4: "8"})
    assert a["tracks"] == 4
    assert a["correct"] == 0.5
    assert a["numbered_read"] == pytest.approx(1 / 3, abs=1e-4)
    assert a["wrong_number"] == 0.25


def test_model_forward_and_loss():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from pitchlens.vision.jersey import build_model, digit_loss, to_input

    model = build_model(pretrained=False).eval()
    x = torch.from_numpy(np.stack([to_input(np.zeros((CROP_SIZE[1], CROP_SIZE[0], 3), np.uint8),
                                            np.random.default_rng(0))] * 2))
    out = model(x)
    assert out.shape == (2, 20)
    assert float(digit_loss(out, torch.tensor([[1, 0], [0, 7]]))) > 0
