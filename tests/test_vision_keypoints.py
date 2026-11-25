import cv2
import numpy as np
import polars as pl
import pytest

from pitchlens.vision.keypoints import (
    KEYPOINT_NAMES,
    KEYPOINTS_M,
    NUM_KEYPOINTS,
    decode_heatmaps,
    frame_homography,
    heatmap_targets,
    homography_from_keypoints,
    position_errors,
    project_keypoints,
)

# a broadcast-like view: image trapezoid <-> part of the pitch (centre origin, metres)
IMG = np.float32([[0, 1080], [1920, 1080], [1700, 300], [220, 300]])
PIT = np.float32([[-20, -34], [20, -34], [30, 20], [-30, 20]])
H_TRUE = cv2.getPerspectiveTransform(IMG, PIT)  # image -> pitch


def _people(n=12, seed=0):
    rng = np.random.default_rng(seed)
    pitch = np.column_stack([rng.uniform(-18, 18, n), rng.uniform(-30, 15, n)])
    img = cv2.perspectiveTransform(pitch[None].astype(np.float64), np.linalg.inv(H_TRUE))[0]
    w, h = 30.0, 80.0
    return pl.DataFrame({
        "role": ["player"] * n, "x": img[:, 0] - w / 2, "y": img[:, 1] - h, "w": [w] * n, "h": [h] * n,
        "pitch_raw_x": pitch[:, 0], "pitch_raw_y": pitch[:, 1], "pitch_x": pitch[:, 0], "pitch_y": pitch[:, 1],
    })


def test_keypoint_set():
    assert NUM_KEYPOINTS == 33 and KEYPOINTS_M.shape == (33, 2)
    assert np.allclose(KEYPOINTS_M.min(0), [-52.5, -34]) and np.allclose(KEYPOINTS_M.max(0), [52.5, 34])


def test_frame_homography_recovers_camera():
    H, n, res = frame_homography(_people())
    assert n == 12 and res < 1e-3
    assert np.allclose(H / H[2, 2], H_TRUE / H_TRUE[2, 2], atol=1e-6)
    assert frame_homography(_people(n=4))[0] is None  # fewer than min_points


def test_project_and_back():
    kp = project_keypoints(H_TRUE, 1920, 1080)
    vis = kp[:, 2] > 0
    assert 3 <= vis.sum() < NUM_KEYPOINTS  # some landmarks in view, not all
    back = cv2.perspectiveTransform(kp[vis, None, :2], H_TRUE)[:, 0]
    assert np.allclose(back, KEYPOINTS_M[vis], atol=1e-6)


def test_heatmap_roundtrip_and_homography():
    # a denser synthetic set: every keypoint "visible" at its projected position, decoded from heatmaps
    H, _ = cv2.findHomography(PIT, IMG)  # pitch -> image
    uv = cv2.perspectiveTransform(KEYPOINTS_M[:, None].astype(np.float64), H)[:, 0]
    inside = (uv[:, 0] > 0) & (uv[:, 0] < 1920) & (uv[:, 1] > 0) & (uv[:, 1] < 1080)
    kp = np.column_stack([uv, inside.astype(float)])
    dec = decode_heatmaps(heatmap_targets(kp))
    assert ((dec[:, 2] > 0) == inside).all()
    assert np.abs(dec[inside, :2] - uv[inside]).max() < 4.0  # within a few source pixels
    Hk, n = homography_from_keypoints(dec)
    assert Hk is not None and n == inside.sum() >= 6
    assert np.median(position_errors(Hk, _people())) < 0.5
    keep = [KEYPOINT_NAMES.index(n) for n in ("halfway_top", "circle_bottom", "left_pa_tr", "right_pa_tl")]
    assert (dec[keep, 2] > 0).all()  # four detections, no three on one line
    four = dec.copy()
    four[[i for i in range(NUM_KEYPOINTS) if i not in keep], 2] = 0
    assert homography_from_keypoints(four)[0] is None  # unverifiable: rejected by default
    assert homography_from_keypoints(four, min_inliers=4)[0] is not None


def test_network_shapes():
    torch = pytest.importorskip("torch")
    from pitchlens.vision.keypoints import INPUT_SIZE, STRIDE, build_model, heatmap_loss

    m = build_model(pretrained=False).eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, INPUT_SIZE[1], INPUT_SIZE[0]))
    assert tuple(y.shape) == (1, NUM_KEYPOINTS, INPUT_SIZE[1] // STRIDE, INPUT_SIZE[0] // STRIDE)
    assert heatmap_loss(y, torch.zeros_like(y)).item() >= 0
