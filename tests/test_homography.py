import numpy as np
import pytest

from pitchlens.geometry.homography import apply_homography, fit_homography
from pitchlens.geometry.pitch import Pitch

# A plausible broadcast-like homography (pitch metres -> image pixels).
H_PITCH_TO_IMG = np.array(
    [
        [12.0, -3.0, 200.0],
        [0.5, -6.0, 700.0],
        [0.0005, -0.004, 1.0],
    ]
)


def _correspondences():
    pitch_pts = Pitch().landmark_array()
    img_pts = apply_homography(H_PITCH_TO_IMG, pitch_pts)
    return img_pts, pitch_pts


def test_recovers_exact_homography():
    img, pit = _correspondences()
    res = fit_homography(img, pit)
    assert res.n_inliers == len(pit)
    assert res.reprojection_error < 1e-4  # metres: sub-millimetre
    np.testing.assert_allclose(apply_homography(res.H, img), pit, atol=1e-4)


def test_ransac_rejects_outliers():
    img, pit = _correspondences()
    img = img.copy()
    img[:4] += np.array([150.0, -90.0])  # 4 badly detected keypoints
    res = fit_homography(img, pit, ransac_threshold_m=0.5)
    assert not res.inliers[:4].any()
    assert res.inliers[4:].all()
    assert res.reprojection_error < 1e-3


def test_needs_four_points():
    img, pit = _correspondences()
    with pytest.raises(ValueError):
        fit_homography(img[:3], pit[:3])


def test_pitch_landmarks_symmetric():
    p = Pitch()
    lm = p.landmarks()
    assert lm["centre_spot"] == (p.length / 2, p.width / 2)
    lx, ly = lm["left_penalty_spot"]
    rx, ry = lm["right_penalty_spot"]
    assert lx + rx == pytest.approx(p.length) and ly == ry
    assert p.contains(p.landmark_array()).all()
