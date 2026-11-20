"""Image <-> pitch homography estimation.

Detected pitch keypoints (pixels) are matched with known pitch landmarks (metres)
and a planar homography is fitted with RANSAC. The reprojection error is kept,
because it feeds the uncertainty propagation in the research plan (RQ2).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HomographyResult:
    H: np.ndarray  # 3x3, maps image pixels -> pitch metres
    inliers: np.ndarray  # bool mask over the input correspondences
    reprojection_error: float  # mean error (metres) over inliers

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, dtype=np.float64).T
    return homog[:, :2] / homog[:, 2:3]


def fit_homography(
    image_points: np.ndarray,
    pitch_points: np.ndarray,
    ransac_threshold_m: float = 1.0,
) -> HomographyResult:
    """Fit image->pitch homography with RANSAC.

    Args:
        image_points: (N, 2) pixel coordinates of detected keypoints.
        pitch_points: (N, 2) matching pitch coordinates in metres.
        ransac_threshold_m: max reprojection error (metres, on the pitch) for an inlier.
    """
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    pit = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
    if len(img) != len(pit):
        raise ValueError("image_points and pitch_points must have the same length")
    if len(img) < 4:
        raise ValueError("At least 4 correspondences are needed for a homography")

    H, mask = cv2.findHomography(img, pit, cv2.RANSAC, ransac_threshold_m)
    if H is None:
        raise RuntimeError("Homography estimation failed (degenerate correspondences?)")
    inliers = mask.ravel().astype(bool)
    err = np.linalg.norm(apply_homography(H, img[inliers]) - pit[inliers], axis=1)
    return HomographyResult(H=H, inliers=inliers, reprojection_error=float(err.mean()))
