"""Standard football pitch model in metres.

Coordinate system: origin at the bottom-left corner, x along the length (0..L),
y along the width (0..W). Default 105 x 68 m (FIFA recommended, also used by IDSSE).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pitch:
    length: float = 105.0
    width: float = 68.0
    penalty_area_depth: float = 16.5
    penalty_area_width: float = 40.32
    goal_area_depth: float = 5.5
    goal_area_width: float = 18.32
    penalty_spot_distance: float = 11.0
    centre_circle_radius: float = 9.15

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Named pitch landmarks usable as homography correspondences."""
        L, W = self.length, self.width
        cy = W / 2
        pa_y0, pa_y1 = cy - self.penalty_area_width / 2, cy + self.penalty_area_width / 2
        ga_y0, ga_y1 = cy - self.goal_area_width / 2, cy + self.goal_area_width / 2
        pa, ga = self.penalty_area_depth, self.goal_area_depth
        r = self.centre_circle_radius
        return {
            # corners
            "corner_bl": (0.0, 0.0),
            "corner_tl": (0.0, W),
            "corner_br": (L, 0.0),
            "corner_tr": (L, W),
            # halfway line + centre circle
            "halfway_bottom": (L / 2, 0.0),
            "halfway_top": (L / 2, W),
            "centre_spot": (L / 2, cy),
            "circle_bottom": (L / 2, cy - r),
            "circle_top": (L / 2, cy + r),
            "circle_left": (L / 2 - r, cy),
            "circle_right": (L / 2 + r, cy),
            # left penalty area / goal area / spot
            "left_pa_bl": (0.0, pa_y0),
            "left_pa_tl": (0.0, pa_y1),
            "left_pa_br": (pa, pa_y0),
            "left_pa_tr": (pa, pa_y1),
            "left_ga_bl": (0.0, ga_y0),
            "left_ga_tl": (0.0, ga_y1),
            "left_ga_br": (ga, ga_y0),
            "left_ga_tr": (ga, ga_y1),
            "left_penalty_spot": (self.penalty_spot_distance, cy),
            # right penalty area / goal area / spot
            "right_pa_br": (L, pa_y0),
            "right_pa_tr": (L, pa_y1),
            "right_pa_bl": (L - pa, pa_y0),
            "right_pa_tl": (L - pa, pa_y1),
            "right_ga_br": (L, ga_y0),
            "right_ga_tr": (L, ga_y1),
            "right_ga_bl": (L - ga, ga_y0),
            "right_ga_tl": (L - ga, ga_y1),
            "right_penalty_spot": (L - self.penalty_spot_distance, cy),
        }

    def landmark_array(self, names: list[str] | None = None) -> np.ndarray:
        lm = self.landmarks()
        names = names or list(lm)
        return np.array([lm[n] for n in names], dtype=np.float64)

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask: which (N, 2) points lie on the pitch (plus optional margin)."""
        xy = np.asarray(xy, dtype=np.float64)
        return (
            (xy[:, 0] >= -margin)
            & (xy[:, 0] <= self.length + margin)
            & (xy[:, 1] >= -margin)
            & (xy[:, 1] <= self.width + margin)
        )
