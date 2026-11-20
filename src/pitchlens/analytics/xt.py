"""Expected Threat (xT), after Karun Singh (2019).

The pitch is split into an l x w grid. For each cell we estimate:
    s = P(shoot | ball in cell),  m = P(move | ball in cell)
    g = P(goal | shot from cell), T = P(move ends in cell z | move from cell)
and solve xT = s*g + m * T @ xT by fixed-point iteration.
Failed moves count towards m but never transition, so they contribute 0 value.

Coordinates are metres, origin bottom-left, team attacking left -> right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExpectedThreat:
    n_x: int = 16
    n_y: int = 12
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    grid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.grid = np.zeros((self.n_y, self.n_x))

    def cell_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cx = np.clip((np.asarray(x) / self.pitch_length * self.n_x).astype(int), 0, self.n_x - 1)
        cy = np.clip((np.asarray(y) / self.pitch_width * self.n_y).astype(int), 0, self.n_y - 1)
        return cy * self.n_x + cx

    def fit(
        self,
        start_x: np.ndarray,
        start_y: np.ndarray,
        end_x: np.ndarray,
        end_y: np.ndarray,
        is_shot: np.ndarray,
        success: np.ndarray,
        is_goal: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> ExpectedThreat:
        """Fit xT from actions. ``is_shot`` False means a move (pass/carry/dribble)."""
        is_shot = np.asarray(is_shot, bool)
        success = np.asarray(success, bool)
        is_goal = np.asarray(is_goal, bool)
        n = self.n_x * self.n_y
        start = self.cell_index(start_x, start_y)
        end = self.cell_index(end_x, end_y)

        shots = np.bincount(start[is_shot], minlength=n).astype(float)
        goals = np.bincount(start[is_shot & is_goal], minlength=n).astype(float)
        moves = np.bincount(start[~is_shot], minlength=n).astype(float)

        T = np.zeros((n, n))
        ok = ~is_shot & success
        np.add.at(T, (start[ok], end[ok]), 1.0)

        total = shots + moves
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(total > 0, shots / total, 0.0)
            m = np.where(total > 0, moves / total, 0.0)
            g = np.where(shots > 0, goals / shots, 0.0)
            T = np.where(moves[:, None] > 0, T / moves[:, None], 0.0)

        xt = np.zeros(n)
        for _ in range(max_iter):
            new = s * g + m * (T @ xt)
            if np.max(np.abs(new - xt)) < tol:
                xt = new
                break
            xt = new
        self.grid = xt.reshape(self.n_y, self.n_x)
        return self

    def value(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.grid.ravel()[self.cell_index(x, y)]

    def rate(
        self,
        start_x: np.ndarray,
        start_y: np.ndarray,
        end_x: np.ndarray,
        end_y: np.ndarray,
        success: np.ndarray,
    ) -> np.ndarray:
        """xT added by each move: xT(end) - xT(start) if successful, else 0."""
        delta = self.value(end_x, end_y) - self.value(start_x, start_y)
        return np.where(np.asarray(success, bool), delta, 0.0)
