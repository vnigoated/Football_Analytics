"""Simple detection metrics: per-class precision / recall with greedy matching.

COCO mAP comes from RF-DETR's own evaluation during training. These metrics answer the
questions that matter for the pipeline: how many players and balls are found at all.
For the ball, a box IoU is too strict (a few pixels wide), so a centre-distance match
can be used instead.
"""

from __future__ import annotations

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` (N, 4) and ``b`` (M, 4), boxes as x0, y0, x1, y1."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_count(
    pred: np.ndarray,
    gt: np.ndarray,
    scores: np.ndarray | None = None,
    iou_thr: float = 0.5,
    max_center_dist: float | None = None,
) -> int:
    """Number of one-to-one matches, predictions taken in descending score order.

    A pair matches if IoU >= ``iou_thr`` or, when ``max_center_dist`` is given, if the box
    centres are at most that many pixels apart.
    """
    pred, gt = np.asarray(pred, float).reshape(-1, 4), np.asarray(gt, float).reshape(-1, 4)
    if len(pred) == 0 or len(gt) == 0:
        return 0
    order = np.argsort(-np.asarray(scores)) if scores is not None else np.arange(len(pred))
    iou = box_iou(pred, gt)
    ok = iou >= iou_thr
    if max_center_dist is not None:
        cp = (pred[:, :2] + pred[:, 2:]) / 2
        cg = (gt[:, :2] + gt[:, 2:]) / 2
        dist = np.linalg.norm(cp[:, None] - cg[None], axis=2)
        ok |= dist <= max_center_dist
    used = np.zeros(len(gt), bool)
    n = 0
    for i in order:
        cand = np.where(ok[i] & ~used)[0]
        if len(cand):
            j = cand[np.argmax(iou[i, cand])]
            used[j] = True
            n += 1
    return n


class DetectionCounter:
    """Accumulate true positives, predictions and ground truths per class over many images."""

    def __init__(self, classes: dict[int, str], center_dist: dict[int, float] | None = None):
        self.classes = classes
        self.center_dist = center_dist or {}
        self.tp = dict.fromkeys(classes, 0)
        self.n_pred = dict.fromkeys(classes, 0)
        self.n_gt = dict.fromkeys(classes, 0)

    def update(self, pred_xyxy, pred_cls, pred_scores, gt_xyxy, gt_cls, iou_thr=0.5):
        pred_xyxy, gt_xyxy = np.asarray(pred_xyxy, float), np.asarray(gt_xyxy, float)
        pred_cls, gt_cls = np.asarray(pred_cls), np.asarray(gt_cls)
        pred_scores = np.asarray(pred_scores, float)
        for c in self.classes:
            p, g = pred_cls == c, gt_cls == c
            self.n_pred[c] += int(p.sum())
            self.n_gt[c] += int(g.sum())
            self.tp[c] += match_count(
                pred_xyxy[p], gt_xyxy[g], pred_scores[p], iou_thr, self.center_dist.get(c)
            )

    def summary(self) -> list[dict]:
        rows = []
        for c, name in self.classes.items():
            tp, npred, ngt = self.tp[c], self.n_pred[c], self.n_gt[c]
            rows.append(
                {
                    "class": name,
                    "ground_truth": ngt,
                    "predicted": npred,
                    "recall": round(tp / ngt, 3) if ngt else None,
                    "precision": round(tp / npred, 3) if npred else None,
                }
            )
        return rows
