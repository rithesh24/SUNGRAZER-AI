"""Ranking evaluation metrics (claude.md section 18), pure NumPy.

Implemented without scikit-learn: the project needs five small formulas,
not a dependency. Average precision follows the standard step-wise
definition (sum over positives of precision at each positive's rank).
"""

from __future__ import annotations

import numpy as np


def precision_recall_f1(labels: np.ndarray, scores: np.ndarray,
                        threshold: float = 0.5) -> dict:
    """Precision/recall/F1 of ``scores >= threshold`` against 0/1 labels."""
    predicted = scores >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC as average precision (step interpolation, sklearn-equivalent)."""
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    hits = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, len(sorted_labels) + 1)
    precision_at_rank = hits / ranks
    return float(precision_at_rank[sorted_labels == 1].sum() / n_pos)


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the top-k scored items that are positive."""
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    top = np.argsort(-scores, kind="stable")[:k]
    return float((labels[top] == 1).mean())


def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of all positives found in the top-k scored items."""
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return 0.0
    top = np.argsort(-scores, kind="stable")[:min(k, len(scores))]
    return float((labels[top] == 1).sum() / n_pos)


def ranking_report(labels: np.ndarray, scores: np.ndarray,
                   ks: tuple[int, ...] = (10, 50)) -> dict:
    """All ranking metrics in one dict (used by ml.train evaluation)."""
    report = precision_recall_f1(labels, scores)
    report["average_precision"] = average_precision(labels, scores)
    for k in ks:
        report[f"precision_at_{k}"] = precision_at_k(labels, scores, k)
        report[f"recall_at_{k}"] = recall_at_k(labels, scores, k)
    report["n_pos"] = int((labels == 1).sum())
    report["n_total"] = int(len(labels))
    return report
