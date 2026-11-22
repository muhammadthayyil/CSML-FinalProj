"""Evaluation metrics shared by every approach.

The headline research texture is CALIBRATION, so this module centers on ECE / Brier /
reliability diagrams alongside the usual accuracy / macro-F1.

All functions take:
    y_true:     (N,) int array of gold label indices
    y_pred:     (N,) int array of predicted label indices
    confidence: (N,) float array in [0,1], the model's confidence in its prediction
Optionally:
    probs:      (N, K) float array of full class probabilities (enables multiclass Brier)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    ece: float
    mce: float
    brier: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def expected_calibration_error(
    correct: np.ndarray, confidence: np.ndarray, n_bins: int = 15
):
    """Equal-width binning ECE/MCE on top-label confidence.

    Returns (ece, mce, bins) where bins is a list of per-bin stats for reliability plots.
    """
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, bins = 0.0, 0.0, []
    n = len(confidence)
    for lo, hi in zip(edges[:-1], edges[1:]):
        # include right edge in the last bin
        if hi == 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": lo, "hi": hi, "count": 0, "acc": None, "conf": None})
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        gap = abs(acc - conf)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append({"lo": lo, "hi": hi, "count": count, "acc": acc, "conf": conf})
    return ece, mce, bins


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error between one-hot gold and predicted probability vectors."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    n, k = probs.shape
    onehot = np.zeros((n, k), dtype=float)
    onehot[np.arange(n), y_true] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    probs: Optional[np.ndarray] = None,
    n_bins: int = 15,
):
    """Compute the full metric suite. Returns (EvalResult, reliability_bins)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)
    correct = (y_true == y_pred).astype(float)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    ece, mce, bins = expected_calibration_error(correct, confidence, n_bins=n_bins)

    if probs is not None:
        brier = multiclass_brier(y_true, probs)
    else:
        # Fall back to a two-class Brier on (confidence, correctness).
        brier = float(((confidence - correct) ** 2).mean())

    return EvalResult(acc, macro_f1, ece, mce, brier, int(len(y_true))), bins


def temperature_scale(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 200):
    """Fit a single temperature T on a validation set (minimize NLL). Returns T (float).

    logits: (N, K) raw scores. Use the fitted T on test-set logits: softmax(logits / T).
    """
    from scipy.optimize import minimize_scalar  # local import; scipy optional

    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true)

    def nll(T: float) -> float:
        z = logits / max(T, 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y_true)), y_true].mean())

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded",
                          options={"maxiter": max_iter})
    return float(res.x)
