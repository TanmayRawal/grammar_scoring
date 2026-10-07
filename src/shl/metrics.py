"""Evaluation metrics for the SHL grammar-scoring task.

The leaderboard combines RMSE and Pearson correlation. We track the
composite ``(RMSE + (1 - r)) / 2`` (lower is better). That formula is
inferred from public CV/LB pairs, not stated by Kaggle. RMSE and r are
always reported separately as well.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred, w=None) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    w = np.ones(len(y_true)) if w is None else np.asarray(w, float)
    return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=w)))


def pearson(y_true, y_pred, w=None) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    w = np.ones(len(y_true)) if w is None else np.asarray(w, float)
    mt, mp = np.average(y_true, weights=w), np.average(y_pred, weights=w)
    vt = np.average((y_true - mt) ** 2, weights=w)
    vp = np.average((y_pred - mp) ** 2, weights=w)
    if vt == 0 or vp == 0:
        return 0.0
    return float(np.average((y_true - mt) * (y_pred - mp), weights=w) / np.sqrt(vt * vp))


# Leaderboard metric, identified from 5 diagnostic submissions (run-3 model
# shifted by 0, -0.5, -1.0, -1.5 and shrunk 2x around its mean):
#   score = 0.6 * RMSE + 0.4 * (1 - Pearson)   (max misfit 0.0003)
# The 50/50 form assumed in public write-ups misfits by 0.033.
W_RMSE, W_PEARSON = 0.6, 0.4


def composite(y_true, y_pred, w=None) -> float:
    """Leaderboard metric 0.6*RMSE + 0.4*(1 - Pearson). Lower is better.

    With ``w`` (e.g. ``population.importance_weights``) both terms are
    computed on the weighted sample: the *test-weighted* composite.
    """
    return W_RMSE * rmse(y_true, y_pred, w) + W_PEARSON * (1.0 - pearson(y_true, y_pred, w))


def composite_5050(y_true, y_pred, w=None) -> float:
    """The (RMSE + 1 - r) / 2 form used before the metric was identified."""
    return 0.5 * (rmse(y_true, y_pred, w) + 1.0 - pearson(y_true, y_pred, w))


def bootstrap_ci(y_true, y_pred, metric=composite, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``metric``."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_delta(y_true, pred_a, pred_b, metric=composite,
                           n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the difference metric(b) - metric(a) on identical resamples.

    Used for ablations: a negative mean delta means ``b`` is better.
    """
    y_true = np.asarray(y_true, float)
    pred_a, pred_b = np.asarray(pred_a, float), np.asarray(pred_b, float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = metric(y_true[idx], pred_b[idx]) - metric(y_true[idx], pred_a[idx])
    return {
        "delta": float(metric(y_true, pred_b) - metric(y_true, pred_a)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "p_better": float(np.mean(deltas < 0)),
    }


def report(y_true, y_pred, name: str = "", ci: bool = True, w=None) -> dict:
    """One-line summary dict used for every result in the project.

    With ``w`` it adds the test-weighted metrics (``*_tw``).
    """
    out = {
        "name": name,
        "rmse": rmse(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "composite": composite(y_true, y_pred),
    }
    if w is not None:
        out.update(rmse_tw=rmse(y_true, y_pred, w), pearson_tw=pearson(y_true, y_pred, w),
                   composite_tw=composite(y_true, y_pred, w))
    if ci:
        if w is None:
            out["ci_low"], out["ci_high"] = bootstrap_ci(y_true, y_pred)
        else:
            out["ci_low"], out["ci_high"] = bootstrap_ci_weighted(y_true, y_pred, w)
    return out


def bootstrap_ci_weighted(y_true, y_pred, w, n_boot: int = 2000, alpha: float = 0.05,
                          seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the weighted composite (clips resampled, weights kept)."""
    y_true, y_pred, w = (np.asarray(a, float) for a in (y_true, y_pred, w))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = composite(y_true[idx], y_pred[idx], w[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
