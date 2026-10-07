"""Recording populations and test-composition weighting.

Clip length marks distinct populations with different label distributions:

| population | train share | test share | train mean label |
|---|---|---|---|
| ~45 s      | 15% | 53% | 2.93 (SD 0.80) |
| ~60 s      | 67% | 27% | 3.58 (SD 1.03) |
| < 44.5 s   |  7% | 12% | |
| in-between |  7% |  7% | |

Unweighted CV is dominated by 60 s clips while the leaderboard is dominated
by 45 s clips, which the models rank worst (r ≈ 0.72 vs 0.85). This module
provides importance weights w = p_test(pop) / p_train(pop), used to
1. train heads on the test mixture, and
2. score every model on the test mixture (test-weighted CV), which tracks
   the leaderboard much more closely than the unweighted CV.
"""
from __future__ import annotations

import numpy as np

NAMES = ("short", "45s", "mid", "60s")


def population(duration_s) -> np.ndarray:
    """0 = shorter than 44.5 s, 1 = ~45 s, 2 = 45.5–59.5 s, 3 = ~60 s."""
    d = np.asarray(duration_s, float)
    return np.select([d < 44.5, d < 45.5, d < 59.5], [0, 1, 2], 3)


def importance_weights(pop_train: np.ndarray, pop_test: np.ndarray) -> np.ndarray:
    """Per-train-clip weight p_test(pop) / p_train(pop), with mean 1."""
    k = len(NAMES)
    p_tr = np.bincount(pop_train, minlength=k) / len(pop_train)
    p_te = np.bincount(pop_test, minlength=k) / len(pop_test)
    w = (p_te / np.maximum(p_tr, 1e-9))[pop_train]
    return w / w.mean()
