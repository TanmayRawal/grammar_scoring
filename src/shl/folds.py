"""Pseudo-speaker grouping and speaker-grouped cross-validation folds.

Speakers repeat in the training set and test speakers are unseen, so random
K-fold leaks voice identity into validation. We cluster clips into
pseudo-speakers from speaker embeddings and keep every cluster inside a
single fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


def prepare_embeddings(emb: np.ndarray, center=None) -> np.ndarray:
    """Mean-centre, then L2-normalise.

    Raw x-vectors share a large common component (median nearest-neighbour
    cosine 0.98 here), so without centring every clip looks alike.
    """
    emb = np.asarray(emb, float)
    center = emb.mean(0) if center is None else center
    return normalize(emb - center)


def cluster_speakers(emb: np.ndarray, distance_threshold: float,
                     linkage: str = "average") -> np.ndarray:
    """Agglomerative cosine clustering of prepared embeddings into pseudo-speakers."""
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage=linkage,
        distance_threshold=distance_threshold,
    )
    return model.fit_predict(emb)


def same_speaker_pairs(emb: np.ndarray, sim_cut: float = 0.93):
    """Index pairs (i, j) whose cosine similarity marks them as likely the same speaker."""
    S = emb @ emb.T
    iu = np.triu_indices(len(emb), 1)
    keep = S[iu] > sim_cut
    return iu[0][keep], iu[1][keep]


def threshold_sweep(emb: np.ndarray, y: np.ndarray, thresholds,
                    sim_cut: float = 0.93) -> pd.DataFrame:
    """Diagnostics for choosing the clustering threshold.

    The criterion that matters for honest CV is **leakage**: the share of
    likely same-speaker pairs (cosine > ``sim_cut``) that end up in
    different groups, and hence potentially in different folds. Larger,
    mixed groups are harmless as long as folds stay balanced.
    """
    a, b = same_speaker_pairs(emb, sim_cut)
    rows = []
    for t in thresholds:
        g = cluster_speakers(emb, t)
        sizes = pd.Series(g).value_counts()
        rows.append({
            "threshold": t,
            "n_groups": int(sizes.size),
            "largest_group": int(sizes.max()),
            "pair_leak": float(np.mean(g[a] != g[b])) if len(a) else 0.0,
        })
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame, max_leak: float = 0.02,
                     max_group_frac: float = 0.06, n: int | None = None) -> float:
    """Smallest-leak threshold whose largest group stays small enough to
    allow balanced 5-fold splits (at most ``max_group_frac`` of clips)."""
    ok = sweep
    if n is not None:
        ok = ok[ok.largest_group <= max_group_frac * n]
    ok_leak = ok[ok.pair_leak <= max_leak]
    pick = ok_leak.iloc[0] if len(ok_leak) else ok.sort_values("pair_leak").iloc[0]
    return float(pick.threshold)


def _greedy_assign(bins: np.ndarray, groups: np.ndarray, n_splits: int,
                   seed: int, label_weight: float = 1.0) -> np.ndarray:
    """Assign whole groups to folds, balancing fold size and label histogram.

    Groups are placed largest first (random tie-break per seed). Each goes
    to the fold that minimises the relative size overshoot plus the L1 gap
    between the fold's label histogram and the global one.
    """
    rng = np.random.default_rng(seed)
    n, n_bins = len(bins), int(bins.max()) + 1
    target_hist = np.bincount(bins, minlength=n_bins) / n
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    order = sorted(uniq, key=lambda g: (-len(members[g]), rng.random()))
    hist = np.zeros((n_splits, n_bins))
    fold = np.full(n, -1, dtype=int)
    for g in order:
        idx = members[g]
        add = np.bincount(bins[idx], minlength=n_bins)
        costs = []
        for k in range(n_splits):
            h = hist[k] + add
            size_cost = h.sum() / (n / n_splits)
            label_cost = np.abs(h / h.sum() - target_hist).sum()
            costs.append(size_cost + label_weight * label_cost + 1e-9 * rng.random())
        k = int(np.argmin(costs))
        hist[k] += add
        fold[idx] = k
    return fold


def make_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
               seeds=(0, 1, 2, 3, 4), n_bins: int = 5) -> pd.DataFrame:
    """Repeated speaker-grouped folds, balanced on size and score.

    Returns one column of fold ids per seed. sklearn's StratifiedGroupKFold
    produced folds of 99-196 clips on this data, so a greedy assigner is
    used instead (see ``_greedy_assign``).
    """
    y = np.asarray(y, float)
    bins = np.asarray(pd.qcut(y, q=n_bins, labels=False, duplicates="drop"), int)
    groups = np.asarray(groups)
    folds = pd.DataFrame({f"fold_s{seed}": _greedy_assign(bins, groups, n_splits, seed)
                          for seed in seeds})
    check_no_group_leak(folds, groups)
    return folds


def check_no_group_leak(folds: pd.DataFrame, groups: np.ndarray) -> None:
    """Assert that every pseudo-speaker lands in exactly one fold per seed."""
    groups = np.asarray(groups)
    for col in folds.columns:
        per_group = pd.Series(folds[col].values).groupby(groups).nunique()
        assert (per_group == 1).all(), f"speaker leakage across folds in {col}"
        assert (folds[col] >= 0).all(), f"unassigned rows in {col}"


def iter_folds(folds: pd.DataFrame, col: str):
    """Yield (train_idx, valid_idx) for one fold column."""
    f = folds[col].values
    for k in np.unique(f):
        yield np.where(f != k)[0], np.where(f == k)[0]
