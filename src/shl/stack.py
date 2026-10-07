"""Level-2 stacking, calibration and post-processing.

* **Stacker:** non-negative least squares (NNLS) over out-of-fold (OOF)
  predictions, evaluated with nested grouped CV so its score is not
  optimistic.
* **Calibration:** ``y = b * stack + a_pop``, one shared slope plus one
  intercept per recording population (``shl.population``). The 45 s
  population is systematically over-predicted (labels are lower) and
  dominates the test set. Pearson is affine-invariant, so the least-squares
  fit is RMSE-optimal; variance matching would be worse:
  2*sd^2*(1-r) vs sd^2*(1-r^2).
* **Weights:** ``w_fit`` (importance weights) fits the stacker and the
  calibration on the test mixture. ``w_eval`` scores on the test mixture
  (test-weighted composite), which is what model pruning optimises.
* Optional transductive smoothing within test-speaker clusters, kept only if
  a grouped-OOF simulation shows a clear gain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .folds import iter_folds
from .metrics import composite

N_POP = 4


def nnls_weights(P: np.ndarray, y: np.ndarray, w=None, intercept: bool = True) -> tuple[np.ndarray, float]:
    """Non-negative weights over the columns of P (n, k), plus a free intercept."""
    sw = np.ones(len(y)) if w is None else np.sqrt(np.asarray(w, float))
    if intercept:
        # centre so the intercept is unconstrained while weights stay >= 0
        ww = sw ** 2
        Pm, ym = np.average(P, axis=0, weights=ww), np.average(y, weights=ww)
        coef, _ = nnls((P - Pm) * sw[:, None], (y - ym) * sw)
        return coef, float(ym - Pm @ coef)
    coef, _ = nnls(P * sw[:, None], y * sw)
    return coef, 0.0


def _calib_design(raw: np.ndarray, pop: np.ndarray | None) -> np.ndarray:
    if pop is None:
        return np.column_stack([raw, np.ones_like(raw)])
    return np.column_stack([raw, np.eye(N_POP)[pop]])


def fit_calibration(raw, y, pop=None, w=None) -> np.ndarray:
    """Least-squares (weighted) fit of y ~ b*raw + intercept(s)."""
    X = _calib_design(raw, pop)
    sw = np.ones(len(y)) if w is None else np.sqrt(np.asarray(w, float))
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    if pop is not None:
        # populations absent from the training rows keep the mean intercept
        present = np.bincount(pop, minlength=N_POP) > 0
        beta[1:][~present] = beta[1:][present].mean()
    return beta


def apply_calibration(beta, raw, pop=None):
    return _calib_design(raw, pop) @ beta


def ols_calibration(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (a, b) minimising ||y - (a + b * pred)||^2."""
    b, a = np.polyfit(pred, y, 1)
    return float(a), float(b)


def nested_stack(P: np.ndarray, y: np.ndarray, folds: pd.DataFrame, pop=None,
                 w_fit=None, w_eval=None, clip=(1.0, 5.0)) -> dict:
    """Score the stacker honestly: weights and calibration refitted per fold.

    Returns the OOF predictions, the unweighted composite and, with
    ``w_eval``, the test-weighted composite (``composite_tw``).
    """
    oofs = []
    for col in folds.columns:
        oof = np.zeros(len(y))
        for tr, va in iter_folds(folds, col):
            wt = None if w_fit is None else w_fit[tr]
            coef, c = nnls_weights(P[tr], y[tr], wt)
            raw = P @ coef + c
            beta = fit_calibration(raw[tr], y[tr], None if pop is None else pop[tr], wt)
            oof[va] = np.clip(apply_calibration(beta, raw[va], None if pop is None else pop[va]), *clip)
        oofs.append(oof)
    oof = np.mean(oofs, axis=0)
    out = {"oof": oof, "composite": composite(y, oof),
           "composite_per_seed": [composite(y, o) for o in oofs]}
    if w_eval is not None:
        out["composite_tw"] = composite(y, oof, w_eval)
        out["composite_tw_per_seed"] = [composite(y, o, w_eval) for o in oofs]
    return out


def fit_stack(P: np.ndarray, y: np.ndarray, names: list[str], pop=None, w_fit=None) -> dict:
    """Final stacker on all OOF predictions (applied to the test predictions)."""
    coef, c = nnls_weights(P, y, w_fit)
    beta = fit_calibration(P @ coef + c, y, pop, w_fit)
    return {"weights": pd.Series(coef, index=names), "intercept": c, "beta": beta, "pop": pop is not None}


def apply_stack(model: dict, P_test: np.ndarray, pop_test=None, clip=(1.0, 5.0)) -> np.ndarray:
    raw = P_test @ model["weights"].values + model["intercept"]
    return np.clip(apply_calibration(model["beta"], raw, pop_test if model["pop"] else None), *clip)


def greedy_prune(P: np.ndarray, y: np.ndarray, folds: pd.DataFrame, names: list[str],
                 max_models: int = 15, tol: float = 1e-4, pop=None, w_fit=None, w_eval=None) -> list[str]:
    """Backward elimination of level-1 models on nested-CV composite.

    Uses the test-weighted composite when ``w_eval`` is given. Drops a model
    whenever removing it does not hurt by more than ``tol``, and keeps at
    most ``max_models``. Fewer inputs means a stabler stacker.
    """
    key = "composite_tw" if w_eval is not None else "composite"
    score = lambda cols: nested_stack(P[:, cols], y, folds, pop, w_fit, w_eval)[key]
    keep = list(range(P.shape[1]))
    best = score(keep)
    improved = True
    while improved and len(keep) > 1:
        improved = False
        s, j = min((score([k for k in keep if k != j]), j) for j in keep)
        if s <= best + tol or len(keep) > max_models:
            keep.remove(j)
            best = min(best, s)
            improved = True
    return [names[k] for k in keep]


def cluster_smooth(pred: np.ndarray, clusters: np.ndarray, lam: float) -> np.ndarray:
    """Shrink each prediction toward its speaker-cluster mean by ``lam``."""
    s = pd.Series(pred)
    means = s.groupby(clusters).transform("mean").values
    return (1 - lam) * pred + lam * means


def speaker_label_prior(pred: np.ndarray, sim_to_train: np.ndarray, nn_label: np.ndarray,
                        threshold: float, mu: float) -> np.ndarray:
    """Blend predictions toward the label of the most similar *train* voice.

    Same-voice train clips share the exact label 67% of the time (notebook
    00), so for a clip whose nearest train clip is above ``threshold`` cosine
    similarity, pred <- (1 - mu) * pred + mu * label_of_that_clip. This uses
    train labels and test audio only, never test labels.
    """
    out = pred.copy()
    m = sim_to_train > threshold
    out[m] = (1 - mu) * pred[m] + mu * nn_label[m]
    return out


def simulate_label_prior(oof, y, emb, thresholds=(0.90, 0.93, 0.95, 0.97), mus=(0.25, 0.5, 0.75), w=None):
    """Score the prior on OOF predictions: each train clip's nearest *other*
    train clip plays the role of the seen speaker. Grouped folds keep likely
    same-speaker clips together, so ``oof`` was produced without that clip
    in training, which matches a test clip whose speaker was in train."""
    S = emb @ emb.T
    np.fill_diagonal(S, -1)
    nn, sim = S.argmax(1), S.max(1)
    rows = [{"threshold": None, "mu": 0.0, "composite": composite(y, oof, w), "affected": 0.0}]
    for t in thresholds:
        for mu in mus:
            rows.append({"threshold": t, "mu": mu, "affected": float((sim > t).mean()),
                         "composite": composite(y, speaker_label_prior(oof, sim, y[nn], t, mu), w)})
    return pd.DataFrame(rows)


def simulate_cluster_smoothing(oof: np.ndarray, y: np.ndarray, clusters: np.ndarray,
                               lams=np.linspace(0, 1, 11), w=None) -> pd.DataFrame:
    """Score cluster smoothing on grouped-OOF predictions.

    With grouped folds, all clips of a pseudo-speaker are predicted by the
    same model that never saw that speaker, which is exactly the situation
    for unseen test speakers.
    """
    return pd.DataFrame([
        {"lambda": lam, "composite": composite(y, cluster_smooth(oof, clusters, lam), w)}
        for lam in lams
    ])
