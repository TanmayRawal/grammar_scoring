"""Per-view regression heads evaluated with speaker-grouped cross-validation.

Every head is tuned with an inner GroupKFold on the outer-training speakers
only. Hyperparameters picked with random or LOO-style CV would let the
model memorise voices and look better than it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .folds import iter_folds
from .metrics import composite, report


# --------------------------------------------------------------------------
# Model definitions: (pipeline, grid). The last pipeline step is named "m".
# --------------------------------------------------------------------------
def ridge_head():
    pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge())])
    return pipe, {"m__alpha": np.logspace(0, 5, 11)}


def svr_head(n_pca: int = 128):
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=0)),  # no whitening: it inflates noise components
        ("m", SVR(kernel="rbf")),
    ])
    return pipe, {"m__C": [0.3, 1, 3, 10], "m__epsilon": [0.05, 0.2],
                  "m__gamma": ["scale"]}


def krr_head():
    pipe = Pipeline([("sc", StandardScaler()), ("m", KernelRidge(kernel="rbf"))])
    return pipe, {"m__alpha": [0.1, 0.3, 1, 3], "m__gamma": [1e-4, 3e-4, 1e-3]}


def lgbm_head():
    """Small, heavily regularised gradient boosting for the ~30 handcrafted
    transcript features (non-linear interactions, robust to scale)."""
    from lightgbm import LGBMRegressor

    pipe = Pipeline([("m", LGBMRegressor(n_estimators=300, learning_rate=0.03, subsample=0.8,
                                         subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                                         reg_lambda=1.0, verbose=-1, random_state=0,
                                         n_jobs=2))])  # grid search already runs fits in parallel
    return pipe, {"m__num_leaves": [4, 8], "m__n_estimators": [200, 400]}


HEADS = {"ridge": ridge_head, "svr": svr_head, "krr": krr_head, "lgbm": lgbm_head}


def sample_weights(y: np.ndarray, power: float = 0.5, n_bins: int = 8) -> np.ndarray:
    """Mild inverse-frequency weights so rare extreme scores count more."""
    bins = pd.cut(y, bins=n_bins, labels=False)
    freq = pd.Series(bins).map(pd.Series(bins).value_counts()).values
    w = (1.0 / freq) ** power
    return w / w.mean()


@dataclass
class CVResult:
    name: str
    oof: np.ndarray                 # mean OOF prediction over fold seeds
    test: np.ndarray | None         # mean over all fold models
    oof_per_seed: np.ndarray        # (n_seeds, n)
    best_params: list = field(default_factory=list)
    oof_short: np.ndarray | None = None          # OOF on 40-50 s crops of held-out
    oof_short_per_seed: np.ndarray | None = None  # clips (test-like), mean per clip
    insample: np.ndarray | None = None  # mean of fold models on ALL train clips
                                        # (each clip seen in training by most
                                        # of them); only for the required
                                        # "training RMSE", never for selection

    def summary(self, y, w=None) -> dict:
        """Metrics of the OOF predictions; ``w`` adds test-weighted (``*_tw``) metrics."""
        r = report(y, self.oof, self.name, w=w)
        r["composite_sd_over_seeds"] = float(np.std([composite(y, o, w) for o in self.oof_per_seed]))
        if self.oof_short is not None:
            s = report(y, self.oof_short, ci=False)
            r.update(rmse_short=s["rmse"], pearson_short=s["pearson"], composite_short=s["composite"])
        return r


def _short_means(pred_crops: np.ndarray, src: np.ndarray, n: int) -> np.ndarray:
    """Average crop predictions per source clip (NaN where a clip has none)."""
    tot, cnt = np.zeros(n), np.zeros(n)
    np.add.at(tot, src, pred_crops)
    np.add.at(cnt, src, 1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def run_cv(head: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
           folds: pd.DataFrame, X_test: np.ndarray | None = None,
           weights: np.ndarray | None = None, name: str | None = None,
           inner_splits: int = 3, verbose: bool = False,
           X_short: np.ndarray | None = None, short_index: np.ndarray | None = None,
           X_aug: np.ndarray | None = None, aug_index: np.ndarray | None = None,
           **head_kw) -> CVResult:
    """Outer: repeated speaker-grouped folds. Inner: GroupKFold grid search.

    ``X_short`` / ``short_index``: crop features and their source row. Crops
    of held-out clips are predicted to give a test-like (short-clip) OOF.
    """
    X = np.asarray(X, np.float32)
    oofs, tests, params, ins, shorts = [], [], [], [], []
    for col in folds.columns:
        oof = np.zeros(len(y))
        short = np.full(len(y), np.nan)
        for tr, va in iter_folds(folds, col):
            pipe, grid = HEADS[head](**head_kw)
            if "pca__n_components" not in grid and "pca" in pipe.named_steps:
                pipe.set_params(pca__n_components=min(
                    pipe.named_steps["pca"].n_components, len(tr) // 2, X.shape[1]))
            gs = GridSearchCV(pipe, grid, cv=GroupKFold(inner_splits),
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
            Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
            wtr = None if weights is None else weights[tr]
            if X_aug is not None:              # crops of training clips only
                keep = np.isin(aug_index, tr)
                src = aug_index[keep]
                Xtr = np.vstack([Xtr, np.asarray(X_aug[keep], np.float32)])
                ytr, gtr = np.r_[ytr, y[src]], np.r_[gtr, groups[src]]
                if wtr is not None:
                    wtr = np.r_[wtr, weights[src]]
            fit_kw = {"groups": gtr}
            if wtr is not None:
                fit_kw["m__sample_weight"] = wtr
            gs.fit(Xtr, ytr, **fit_kw)
            oof[va] = gs.predict(X[va])
            ins.append(gs.predict(X))
            if X_short is not None:
                m = np.isin(short_index, va)
                sm = _short_means(gs.predict(np.asarray(X_short[m], np.float32)), short_index[m], len(y))
                short[va] = sm[va]
            params.append(gs.best_params_)
            if X_test is not None:
                tests.append(gs.predict(np.asarray(X_test, np.float32)))
        oofs.append(oof)
        shorts.append(short)
        if verbose:
            print(f"  {col}: composite={composite(y, oof):.4f}")
    oofs = np.vstack(oofs)
    shorts = np.vstack(shorts) if X_short is not None else None
    return CVResult(
        name=name or head,
        oof=oofs.mean(0),
        test=np.mean(tests, axis=0) if tests else None,
        oof_per_seed=oofs,
        best_params=params,
        oof_short=None if shorts is None else shorts.mean(0),
        oof_short_per_seed=shorts,
        insample=np.mean(ins, axis=0),
    )


# --------------------------------------------------------------------------
# Fast ridge: one SVD per training set gives predictions for every alpha.
# With d (1-2.5k features) > n (~585 clips) this is ~50x faster than a
# GridSearchCV over alphas, which makes per-layer sweeps practical.
# --------------------------------------------------------------------------
ALPHAS = np.logspace(-1, 5, 25)


def _standardize(X_tr, *others):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    return [(X_tr - mu) / sd] + [(o - mu) / sd for o in others]


def ridge_path(X_tr, y_tr, X_ev, alphas=ALPHAS, w=None) -> np.ndarray:
    """Ridge predictions on ``X_ev`` for every alpha: shape (n_alphas, n_ev).

    Features are standardised on ``X_tr``; the intercept is the (weighted)
    target mean.
    """
    X_tr, X_ev = _standardize(np.asarray(X_tr, np.float64), np.asarray(X_ev, np.float64))
    w = np.ones(len(y_tr)) if w is None else np.asarray(w, float)
    sw = np.sqrt(w / w.mean())
    y_mu = np.average(y_tr, weights=w)
    # centre features with the same (weighted) mean as the target, so the
    # intercept is exact for weighted fits too (matches sklearn's Ridge)
    x_mu = np.average(X_tr, axis=0, weights=w)
    X_tr, X_ev = X_tr - x_mu, X_ev - x_mu
    A = X_tr * sw[:, None]
    yc = (y_tr - y_mu) * sw
    alphas = np.asarray(alphas)[:, None]
    if A.shape[0] < A.shape[1]:
        # Dual form (n < d): eigendecompose the n x n Gram matrix instead of an
        # SVD of the n x d matrix. Same predictions, ~3x faster here:
        # y_ev = K_ev (K + a I)^-1 y with K = A A^T = U diag(lam) U^T.
        lam, U = np.linalg.eigh(A @ A.T)
        lam = np.clip(lam, 0, None)
        Uty = U.T @ yc                                   # (n,)
        K_ev = X_ev @ A.T                                # (n_ev, n)
        return y_mu + (Uty / (lam + alphas)) @ (K_ev @ U).T
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Uty = U.T @ yc
    proj = X_ev @ Vt.T                                   # (n_ev, r)
    coef = (s / (s ** 2 + alphas)) * Uty                 # (n_alphas, r)
    return y_mu + coef @ proj.T


def fast_ridge_fit_predict(X_tr, y_tr, g_tr, X_out, w=None, inner_splits=3,
                           alphas=ALPHAS) -> tuple[np.ndarray, float]:
    """Pick alpha by inner GroupKFold RMSE, refit on all of X_tr, predict X_out."""
    sse = np.zeros(len(alphas))
    for itr, iva in GroupKFold(inner_splits).split(X_tr, y_tr, g_tr):
        P = ridge_path(X_tr[itr], y_tr[itr], X_tr[iva], alphas, None if w is None else w[itr])
        sse += ((P - y_tr[iva]) ** 2).sum(1)
    best = float(alphas[int(np.argmin(sse))])
    return ridge_path(X_tr, y_tr, X_out, [best], w)[0], best


def run_fast_ridge(X, y, groups, folds, X_test=None, weights=None, name="ridge",
                   X_aug=None, aug_index=None, X_short=None, short_index=None) -> CVResult:
    """Repeated grouped-CV ridge with optional crop augmentation.

    ``X_aug`` holds extra rows (e.g. 40-50 s crops); ``aug_index[i]`` is the
    row of ``X`` the crop came from, so it inherits that clip's label, group,
    weight and fold. Augmented rows only ever join the training side.
    ``X_short`` / ``short_index``: crops scored for the test-like short-clip
    OOF (may be the same arrays as ``X_aug``).
    """
    X = np.asarray(X, np.float32)
    oofs, tests, alphas, ins, shorts = [], [], [], [], []
    for col in folds.columns:
        oof = np.zeros(len(y))
        short = np.full(len(y), np.nan)
        for tr, va in iter_folds(folds, col):
            Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
            wtr = None if weights is None else weights[tr]
            if X_aug is not None:
                keep = np.isin(aug_index, tr)
                src = aug_index[keep]
                Xtr = np.vstack([Xtr, X_aug[keep]])
                ytr, gtr = np.r_[ytr, y[src]], np.r_[gtr, groups[src]]
                if wtr is not None:
                    wtr = np.r_[wtr, weights[src]]
            outs = [X[va], X]
            if X_short is not None:
                sm_mask = np.isin(short_index, va)
                outs.append(np.asarray(X_short[sm_mask], np.float32))
            if X_test is not None:
                outs.append(np.asarray(X_test, np.float32))
            pred, a = fast_ridge_fit_predict(Xtr, ytr, gtr, np.vstack(outs), wtr)
            parts = np.split(pred, np.cumsum([len(o) for o in outs])[:-1])
            oof[va] = parts[0]
            ins.append(parts[1])
            if X_short is not None:
                short[va] = _short_means(parts[2], short_index[sm_mask], len(y))[va]
            if X_test is not None:
                tests.append(parts[-1])
            alphas.append(a)
        oofs.append(oof)
        shorts.append(short)
    oofs = np.vstack(oofs)
    shorts = np.vstack(shorts) if X_short is not None else None
    return CVResult(name=name, oof=oofs.mean(0),
                    test=np.mean(tests, axis=0) if tests else None,
                    oof_per_seed=oofs, best_params=alphas,
                    oof_short=None if shorts is None else shorts.mean(0),
                    oof_short_per_seed=shorts,
                    insample=np.mean(ins, axis=0))


def layer_profile(layer_feats: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  folds: pd.DataFrame) -> pd.DataFrame:
    """Grouped-CV ridge score for each layer separately.

    ``layer_feats`` has shape (n_clips, n_layers, d). Uses the first fold
    seed only, to keep the sweep cheap; the result picks the layer bands.
    """
    one_seed = folds.iloc[:, :1]
    rows = []
    for layer in range(layer_feats.shape[1]):
        res = run_fast_ridge(np.asarray(layer_feats[:, layer, :], np.float32), y, groups, one_seed)
        r = report(y, res.oof, f"L{layer}", ci=False)
        r["layer"] = layer
        rows.append(r)
    return pd.DataFrame(rows)


def bands_for(profile: pd.DataFrame, width: int = 4) -> list[tuple[int, int]]:
    """Best contiguous band of ``width`` layers (A) and the best non-overlapping
    one (B), by mean composite. Half-open (start, end) pairs."""
    c = profile.sort_values("layer").composite.values
    means = np.convolve(c, np.ones(width) / width, mode="valid")
    a = int(np.argmin(means))
    masked = means.copy()
    masked[max(0, a - width + 1): a + width] = np.inf
    b = int(np.argmin(masked))
    return [(a, a + width), (b, b + width)]


def best_band(profile: pd.DataFrame, width: int = 4) -> tuple[int, int]:
    """Contiguous band of ``width`` layers with the lowest mean composite."""
    c = profile.sort_values("layer").composite.values
    width = min(width, len(c))
    means = np.convolve(c, np.ones(width) / width, mode="valid")
    start = int(np.argmin(means))
    return start, start + width  # half-open [start, end)
