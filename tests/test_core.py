"""Smoke tests on synthetic data: the folds, heads and stacker must run and
the grouped CV must not leak speakers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shl import folds, heads, metrics, stack  # noqa: E402


def make_data(n_speakers=120, seed=0):
    rng = np.random.default_rng(seed)
    clips_per = rng.integers(1, 4, n_speakers)
    spk = np.repeat(np.arange(n_speakers), clips_per)
    level = rng.uniform(1, 5, n_speakers)[spk]          # speaker-level score
    voice = rng.normal(size=(n_speakers, 16))[spk]      # speaker identity
    X = np.hstack([level[:, None] + rng.normal(0, .5, (len(spk), 1)),
                   voice + rng.normal(0, .1, voice.shape),
                   rng.normal(size=(len(spk), 20))])
    y = np.clip(np.round(2 * (level + rng.normal(0, .2, len(spk)))) / 2, 1, 5)
    return X, y, spk, voice


def test_metrics():
    y = np.array([1, 2, 3, 4, 5.])
    assert metrics.rmse(y, y) == 0 and abs(metrics.pearson(y, y) - 1) < 1e-12
    assert abs(metrics.composite(y, y)) < 1e-12
    assert abs(metrics.composite(y, y + 1) - 0.6) < 1e-12   # 0.6*RMSE + 0.4*(1-r)
    # affine invariance of Pearson, OLS calibration recovers the target
    a, b = stack.ols_calibration(0.5 * y + 1, y)
    assert abs(a + 2) < 1e-9 and abs(b - 2) < 1e-9


def test_speaker_clustering_and_folds():
    X, y, spk, voice = make_data()
    g = folds.cluster_speakers(folds.prepare_embeddings(voice + 0.01), distance_threshold=0.2)
    # clusters should recover the true speakers almost exactly
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(spk, g) > 0.95
    f = folds.make_folds(y, g, seeds=(0, 1))
    folds.check_no_group_leak(f, g)
    assert f.shape == (len(y), 2)


def test_heads_and_stack():
    X, y, spk, _ = make_data()
    f = folds.make_folds(y, spk, seeds=(0,))
    res = [heads.run_cv(h, X, y, spk, f, X_test=X[:10]) for h in ("ridge", "svr")]
    for r in res:
        assert r.test.shape == (10,)
        assert metrics.pearson(y, r.oof) > {"ridge": 0.85, "svr": 0.75}[r.name], r.name
    P = np.column_stack([r.oof for r in res])
    nested = stack.nested_stack(P, y, f)
    # the honest (nested) stack should be no worse than its best input
    assert nested["composite"] <= min(metrics.composite(y, r.oof) for r in res) + 0.01
    model = stack.fit_stack(P, y, [r.name for r in res])
    assert (model["weights"] >= 0).all()
    sim = stack.simulate_cluster_smoothing(nested["oof"], y, spk)
    # true speakers share scores here, so smoothing must help
    assert sim.composite.iloc[-1] <= sim.composite.iloc[0]


def test_fast_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y, spk, _ = make_data()
    ref = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X[:200], y[:200]).predict(X[200:])
    fast = heads.ridge_path(X[:200], y[:200], X[200:], [10.0])[0]
    assert np.allclose(ref, fast, atol=1e-6)
    # weighted fit, and both the primal (n > d) and dual (n < d) code paths
    w = np.random.default_rng(0).uniform(0.5, 2, 200)
    for cols in (slice(None), slice(0, 5)):
        Xc = np.hstack([X[:, cols]] * (1 if cols == slice(0, 5) else 8))
        ref = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
            Xc[:200], y[:200], ridge__sample_weight=w / w.mean()).predict(Xc[200:])
        assert np.allclose(ref, heads.ridge_path(Xc[:200], y[:200], Xc[200:], [10.0], w)[0], atol=1e-5)


def test_fast_ridge_cv_and_crop_aug():
    X, y, spk, _ = make_data()
    f = folds.make_folds(y, spk, seeds=(0,))
    base = heads.run_fast_ridge(X, y, spk, f, X_test=X[:5])
    assert metrics.pearson(y, base.oof) > 0.85 and base.test.shape == (5,)
    # crops = noisy copies of their source clip; they must never leak into the
    # validation fold of their source, so OOF quality stays comparable
    rng = np.random.default_rng(1)
    src = np.repeat(np.arange(len(y)), 2)
    X_aug = X[src] + rng.normal(0, 0.05, (len(src), X.shape[1]))
    aug = heads.run_fast_ridge(X, y, spk, f, X_aug=X_aug, aug_index=src)
    assert abs(metrics.composite(y, aug.oof) - metrics.composite(y, base.oof)) < 0.05


if __name__ == "__main__":
    test_metrics(); test_speaker_clustering_and_folds(); test_heads_and_stack()
    test_fast_ridge_matches_sklearn(); test_fast_ridge_cv_and_crop_aug()
    print("all tests passed")
