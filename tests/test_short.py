"""Short-clip (test-like) OOF: crops of held-out clips are scored by models
that never saw the clip, for both the fast ridge and the sklearn heads."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shl import folds, heads, metrics  # noqa: E402
from test_core import make_data  # noqa: E402


def test_short_oof():
    X, y, spk, _ = make_data()
    f = folds.make_folds(y, spk, seeds=(0, 1))
    rng = np.random.default_rng(0)
    src = np.repeat(np.arange(len(y)), 2)
    X_crop = X[src] + rng.normal(0, 0.05, (len(src), X.shape[1]))
    for r in (heads.run_fast_ridge(X, y, spk, f, X_test=X[:4], X_short=X_crop, short_index=src),
              heads.run_cv("ridge", X, y, spk, f, X_test=X[:4], X_short=X_crop, short_index=src)):
        assert r.oof_short is not None and np.isfinite(r.oof_short).all(), r.name
        assert r.oof_short_per_seed.shape == (2, len(y))
        # crops are near-copies of their clip, so short OOF ~ full OOF
        assert abs(metrics.composite(y, r.oof_short) - metrics.composite(y, r.oof)) < 0.03, r.name
        assert r.test.shape == (4,)
        s = r.summary(y)
        assert "composite_short" in s


def test_short_oof_is_out_of_fold():
    # target = clip identity noise that only memorisation could predict:
    # a leak would make short OOF far better than chance
    rng = np.random.default_rng(1)
    n = 200
    spk = np.arange(n)
    X = rng.normal(size=(n, 50))
    y = rng.normal(size=n)
    f = folds.make_folds(y, spk, seeds=(0,))
    src = np.arange(n)
    r = heads.run_fast_ridge(X, y, spk, f, X_short=X.copy(), short_index=src)
    assert metrics.pearson(y, r.oof_short) < 0.3


if __name__ == "__main__":
    test_short_oof(); test_short_oof_is_out_of_fold()
    print("short tests passed")
