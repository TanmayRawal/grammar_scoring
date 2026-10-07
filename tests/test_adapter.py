"""Adapter head on synthetic segment states: learns a planted signal and
returns the same outputs as the other heads (OOF, short OOF, test, in-sample)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shl import adapter, folds, metrics  # noqa: E402


def test_adapter_learns():
    rng = np.random.default_rng(0)
    n_spk, L, T, d = 90, 2, 60, 16
    spk = np.repeat(np.arange(n_spk), 2)
    level = rng.uniform(1, 5, n_spk)[spk]
    lens = rng.integers(45, 61, len(spk))
    S = rng.normal(size=(len(spk), L, T, d)).astype(np.float32)
    S[:, :, :, 0] += (level[:, None, None] - 3)          # signal in every segment
    y = np.clip(np.round(2 * level) / 2, 1, 5)
    f = folds.make_folds(y, spk, seeds=(0,))
    crop_src = np.repeat(np.arange(len(y)), 2)
    crop_win = np.array([(int(rng.integers(0, 10)), 45) for _ in crop_src])
    S_test, lens_test = S[:7], lens[:7]
    cfg = adapter.AdapterConfig(epochs=25, patience=8)
    r = adapter.run_adapter_cv(S, lens, y, spk, f, S_test, lens_test, crop_src, crop_win, cfg, log=lambda *_: None)
    assert r["test"].shape == (7,) and r["insample"].shape == y.shape
    assert np.isfinite(r["oof_short"]).all()
    assert metrics.pearson(y, r["oof"]) > 0.8, metrics.pearson(y, r["oof"])
    assert metrics.pearson(y, r["oof_short"]) > 0.8


if __name__ == "__main__":
    test_adapter_learns()
    print("adapter test passed")
