"""Test-weighted metrics and population-aware calibration."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shl import metrics, population, stack  # noqa: E402


def test_weighted_metrics():
    rng = np.random.default_rng(0)
    y = rng.uniform(1, 5, 300); p = y + rng.normal(0, .5, 300)
    assert abs(metrics.composite(y, p, np.ones(300)) - metrics.composite(y, p)) < 1e-12
    wi = rng.integers(1, 4, 300)                      # integer weights == duplicated rows
    assert abs(metrics.composite(y, p, wi) - metrics.composite(np.repeat(y, wi), np.repeat(p, wi))) < 1e-12


def test_population_calibration():
    rng = np.random.default_rng(1)
    y = rng.uniform(1, 5, 300); pop = rng.integers(0, 4, 300)
    raw = y + np.array([0.3, -0.2, 0.1, 0.0])[pop]
    beta = stack.fit_calibration(raw, y, pop)
    assert np.allclose(stack.apply_calibration(beta, raw, pop), y, atol=1e-8)


def test_importance_weights():
    w = population.importance_weights(np.array([3, 3, 3, 1]), np.array([1, 1, 3, 0]))
    assert np.allclose(w, [4 / 9, 4 / 9, 4 / 9, 8 / 3]) and abs(w.mean() - 1) < 1e-12
    assert list(population.population([30, 45.0, 50, 60.07])) == [0, 1, 2, 3]


if __name__ == "__main__":
    test_weighted_metrics(); test_population_calibration(); test_importance_weights()
    print("population tests passed")
