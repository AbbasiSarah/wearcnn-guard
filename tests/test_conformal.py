"""Checks that the conformal guarantee holds on simulated data, and that the CNN gradients are correct."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from wearcnn_guard import conformal as cp
from wearcnn_guard.model import WeekCNN


def test_coverage_guarantee_holds_on_average():
    """Over many random calibration/test draws, coverage should be at least 1 - alpha."""
    rng = np.random.default_rng(0)
    covs, covs_pos = [], []
    for _ in range(300):
        y = (rng.random(1200) < 0.1).astype(int)
        p = np.clip(0.1 + 0.25 * y + rng.normal(0, 0.15, len(y)), 0.001, 0.999)  # a weak, noisy model
        q = cp.calibrate(p[:600], y[:600], alpha=0.1, mondrian=True)
        ev = cp.evaluate(cp.predict_sets(p[600:], q), y[600:])
        covs.append(ev["coverage"]); covs_pos.append(ev["coverage_depressed"])
    assert np.mean(covs) >= 0.895
    assert np.mean(covs_pos) >= 0.895      # Mondrian: guarantee also for the minority class


def test_cnn_gradient_matches_numerical_estimate():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 2, 144)); y = np.array([0, 1, 0, 1, 1, 0], float)
    m = WeekCNN(epochs=0, dropout=0.0, seed=0); m._fit_normalizer(X); m._build()
    xn = m._normalize(X)

    def loss():
        out = m._forward(xn, train=False)
        return m._loss_grad(out, y, np.ones_like(y))[0]

    out = m._forward(xn, train=False)
    _, dout = m._loss_grad(out, y, np.ones_like(y))
    m._backward(dout)
    W = m.layers[0].params["W"]; g = m.layers[0].grads["W"]
    for idx in [(0, 0), (3, 5), (10, 15)]:
        old = W[idx]; eps = 1e-5
        W[idx] = old + eps; lp = loss(); W[idx] = old - eps; lm = loss(); W[idx] = old
        num = (lp - lm) / (2 * eps)
        assert abs(num - g[idx]) < 1e-5 + 1e-3 * abs(num)
