"""Split conformal prediction for binary screening (with optional per-class guarantee).

Theorem (split conformal, Vovk et al. 2005): if calibration and test people are
exchangeable, the set C(x) = {y : s(x, y) <= q_hat} satisfies P(Y in C(X)) >= 1 - alpha,
where s(x, y) = 1 - p_hat(y | x) and q_hat is the ceil((n+1)(1-alpha))/n quantile of the
calibration scores. "Mondrian" (class-conditional) calibration computes q_hat separately
for each true class, giving P(Y in C(X) | Y = y) >= 1 - alpha for every class y, so the
guarantee also holds for the depressed minority, not just on average.
"""
import numpy as np


def _qhat(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]


def calibrate(p_cal, y_cal, alpha=0.1, mondrian=True):
    probs = np.stack([1 - p_cal, p_cal], 1)
    s = 1 - probs[np.arange(len(y_cal)), y_cal.astype(int)]
    if not mondrian:
        q = _qhat(s, alpha); return np.array([q, q])
    return np.array([_qhat(s[y_cal == c], alpha) for c in (0, 1)])


def predict_sets(p, qhat):
    probs = np.stack([1 - p, p], 1)
    return (1 - probs) <= qhat[None, :]          # (n, 2) boolean: label c in the set?


def evaluate(sets, y):
    y = y.astype(int); cov = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    return {"coverage": cov.mean(), "coverage_depressed": cov[y == 1].mean(), "coverage_not_depressed": cov[y == 0].mean(),
            "avg_set_size": size.mean(), "share_single": (size == 1).mean(), "share_both": (size == 2).mean(),
            "share_empty": (size == 0).mean()}
