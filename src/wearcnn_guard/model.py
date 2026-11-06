"""Multi-channel version of the NumPy 1-D CNN (layers reused from cnn.py)."""
import numpy as np
from .cnn import ActivityCNN, Conv1D, ReLU, MaxPool1D, GlobalAvgPool, Dropout, Dense


class WeekCNN(ActivityCNN):
    """Input (n, C, 144): one channel per summary of the person's week."""

    def __init__(self, in_ch=2, **kw):
        super().__init__(task="classification", **kw)
        self.in_ch = in_ch

    def _build(self):
        rng = np.random.default_rng(self.seed); self.rng = rng
        self.layers = [
            Conv1D(self.in_ch, 16, 7, rng), ReLU(), MaxPool1D(2),   # 144 -> 72
            Conv1D(16, 32, 5, rng), ReLU(), MaxPool1D(2),           # 72 -> 36
            Conv1D(32, 32, 3, rng), ReLU(),
            GlobalAvgPool(), Dropout(self.dropout, rng), Dense(32, 1, rng),
        ]

    def _fit_normalizer(self, X):                 # per-channel standardisation
        self.mu = X.mean(axis=(0, 2), keepdims=True)[0]
        self.sd = X.std(axis=(0, 2), keepdims=True)[0] + 1e-8

    def _normalize(self, X):
        return (X - self.mu) / self.sd
