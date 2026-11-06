"""A small 1-D Convolutional Neural Network written from scratch in NumPy.

No deep-learning framework is used: every layer implements its own forward
and backward pass, so each step of the algorithm is visible and can be
explained (and gradient-checked, see ``tests/test_cnn.py``).

Architecture (input = one day, 288 five-minute activity values)::

    Conv1D(1 -> 16, k=9) -> ReLU -> MaxPool(4)      288 -> 72
    Conv1D(16 -> 32, k=5) -> ReLU -> MaxPool(4)     72  -> 18
    Conv1D(32 -> 32, k=3) -> ReLU
    GlobalAveragePooling                            -> 32 features
    Dropout(0.3) -> Dense(32 -> 1)                  -> logit / score
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# --------------------------------------------------------------------- layers
class Layer:
    params: dict
    grads: dict

    def __init__(self):
        self.params, self.grads = {}, {}

    def forward(self, x, train=True):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


class Conv1D(Layer):
    """'Same'-padded 1-D convolution implemented with im2col."""

    def __init__(self, in_ch, out_ch, kernel, rng):
        super().__init__()
        self.k = kernel
        fan_in = in_ch * kernel
        self.params["W"] = rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, out_ch))  # He init
        self.params["b"] = np.zeros(out_ch)

    def forward(self, x, train=True):
        n, c, length = x.shape
        left = (self.k - 1) // 2
        xp = np.pad(x, ((0, 0), (0, 0), (left, self.k - 1 - left)))
        win = sliding_window_view(xp, self.k, axis=2)            # (n, c, L, k)
        cols = win.transpose(0, 2, 1, 3).reshape(n, length, c * self.k)
        self.cache = (cols, x.shape, left)
        out = cols @ self.params["W"] + self.params["b"]         # (n, L, out)
        return out.transpose(0, 2, 1)

    def backward(self, dout):
        cols, (n, c, length), left = self.cache
        d = dout.transpose(0, 2, 1)                              # (n, L, out)
        self.grads["W"] = cols.reshape(-1, cols.shape[2]).T @ d.reshape(-1, d.shape[2])
        self.grads["b"] = d.sum(axis=(0, 1))
        dcols = (d @ self.params["W"].T).reshape(n, length, c, self.k)
        dxp = np.zeros((n, c, length + self.k - 1))
        for j in range(self.k):
            dxp[:, :, j:j + length] += dcols[:, :, :, j].transpose(0, 2, 1)
        return dxp[:, :, left:left + length]


class ReLU(Layer):
    def forward(self, x, train=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool1D(Layer):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x, train=True):
        n, c, length = x.shape
        xr = x[:, :, : length // self.size * self.size].reshape(n, c, -1, self.size)
        out = xr.max(axis=3)
        self.mask = xr == out[..., None]
        self.mask = self.mask / self.mask.sum(axis=3, keepdims=True)  # split ties
        self.shape = x.shape
        return out

    def backward(self, dout):
        n, c, length = self.shape
        dx = np.zeros(self.shape)
        g = (self.mask * dout[..., None]).reshape(n, c, -1)
        dx[:, :, : g.shape[2]] = g
        return dx


class GlobalAvgPool(Layer):
    def forward(self, x, train=True):
        self.length = x.shape[2]
        return x.mean(axis=2)

    def backward(self, dout):
        return np.repeat(dout[:, :, None], self.length, axis=2) / self.length


class Dropout(Layer):
    def __init__(self, p, rng):
        super().__init__()
        self.p, self.rng = p, rng

    def forward(self, x, train=True):
        if not train or self.p == 0:
            self.mask = np.ones_like(x)
            return x
        self.mask = (self.rng.random(x.shape) > self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Dense(Layer):
    def __init__(self, n_in, n_out, rng):
        super().__init__()
        self.params["W"] = rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_out))
        self.params["b"] = np.zeros(n_out)

    def forward(self, x, train=True):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"] = self.x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T


# ---------------------------------------------------------------------- model
class Adam:
    def __init__(self, layers, lr=1e-3, b1=0.9, b2=0.999, weight_decay=1e-4):
        self.layers, self.lr, self.b1, self.b2, self.wd = layers, lr, b1, b2, weight_decay
        self.m = [{k: np.zeros_like(v) for k, v in l.params.items()} for l in layers]
        self.v = [{k: np.zeros_like(v) for k, v in l.params.items()} for l in layers]
        self.t = 0

    def step(self):
        self.t += 1
        for layer, m, v in zip(self.layers, self.m, self.v):
            for k in layer.params:
                g = layer.grads[k] + (self.wd * layer.params[k] if k == "W" else 0)
                m[k] = self.b1 * m[k] + (1 - self.b1) * g
                v[k] = self.b2 * v[k] + (1 - self.b2) * g * g
                mh = m[k] / (1 - self.b1 ** self.t)
                vh = v[k] / (1 - self.b2 ** self.t)
                layer.params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class ActivityCNN:
    """1-D CNN for one day of activity.

    task="classification": binary cross-entropy, ``predict_proba`` returns P(depressed)
    task="regression":     mean squared error on a standardised target
    """

    def __init__(self, task="classification", epochs=40, batch_size=32, lr=2e-3,
                 dropout=0.3, seed=0, verbose=False):
        self.task, self.epochs, self.batch_size, self.lr = task, epochs, batch_size, lr
        self.dropout, self.seed, self.verbose = dropout, seed, verbose

    def _build(self):
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        self.layers = [
            Conv1D(1, 16, 9, rng), ReLU(), MaxPool1D(4),
            Conv1D(16, 32, 5, rng), ReLU(), MaxPool1D(4),
            Conv1D(32, 32, 3, rng), ReLU(),
            GlobalAvgPool(), Dropout(self.dropout, rng), Dense(32, 1, rng),
        ]

    def _fit_normalizer(self, X):
        self.mu, self.sd = X.mean(), X.std() + 1e-8

    def _normalize(self, X):
        """(n, L) raw input -> (n, 1, L) standardised tensor."""
        return ((X - self.mu) / self.sd)[:, None, :]

    def _forward(self, x, train):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x[:, 0]

    def _backward(self, dout):
        d = dout[:, None]
        for layer in reversed(self.layers):
            d = layer.backward(d)

    def _loss_grad(self, out, y, w):
        if self.task == "classification":
            p = _sigmoid(out)
            loss = -np.mean(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
            return loss, w * (p - y) / len(y)
        err = out - y
        return np.mean(err ** 2), 2 * err / len(y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self._fit_normalizer(X)
        Xn = self._normalize(X)
        y = np.asarray(y, dtype=np.float64)
        if self.task == "regression":
            self.y_mu, self.y_sd = y.mean(), y.std() + 1e-8
            y = (y - self.y_mu) / self.y_sd
            sample_w = np.ones_like(y)
        else:  # balance the two classes
            pos = y.mean()
            sample_w = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))
        self._build()
        opt = Adam(self.layers, lr=self.lr)
        self.history = []
        for epoch in range(self.epochs):
            idx = self.rng.permutation(len(y))
            total = 0.0
            for s in range(0, len(idx), self.batch_size):
                b = idx[s:s + self.batch_size]
                xb = Xn[b] * self.rng.uniform(0.9, 1.1, (len(b), 1, 1))  # amplitude jitter
                out = self._forward(xb, train=True)
                loss, dout = self._loss_grad(out, y[b], sample_w[b])
                self._backward(dout)
                opt.step()
                total += loss * len(b)
            self.history.append(total / len(y))
            if self.verbose:
                print(f"epoch {epoch + 1:3d}  loss {self.history[-1]:.4f}")
        return self

    def decision_function(self, X, batch=256):
        Xn = self._normalize(np.asarray(X, dtype=np.float64))
        return np.concatenate([self._forward(Xn[i:i + batch], train=False)
                               for i in range(0, len(Xn), batch)])

    def predict_proba(self, X):
        return _sigmoid(self.decision_function(X))

    def predict(self, X):
        out = self.decision_function(X)
        if self.task == "classification":
            return (out > 0).astype(int)
        return out * self.y_sd + self.y_mu


class DiaryCNN(ActivityCNN):
    """Multichannel 1-D CNN over a window of daily self-report questionnaires.

    Input: (n, channels, days), e.g. 11 questionnaire items x 14 days.
    Each channel is standardised separately. Architecture::

        Conv1D(C -> 16, k=3) -> ReLU -> MaxPool(2)     14 -> 7
        Conv1D(16 -> 16, k=3) -> ReLU
        GlobalAveragePooling -> Dropout -> Dense(16 -> 1)
    """

    def __init__(self, in_channels, epochs=60, batch_size=32, lr=2e-3, dropout=0.3,
                 seed=0, verbose=False):
        super().__init__("classification", epochs, batch_size, lr, dropout, seed, verbose)
        self.in_channels = in_channels

    def _build(self):
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        self.layers = [
            Conv1D(self.in_channels, 16, 3, rng), ReLU(), MaxPool1D(2),
            Conv1D(16, 16, 3, rng), ReLU(),
            GlobalAvgPool(), Dropout(self.dropout, rng), Dense(16, 1, rng),
        ]

    def _fit_normalizer(self, X):
        self.mu = X.mean(axis=(0, 2), keepdims=True)
        self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-8

    def _normalize(self, X):
        return (X - self.mu) / self.sd
