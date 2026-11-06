"""Inputs for the models: CNN channels and classic circadian-rhythm features."""
import numpy as np


def cnn_input(X, M):
    """(n, 7, 144) weekly log-activity -> (n, 2, 144): mean day and day-to-day SD."""
    w = M[:, :, None].astype(float)
    mean = (X * w).sum(1) / w.sum(1)
    sd = np.sqrt(((X - mean[:, None]) ** 2 * w).sum(1) / w.sum(1))
    return np.stack([mean, sd], 1).astype(np.float32)


def rhythm_features(X, M):
    """Standard non-parametric actigraphy features (per person).

    total: mean log-activity; M10 / L5: most active 10 h / least active 5 h;
    RA: relative amplitude; IS: inter-daily stability; IV: intra-daily variability;
    sedentary: share of 10-min bins with very low activity; peak_hour.
    """
    out = []
    for x, m in zip(X, M):
        d = x[m]                                    # valid days (k, 144)
        hourly = d.reshape(len(d), 24, 6).mean(2)   # (k, 24)
        prof = hourly.mean(0)
        circ = np.concatenate([prof, prof])
        m10 = max(circ[i:i + 10].mean() for i in range(24))
        l5 = min(circ[i:i + 5].mean() for i in range(24))
        series = hourly.ravel(); mu = series.mean()
        is_ = (24 * ((prof - mu) ** 2).sum()) / (((series - mu) ** 2).sum() + 1e-9) / 24
        iv = (len(series) * (np.diff(series) ** 2).sum()) / ((len(series) - 1) * ((series - mu) ** 2).sum() + 1e-9)
        out.append([d.mean(), m10, l5, (m10 - l5) / (m10 + l5 + 1e-9), is_, iv,
                    (d < 1.0).mean(), float(np.argmax(prof)), d.mean(1).std()])
    return np.array(out, np.float32)


FEATURE_NAMES = ["total", "M10", "L5", "RA", "IS", "IV", "sedentary_share", "peak_hour", "daily_sd"]
