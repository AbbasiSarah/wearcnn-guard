"""Train the final WearCNN-Guard model for the browser demo and export it to JSON.

Train (80%) / calibrate (20%) split of all people, 3-seed ensemble, class-conditional
(Mondrian) conformal calibration at alpha = 0.1. Output: demo/model.json
"""
import sys, json, os
sys.path.insert(0, "src")
import numpy as np
from wearcnn_guard.model import WeekCNN
from wearcnn_guard.cnn import Conv1D, Dense
from wearcnn_guard import conformal as cp

d = np.load("data/nhanes_week.npz"); inp = np.load("data/inputs.npz")
C = inp["C"]; y = (d["phq9"] >= 10).astype(int)
rng = np.random.default_rng(42); idx = rng.permutation(len(y)); n_tr = int(0.8 * len(y))
tr, cal = idx[:n_tr], idx[n_tr:]
models = [WeekCNN(epochs=25, batch_size=64, lr=2e-3, dropout=0.3, seed=s).fit(C[tr], y[tr]) for s in (0, 1, 2)]
p_cal = np.mean([m.predict_proba(C[cal]) for m in models], 0)
q = cp.calibrate(p_cal, y[cal], alpha=0.1, mondrian=True)
ev = cp.evaluate(cp.predict_sets(p_cal, q), y[cal])
print("qhat", q, {k: round(float(v), 3) for k, v in ev.items()})

def export(m):
    layers = []
    for L in m.layers:
        if isinstance(L, Conv1D):
            layers.append({"type": "conv", "k": L.k, "W": L.params["W"].round(6).tolist(), "b": L.params["b"].round(6).tolist()})
        elif isinstance(L, Dense):
            layers.append({"type": "dense", "W": L.params["W"].round(6).tolist(), "b": L.params["b"].round(6).tolist()})
        else:
            name = type(L).__name__
            if name == "MaxPool1D": layers.append({"type": "maxpool", "size": L.size})
            elif name == "ReLU": layers.append({"type": "relu"})
            elif name == "GlobalAvgPool": layers.append({"type": "gap"})
    return {"mu": m.mu.ravel().tolist(), "sd": m.sd.ravel().tolist(), "layers": layers}

os.makedirs("demo", exist_ok=True)
json.dump({"alpha": 0.1, "qhat": [float(q[0]), float(q[1])], "n_train": int(n_tr), "n_cal": int(len(cal)),
           "calibration": {k: float(v) for k, v in ev.items()}, "models": [export(m) for m in models]},
          open("demo/model.json", "w"))

# reference outputs for checking the JavaScript port
test = C[cal[:5]]
json.dump({"inputs": test.round(6).tolist(), "probs": np.mean([m.predict_proba(test) for m in models], 0).tolist()},
          open("demo/check.json", "w"))
# pick two sample people from the calibration set: one confident "likely depressed", one confident "not depressed"
sets = cp.predict_sets(p_cal, q)
dep = [i for i in range(len(cal)) if sets[i, 1] and not sets[i, 0] and y[cal[i]] == 1]
nod = [i for i in range(len(cal)) if sets[i, 0] and not sets[i, 1] and y[cal[i]] == 0]
unsure = [i for i in range(len(cal)) if sets[i].all()]
print("samples", len(dep), len(nod), len(unsure))
json.dump({"likely_depressed": int(d["ids"][cal[dep[0]]]) if dep else None,
           "likely_not": int(d["ids"][cal[nod[0]]]) if nod else None,
           "unsure": int(d["ids"][cal[unsure[0]]]) if unsure else None}, open("demo/sample_ids.json", "w"))
