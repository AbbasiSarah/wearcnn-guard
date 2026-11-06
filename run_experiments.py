"""WearCNN-Guard experiments: depression screening (PHQ-9 >= 10) from one week of
NHANES 2011-2014 wrist activity, with conformal prediction sets.

Each person appears in exactly one of train (60%), calibration (20%) or test (20%).
Repeated over 5 random person-level splits.
"""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from wearcnn_guard.model import WeekCNN
from wearcnn_guard import conformal as cp

d = np.load("data/nhanes_week.npz"); inp = np.load("data/inputs.npz")
C, F = inp["C"], inp["F"]
y = (d["phq9"] >= 10).astype(int)
demo = np.c_[d["age"], d["sex"] == 2].astype(np.float32)
FD = np.c_[F, demo]

def cnn_fit_predict(tr, others, seeds=(0, 1, 2), epochs=25):
    ps = []
    for s in seeds:
        m = WeekCNN(epochs=epochs, batch_size=64, lr=2e-3, dropout=0.3, seed=s).fit(C[tr], y[tr])
        ps.append([m.predict_proba(C[o]) for o in others])
    return [np.mean([p[k] for p in ps], 0) for k in range(len(others))]

def sk(model, Xm):
    def f(tr, others):
        model.fit(Xm[tr], y[tr]); return [model.predict_proba(Xm[o])[:, 1] for o in others]
    return f

MODELS = {
    "LogReg (rhythm features)": lambda: sk(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), F),
    "GradBoost (rhythm features)": lambda: sk(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=0), F),
    "LogReg (rhythm + age, sex)": lambda: sk(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), FD),
    "GradBoost (rhythm + age, sex)": lambda: sk(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=0), FD),
    "1-D CNN (activity only)": lambda: cnn_fit_predict,
}

def sens_spec(p, y, thr):
    pred = p >= thr
    return pred[y == 1].mean(), (~pred[y == 0]).mean()

results, t0 = {k: [] for k in MODELS}, time.time()
keep_preds = {}
for split in range(5):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=split)
    cal, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=split)
    for name, mk in MODELS.items():
        p_cal, p_te = mk()(tr, [cal, te])
        # threshold for sensitivity/specificity chosen on calibration people (Youden)
        ths = np.quantile(p_cal, np.linspace(0.01, 0.99, 197))
        thr = max(ths, key=lambda t: sum(sens_spec(p_cal, y[cal], t)))
        se, spv = sens_spec(p_te, y[te], thr)
        r = {"auc": roc_auc_score(y[te], p_te), "auprc": average_precision_score(y[te], p_te),
             "sensitivity": se, "specificity": spv}
        for alpha in (0.1, 0.2):
            for mondrian in (False, True):
                q = cp.calibrate(p_cal, y[cal], alpha, mondrian)
                ev = cp.evaluate(cp.predict_sets(p_te, q), y[te])
                r.update({f"{'mondrian' if mondrian else 'marginal'}_a{alpha}_{k}": v for k, v in ev.items()})
        results[name].append({k: float(v) for k, v in r.items()})
        if split == 0: keep_preds[name] = (p_cal, p_te, cal, te)
        print(f"split {split} {name:32s} AUC {r['auc']:.3f}  Mondrian90 cov {r['mondrian_a0.1_coverage']:.3f} "
              f"dep {r['mondrian_a0.1_coverage_depressed']:.3f} size {r['mondrian_a0.1_avg_set_size']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

summary = {name: {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))] for k in rs[0]} for name, rs in results.items()}
json.dump({"n_people": int(len(y)), "n_depressed": int(y.sum()), "splits": results, "summary": summary}, open("results/results.json", "w"), indent=1)
np.savez("results/split0_preds.npz", **{f"{k}|{j}": v for k, (a, b, c_, t_) in keep_preds.items() for j, v in zip(["p_cal", "p_te", "cal", "te"], [a, b, c_, t_])})
print("done", round(time.time() - t0), "s")
