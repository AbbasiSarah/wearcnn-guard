"""Additional analyses for the paper (same 5 person-level splits as run_experiments.py).

1. CNN with age and sex added as two extra input channels, and an ensemble with logistic regression.
2. Coverage at 80/90/95% (Mondrian vs marginal) and by subgroup (sex, age group, survey cycle).
3. Conformal sets vs a simple "abstain when the probability is near 0.5" rule at the same abstention rate.
4. Bootstrap 95% confidence intervals (1,000 resamples of test people, within each split).
"""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from wearcnn_guard.model import WeekCNN
from wearcnn_guard import conformal as cp

d = np.load("data/nhanes_week.npz"); inp = np.load("data/inputs.npz")
C, F = inp["C"], inp["F"]
y = (d["phq9"] >= 10).astype(int)
age, female, cycle = d["age"].astype(np.float32), (d["sex"] == 2).astype(np.float32), d["cycle"]
FD = np.c_[F, age, female]
CD = np.concatenate([C, np.repeat(age[:, None, None], C.shape[2], 2), np.repeat(female[:, None, None], C.shape[2], 2)], 1).astype(np.float32)
agegrp = np.digitize(age, [40, 60])                      # 0: 18-39, 1: 40-59, 2: 60+
GROUPS = {"men": female == 0, "women": female == 1, "age 18-39": agegrp == 0, "age 40-59": agegrp == 1,
          "age 60+": agegrp == 2, "2011-12": cycle == 7, "2013-14": cycle == 8}
rng_boot = np.random.default_rng(123)

def cnn(X, in_ch, tr, others, seeds=(0, 1, 2), epochs=25):
    ps = []
    for s in seeds:
        m = WeekCNN(in_ch=in_ch, epochs=epochs, batch_size=64, lr=2e-3, dropout=0.3, seed=s).fit(X[tr], y[tr])
        ps.append([m.predict_proba(X[o]) for o in others])
    return [np.mean([p[k] for p in ps], 0) for k in range(len(others))]

def logreg(tr, others):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(FD[tr], y[tr])
    return [m.predict_proba(FD[o])[:, 1] for o in others]

def band_rule(p_cal, y_cal, p_te, target_abstain):
    """Abstain ('both') when |p - 0.5| < h, with h set on calibration people so the abstention rate matches."""
    h = np.quantile(np.abs(p_cal - 0.5), target_abstain)
    sets = np.zeros((len(p_te), 2), bool)
    unsure = np.abs(p_te - 0.5) < h
    sets[:, 1] = (p_te >= 0.5) | unsure; sets[:, 0] = (p_te < 0.5) | unsure
    return sets

def boot_ci(fn, idx_te, B=1000):
    vals = []
    n = len(idx_te)
    for _ in range(B):
        b = rng_boot.integers(0, n, n)
        try: vals.append(fn(b))
        except ValueError: pass
    return np.percentile(vals, [2.5, 97.5]).tolist()

out = {"cnn_demo": [], "ensemble": [], "cnn": [], "logreg_demo": []}
saved = {}
t0 = time.time()
for split in range(5):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=split)
    cal, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=split)
    preds = {}
    preds["cnn"] = cnn(C, 2, tr, [cal, te])
    preds["cnn_demo"] = cnn(CD, 4, tr, [cal, te])
    preds["logreg_demo"] = logreg(tr, [cal, te])
    preds["ensemble"] = [(a + b) / 2 for a, b in zip(preds["cnn_demo"], preds["logreg_demo"])]
    for name, (pc, pt) in preds.items():
        saved[f"{name}_s{split}_cal"], saved[f"{name}_s{split}_te"] = pc, pt
        saved[f"idx_s{split}_cal"], saved[f"idx_s{split}_te"] = cal, te
        yc, yt = y[cal], y[te]
        r = {"auc": roc_auc_score(yt, pt),
             "auc_ci": boot_ci(lambda b: roc_auc_score(yt[b], pt[b]), te)}
        for alpha in (0.05, 0.1, 0.2):
            for mond in (False, True):
                sets = cp.predict_sets(pt, cp.calibrate(pc, yc, alpha, mond))
                ev = cp.evaluate(sets, yt)
                key = f"{'mondrian' if mond else 'marginal'}_a{alpha}"
                r[key] = {k: float(v) for k, v in ev.items()}
                if alpha == 0.1:
                    covered = sets[np.arange(len(yt)), yt]
                    r[key]["cov_dep_ci"] = boot_ci(lambda b: covered[b][yt[b] == 1].mean(), te)
                    r[key]["cov_ci"] = boot_ci(lambda b: covered[b].mean(), te)
                    if mond:
                        r[key]["subgroups"] = {g: {"coverage": float(covered[m[te]].mean()),
                                                   "coverage_depressed": float(covered[m[te] & (yt == 1)].mean()),
                                                   "n": int(m[te].sum()), "n_dep": int((m[te] & (yt == 1)).sum())}
                                               for g, m in GROUPS.items()}
                        # abstention rate on calibration people, used to tune the band rule
                        sc = cp.predict_sets(pc, cp.calibrate(pc, yc, alpha, mond))
                        band = band_rule(pc, yc, pt, (sc.sum(1) == 2).mean())
                        # class x sex conditional calibration (one threshold per class within each sex)
                        sets_g = np.zeros_like(sets)
                        for g in (0, 1):
                            mc, mt = female[cal] == g, female[te] == g
                            sets_g[mt] = cp.predict_sets(pt[mt], cp.calibrate(pc[mc], yc[mc], alpha, True))
                        cov_g = sets_g[np.arange(len(yt)), yt]
                        r["mondrian_sex_a0.1"] = {k: float(v) for k, v in cp.evaluate(sets_g, yt).items()}
                        r["mondrian_sex_a0.1"]["subgroups"] = {g: {"coverage": float(cov_g[m[te]].mean()),
                            "coverage_depressed": float(cov_g[m[te] & (yt == 1)].mean())} for g, m in GROUPS.items()}
                        r["band_rule_a0.1"] = {k: float(v) for k, v in cp.evaluate(band, yt).items()}
        out[name].append(r)
        print(f"split {split} {name:12s} AUC {r['auc']:.3f} [{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}]  "
              f"M90 dep {r['mondrian_a0.1']['coverage_depressed']:.3f}  band dep {r['band_rule_a0.1']['coverage_depressed']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

def mean(rs, path):
    v = []
    for r in rs:
        x = r
        for p in path: x = x[p]
        v.append(x)
    return np.mean(v, 0).tolist()

summary = {}
for name, rs in out.items():
    s = {"auc": [float(np.mean([r["auc"] for r in rs])), float(np.std([r["auc"] for r in rs]))],
         "auc_ci_mean_bounds": mean(rs, ["auc_ci"])}
    for alpha in (0.05, 0.1, 0.2):
        for k in ("marginal", "mondrian"):
            key = f"{k}_a{alpha}"
            s[key] = {m: float(np.mean([r[key][m] for r in rs])) for m in ("coverage", "coverage_depressed", "coverage_not_depressed", "avg_set_size", "share_single", "share_both", "share_empty")}
    for k in ("marginal", "mondrian"):
        s[f"{k}_a0.1"]["cov_dep_ci_mean_bounds"] = mean(rs, [f"{k}_a0.1", "cov_dep_ci"])
        s[f"{k}_a0.1"]["cov_ci_mean_bounds"] = mean(rs, [f"{k}_a0.1", "cov_ci"])
    s["subgroups_mondrian_a0.1"] = {g: {m: float(np.mean([r["mondrian_a0.1"]["subgroups"][g][m] for r in rs])) for m in ("coverage", "coverage_depressed", "n", "n_dep")} for g in GROUPS}
    s["mondrian_sex_a0.1"] = {m: float(np.mean([r["mondrian_sex_a0.1"][m] for r in rs])) for m in ("coverage", "coverage_depressed", "coverage_not_depressed", "share_single", "share_both")}
    s["mondrian_sex_a0.1"]["subgroups"] = {g: {m: float(np.mean([r["mondrian_sex_a0.1"]["subgroups"][g][m] for r in rs])) for m in ("coverage", "coverage_depressed")} for g in GROUPS}
    s["band_rule_a0.1"] = {m: float(np.mean([r["band_rule_a0.1"][m] for r in rs])) for m in ("coverage", "coverage_depressed", "coverage_not_depressed", "share_both", "share_single")}
    summary[name] = s
json.dump({"n_people": int(len(y)), "n_depressed": int(y.sum()), "splits": out, "summary": summary},
          open("results/results_extra.json", "w"), indent=1, default=float)
np.savez_compressed("results/predictions_extra.npz", **saved)
print("done", round(time.time() - t0), "s")
