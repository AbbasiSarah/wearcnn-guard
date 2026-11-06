import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
R = json.load(open("results/results.json")); S = R["splits"]
NAVY, TEAL, GOLD, GREY, RED = "#0b1e4a", "#1aa88f", "#d99a00", "#8a94a8", "#c8475f"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
names = list(S)
short = {"LogReg (rhythm features)": "LogReg\nrhythm", "GradBoost (rhythm features)": "GradBoost\nrhythm",
         "LogReg (rhythm + age, sex)": "LogReg\n+age, sex", "GradBoost (rhythm + age, sex)": "GradBoost\n+age, sex",
         "1-D CNN (activity only)": "1-D CNN\nactivity only"}
# Fig 1: coverage for depressed people, marginal vs Mondrian
fig, ax = plt.subplots(figsize=(8.4, 4.8)); x = np.arange(len(names)); w = 0.36
for k, (key, col, lab) in enumerate([("marginal_a0.1_coverage_depressed", GREY, "Standard conformal"), ("mondrian_a0.1_coverage_depressed", TEAL, "Class-conditional (Mondrian)")]):
    v = np.array([[r[key] for r in S[n]] for n in names])
    ax.bar(x + (k - 0.5) * w, v.mean(1), w, yerr=v.std(1), color=col, label=lab, capsize=3)
ax.axhline(0.9, color=RED, ls="--", lw=1.2); ax.text(len(names) - 0.5, 0.915, "90% target", color=RED, ha="right", fontsize=10)
ax.set_xticks(x, [short[n] for n in names]); ax.set_ylim(0, 1.05); ax.set_ylabel("Coverage for depressed people")
ax.set_title("The guarantee only protects the minority class when calibrated per class"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.tight_layout(); fig.savefig("results/fig1_coverage_depressed.png", dpi=200)
# Fig 2: ROC (split 0)
P = np.load("results/split0_preds.npz"); d = np.load("data/nhanes_week.npz"); y = (d["phq9"] >= 10).astype(int)
fig, ax = plt.subplots(figsize=(5.2, 5))
for n, col in zip(names, [GREY, "#b6bfd0", NAVY, "#5d6f99", TEAL]):
    te = P[f"{n}|te"].astype(int); f, t, _ = roc_curve(y[te], P[f"{n}|p_te"])
    m = np.mean([r["auc"] for r in S[n]]); ax.plot(f, t, color=col, lw=2.2 if "CNN" in n else 1.4, label=f"{n} (AUC {m:.3f})")
ax.plot([0, 1], [0, 1], color="#cccccc", lw=1); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
ax.set_title("ROC curves (test set, split 1)"); ax.legend(frameon=False, fontsize=8.2, loc="lower right"); fig.tight_layout(); fig.savefig("results/fig2_roc.png", dpi=200)
# Fig 3: average 24-h activity profile
X, M = d["X"], d["M"]; w_ = M[:, :, None]; prof = (X * w_).sum(1) / w_.sum(1); hrs = np.arange(144) / 6
fig, ax = plt.subplots(figsize=(8, 3.8))
for lab, col, mask in [("PHQ-9 ≥ 10 (depressed)", RED, y == 1), ("PHQ-9 < 10", NAVY, y == 0)]:
    mu = prof[mask].mean(0); se = prof[mask].std(0) / np.sqrt(mask.sum())
    ax.plot(hrs, mu, color=col, lw=2, label=f"{lab}, n = {mask.sum():,}"); ax.fill_between(hrs, mu - 2 * se, mu + 2 * se, color=col, alpha=0.18)
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean log(1 + activity count)")
ax.set_title("Average day: depressed adults are less active from morning to evening"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig("results/fig3_daily_profile.png", dpi=200)
# Fig 4: what the screener tells people (CNN, Mondrian, alpha 0.1, mean over splits)
c = S["1-D CNN (activity only)"]
both = np.mean([r["mondrian_a0.1_share_both"] for r in c]); single = np.mean([r["mondrian_a0.1_share_single"] for r in c])
fig, ax = plt.subplots(figsize=(6, 1.9))
ax.barh([0], [single], color=TEAL); ax.barh([0], [both], left=[single], color=GOLD)
ax.text(single / 2, 0, f"Confident single answer\n{single:.0%}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
ax.text(single + both / 2, 0, f"'Uncertain': refer to the PHQ-9 questionnaire\n{both:.0%}", ha="center", va="center", color=NAVY, fontsize=10, fontweight="bold")
ax.set_xlim(0, 1); ax.axis("off"); ax.set_title("1-D CNN with a 90% guarantee: what each person is told", fontsize=11)
fig.tight_layout(); fig.savefig("results/fig4_triage.png", dpi=200)
print("figures ok")
