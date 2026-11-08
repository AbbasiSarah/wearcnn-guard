import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = json.load(open("results/results_extra.json"))["summary"]
groups = ["men", "women", "age 18-39", "age 40-59", "age 60+"]
rows = [("CNN, activity only\n(class-conditional)", [S["cnn"]["subgroups_mondrian_a0.1"][g]["coverage_depressed"] for g in groups], "#9aa5b1"),
        ("CNN + age, sex\n(class-conditional)", [S["cnn_demo"]["subgroups_mondrian_a0.1"][g]["coverage_depressed"] for g in groups], "#d9822b"),
        ("CNN + age, sex\n(class × sex)", [S["cnn_demo"]["mondrian_sex_a0.1"]["subgroups"][g]["coverage_depressed"] for g in groups], "#2b6cb0")]
fig, ax = plt.subplots(figsize=(7.5, 3.6)); w = 0.26; x = np.arange(len(groups))
for i, (lab, v, c) in enumerate(rows):
    ax.bar(x + (i - 1) * w, v, w, label=lab, color=c)
ax.axhline(0.9, color="k", ls="--", lw=1); ax.text(-0.45, 0.905, "90% target", ha="left", fontsize=8)
ax.set_xticks(x, groups); ax.set_ylim(0.6, 1.02); ax.set_ylabel("Coverage among depressed people")
ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(); plt.savefig("results/fig5_subgroup_coverage.png", dpi=200)
