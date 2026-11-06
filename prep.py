"""Build per-person weekly activity profiles from NHANES 2011-2014 and join PHQ-9."""
import numpy as np, pandas as pd, lzma, time
import os, sys
D = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NHANES_DIR", "nhanes_raw")).rstrip("/") + "/"
os.makedirs("data", exist_ok=True)
t = time.time()
mins = [f"min_{i:04d}" for i in range(1, 1441)]
rows = []
for chunk in pd.read_csv(lzma.open(D + "nhanes_1440_AC.csv.xz", "rt"), chunksize=4000,
                         dtype={c: "float32" for c in mins}, na_values=["NA"]):
    x = chunk[mins].to_numpy(np.float32)                       # (days, 1440)
    # valid wear: >= 10 h of non-missing minutes and not mostly zeros (simple Troiano-style rule)
    obs = np.isfinite(x)
    xz = np.where(obs, x, 0.0)
    # non-wear = runs of >= 60 consecutive zero minutes
    z = (xz == 0) & obs
    run = np.zeros_like(z, dtype=np.int32); nonwear = np.zeros_like(z)
    for j in range(1440):
        run[:, j] = np.where(z[:, j], (run[:, j-1] + 1) if j else 1, 0)
    # mark the whole run once it reaches 60: back-fill by scanning from the right
    end_len = np.zeros(z.shape[0], np.int32)
    for j in range(1439, -1, -1):
        end_len = np.where(z[:, j], np.maximum(end_len, run[:, j]), 0)
        nonwear[:, j] = z[:, j] & (end_len >= 60)
    wear_min = (obs & ~nonwear).sum(1)
    valid = obs & ~nonwear                                         # worn minutes only
    lx = np.where(valid, np.log1p(xz), np.nan).reshape(len(x), 144, 10)
    with np.errstate(all="ignore"):
        b = np.nanmean(lx, 2)                                      # 10-minute bins of worn time (NaN if none)
    for seqn, day, wd, wm, prof in zip(chunk.SEQN.values, chunk.PAXDAYM.values, chunk.PAXDAYWM.values, wear_min, b):
        rows.append((int(seqn), int(day), int(wd), int(wm), prof.astype(np.float32)))
    print(len(rows), round(time.time() - t), "s", flush=True)
df = pd.DataFrame(rows, columns=["SEQN", "day", "weekday", "wear_min", "prof"])
df = df[df.wear_min >= 600]                                        # >= 10 h worn
cnt = df.groupby("SEQN").size(); keep = cnt[cnt >= 4].index        # >= 4 valid days
df = df[df.SEQN.isin(keep)].sort_values(["SEQN", "day"])
# 7-day array (pad missing valid days with the person's mean day), plus mask
ids = np.array(sorted(keep)); X = np.zeros((len(ids), 7, 144), np.float32); M = np.zeros((len(ids), 7), bool)
WD = np.zeros((len(ids), 7), np.int8); idx = {s: i for i, s in enumerate(ids)}
for s, g in df.groupby("SEQN"):
    i = idx[s]; P = np.stack(g.prof.values)[:7]; M[i, :len(P)] = True; WD[i, :len(P)] = g.weekday.values[:7]
    with np.errstate(all="ignore"):
        mean_day = np.nanmean(P, 0)
    mean_day = np.where(np.isfinite(mean_day), mean_day, np.nanmean(mean_day))
    P = np.where(np.isfinite(P), P, mean_day)                      # unworn bins -> person's mean for that time
    X[i, :len(P)] = P
    if len(P) < 7: X[i, len(P):] = mean_day
# PHQ-9
q = pd.concat([pd.read_sas(D + f) for f in ["DPQ_G.xpt", "DPQ_H.xpt"]])
items = [f"DPQ0{k}0" for k in range(1, 10)]
v = q[items].where(q[items] <= 3)
q["phq9"] = v.sum(axis=1, min_count=9)
demo = pd.concat([pd.read_sas(D + f) for f in ["DEMO_G.xpt", "DEMO_H.xpt"]])[["SEQN", "RIAGENDR", "RIDAGEYR", "SDDSRVYR", "INDFMPIR"]]
lab = q[["SEQN", "phq9"]].dropna().merge(demo, on="SEQN")
lab = lab[lab.RIDAGEYR >= 18]
lab["SEQN"] = lab.SEQN.astype(int)
sel = np.isin(ids, lab.SEQN.values)
ids, X, M, WD = ids[sel], X[sel], M[sel], WD[sel]
lab = lab.set_index("SEQN").loc[ids].reset_index()
np.savez_compressed("data/nhanes_week.npz", ids=ids, X=X, M=M, WD=WD, phq9=lab.phq9.values.astype(np.float32),
                    sex=lab.RIAGENDR.values, age=lab.RIDAGEYR.values, cycle=lab.SDDSRVYR.values, pir=lab.INDFMPIR.values)
print("people", len(ids), "depressed (PHQ>=10)", int((lab.phq9 >= 10).sum()), f"{(lab.phq9>=10).mean():.3f}",
      "valid days mean", M.sum(1).mean().round(2), "time", round(time.time() - t), "s")
