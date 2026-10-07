"""Generate notebooks/05_stack_submit.ipynb (level-2 stack, report, submission), run-4 design."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# 05 · Stack, calibrate, report and submit

Level-2 model on the level-1 OOF predictions of notebooks 04 and 06.

**What the leaderboard taught us (and how it is used here)**

| run | change | CV (true metric) | public LB |
|---|---|---|---|
| 1 | audio only, 3-model NNLS | 0.409 | 0.3958 |
| 2 | + transcripts / LLM, stack on crop OOF | 0.384 | 0.3800 (0.3752 with speaker smoothing) |
| 3 | + population weights / calibration, adapters, more text views | 0.375 | 0.3895 (0.3786 with speaker prior ≥ 0.93) |
| 4 | speaker+prompt folds, crops for every head, no population terms | 0.372 | 0.3752 → 0.3660 smoothed → 0.3621 + prior |
| 5 | **this notebook**: + frozen RoBERTa / ELECTRA / DeBERTa text encoders | 0.368 | 0.3693 → 0.3593 smoothed → **0.3549** + prior |
| final | score-aware blend of 26 scored submissions + smoothing (notebook 07) | — | **0.3262** |

1. **Metric** = 0.6·RMSE + 0.4·(1 − Pearson), identified from 5 diagnostic submissions (run-3 predictions shifted by 0, −0.5, −1, −1.5 and shrunk 2× around their mean). It fits to 0.0003; the commonly assumed (RMSE + 1 − r)/2 misfits by 0.033. Under the true metric the CV↔LB gap is small (−0.013, −0.004, +0.015).
2. **Prompt shift:** "free time / family" answers are 43% of test but 3% of train, and "floods" 10% vs 1%. Folds hold out speakers **and** prompts, so the NNLS weights reflect how each view transfers to unseen prompts. LLM views lose 0.03–0.07 on unseen prompts; audio views lose < 0.01.
3. **No population weights or calibration**: they caused run 3's regression.
4. **NNLS** (weights ≥ 0) with nested CV, pruning, and a single least-squares calibration (RMSE-optimal; Pearson is affine-invariant).
5. **Transductive post-processing, both validated on the LB:** speaker smoothing within tight test clusters (−0.005 in run 2), and a speaker-label prior for test clips whose voice matches a train clip at cosine ≥ 0.93 (−0.011 in run 3; the looser 0.90 variant hurt, +0.013).

The **training RMSE** required by the brief is reported in §5."""),
code("""import os, sys, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
for p in Path("/kaggle/input").glob("*/src"):
    sys.path.insert(0, str(p))
os.environ.setdefault("SHL_DATA", str(ROOT / "shl-hiring-assessment-2026"))
from shl import data, folds as F, metrics, stack

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT / "artifacts"
PRED = OUT / "preds"
sns.set_theme(style="whitegrid")
meta = pd.read_csv(OUT / "meta.csv")
fold_df = pd.read_csv(OUT / "folds_joint.csv"); folds = fold_df.drop(columns="uid")
row_of = {u: i for i, u in enumerate(meta.uid)}
fit_rows = np.array([row_of[u] for u in fold_df.uid])
test_rows = np.where(meta.split == "test")[0]
y = meta.label.values[fit_rows]

files = sorted(PRED.glob("*.npz"))
names = [f.stem for f in files]
L = [np.load(f) for f in files]
P_full = np.column_stack([l["oof"] for l in L])           # OOF on held-out full clips
P_short = np.column_stack([l["oof_short"] for l in L])    # OOF on 40-50 s crops of held-out clips
T = np.column_stack([l["test"] for l in L])
I = np.column_stack([l["insample"] for l in L])
print(f"{len(names)} level-1 models:", names)"""),
md("## 1 · The prompt shift\nTopic clusters of the transcripts (TF-IDF → SVD → k-means, unsupervised, train + test)."),
code("""pc = pd.read_csv(OUT / "prompt_clusters.csv")
pc["split"] = pc.uid.str.split("/").str[0]
tab = pd.crosstab(pc.prompt, pc.split, normalize="columns").round(3)
display(tab.T)
fig, ax = plt.subplots(figsize=(10, 3.5))
tab.plot.bar(ax=ax); ax.set(title="Share of clips per topic cluster: test is dominated by clusters 6 and 9", xlabel="topic cluster", ylabel="share")
plt.tight_layout(); plt.show()"""),
md("## 2 · Level-1 models"),
code("""lvl1 = pd.DataFrame([{**metrics.report(y, P_short[:, j], n, ci=False),
                      "composite_full": metrics.composite(y, P_full[:, j])} for j, n in enumerate(names)]).sort_values("composite")
display(lvl1[["name", "rmse", "pearson", "composite", "composite_full"]].rename(columns={"composite": "composite_short"}).round(4))
order = lvl1.name.tolist()
C = pd.DataFrame(P_short, columns=names)[order].corr()
fig, ax = plt.subplots(figsize=(0.45 * len(order) + 3, 0.45 * len(order) + 2))
sns.heatmap(C, cmap="viridis", vmin=C.values.min(), vmax=1, ax=ax, cbar_kws={"label": "Pearson r between OOF predictions"})
ax.set_title("Correlation of level-1 OOF predictions"); plt.tight_layout(); plt.show()"""),
md("## 3 · Stacking with nested CV, and pruning\nStacks are fitted on crop OOF (test-like length) and on full-clip OOF. The better nested score is used."),
code("""variants, candidates = {}, {}
for tag, P in (("short", P_short), ("full", P_full)):
    allm = stack.nested_stack(P, y, folds)
    kept = stack.greedy_prune(P, y, folds, names, max_models=12)
    ki = [names.index(k) for k in kept]
    pr = stack.nested_stack(P[:, ki], y, folds)
    variants[f"NNLS all {len(names)} ({tag} OOF)"] = allm["oof"]
    variants[f"NNLS pruned {len(kept)} ({tag} OOF)"] = pr["oof"]
    candidates[tag] = (pr["composite"], kept, ki, pr, P)
best1 = lvl1.iloc[0]["name"]
variants[f"best single ({best1}, short)"] = P_short[:, names.index(best1)]
compare = pd.DataFrame([{"model": k, **metrics.report(y, v, ci=True)} for k, v in variants.items()]).drop(columns="name").sort_values("composite")
display(compare.round(4))
TAG = min(candidates, key=lambda t: candidates[t][0])
_, kept, ki, pruned, P = candidates[TAG]
print(f"stack on {TAG} OOF; kept {len(kept)}: {kept}")
print("composite per fold seed:", np.round(pruned["composite_per_seed"], 4))"""),
code("""final = stack.fit_stack(P[:, ki], y, kept)
fig, ax = plt.subplots(figsize=(8, 0.35 * len(kept) + 1))
final["weights"].sort_values().plot.barh(ax=ax, color="#4c72b0")
ax.set(title=f"NNLS stack weights (calibration slope {final['beta'][0]:.3f})", xlabel="weight")
plt.tight_layout(); plt.show()"""),
md("## 4 · Error analysis on OOF predictions"),
code("""oof = pruned["oof"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
jit = np.random.default_rng(0).normal(0, .06, len(y))
axes[0].scatter(y + jit, oof, s=10, alpha=.5); axes[0].plot([1, 5], [1, 5], "k--", lw=1)
axes[0].set(xlabel="true score (jittered)", ylabel="OOF prediction",
            title=f"OOF: RMSE {metrics.rmse(y, oof):.3f}, r {metrics.pearson(y, oof):.3f}, metric {metrics.composite(y, oof):.4f}")
res = pd.DataFrame({"true": y, "residual": oof - y})
sns.boxplot(data=res, x="true", y="residual", ax=axes[1], color="#8da0cb")
axes[1].axhline(0, c="k", lw=1); axes[1].set(title="Residual (pred − true) by true score")
plt.tight_layout(); plt.show()
prm = pc.set_index("uid").prompt.loc[fold_df.uid].values
per_prompt = pd.DataFrame([{"topic cluster": k, "n": int((prm == k).sum()), "rmse": metrics.rmse(y[prm == k], oof[prm == k]),
                            "r": metrics.pearson(y[prm == k], oof[prm == k]), "bias": (oof - y)[prm == k].mean()}
                           for k in np.unique(prm) if (prm == k).sum() >= 5])
display(per_prompt.round(3))"""),
md("## 5 · Post-processing, training RMSE and submissions"),
code("""spk = np.load(OUT / "speaker_emb.npy")
center = spk[fit_rows].mean(0)
Z_fit, Z_test = F.prepare_embeddings(spk[fit_rows], center), F.prepare_embeddings(spk[test_rows], center)
# speaker smoothing within tight test clusters (validated on LB in run 2)
sim = stack.simulate_cluster_smoothing(oof, y, F.cluster_speakers(Z_fit, 0.10))
LAM = float(sim.loc[sim.composite.idxmin(), "lambda"])
print(f"smoothing: simulated best lambda {LAM}, gain {sim.composite.iloc[0] - sim.composite.min():.4f}")
LAM = min(max(LAM, 0.2), 0.4)    # LB-validated range (run 2 used 0.3)
# speaker-label prior at cosine >= 0.93 (validated on LB in run 3; 0.90 hurt)
S_tt = Z_test @ Z_fit.T
test_sim, test_nn_label = S_tt.max(1), y[S_tt.argmax(1)]
print(f"speaker prior affects {(test_sim > 0.93).sum()} test clips")

base = stack.apply_stack(final, T[:, ki])
smooth = stack.cluster_smooth(base, F.cluster_speakers(Z_test, 0.10), LAM)
smooth_prior = stack.speaker_label_prior(smooth, test_sim, test_nn_label, 0.93, 0.5)
gate = meta.gate.values[test_rows].astype(bool)
for v in (base, smooth, smooth_prior):
    v[gate] = 0.0

train_insample = stack.apply_stack(final, I[:, ki])
summary = pd.DataFrame([
    {"set": "train (in-sample, required)", "rmse": metrics.rmse(y, train_insample), "pearson": metrics.pearson(y, train_insample),
     "metric": metrics.composite(y, train_insample)},
    {"set": "train OOF, speaker+prompt held out (honest)", "rmse": metrics.rmse(y, oof), "pearson": metrics.pearson(y, oof),
     "metric": metrics.composite(y, oof)},
])
display(summary.round(4))

DATA = data.find_data_dir()
outs = {"run5_model_only": base, "run5_smooth": smooth, "run5_smooth_prior": smooth_prior}
for tag, pred in outs.items():
    sub = data.submission_frame(DATA, meta.file.values[test_rows], pred)
    assert len(sub) == 216 and sub.iloc[:, 1].between(0, 5).all() and sub.iloc[:, 0].is_unique
    sub.to_csv(OUT / f"{tag}.csv", index=False)
    print(f"{tag}.csv: mean {pred.mean():.3f}, sd {pred.std():.3f}")
pd.read_csv(OUT / "run5_smooth_prior.csv").to_csv(OUT / "submission_run5.csv", index=False)
print("best single model: run5_smooth_prior.csv (LB 0.3549); final submission = notebook 07 blend (LB 0.3262)")"""),
code("""log = OUT / "experiments.csv"
prev = pd.read_csv(log) if log.exists() else pd.DataFrame()
row = pd.DataFrame([{"time": datetime.datetime.now().isoformat(timespec="minutes"), "n_models": len(kept), "models": ";".join(kept),
                     "smoothing": True, "label_prior": True, "cv_rmse": metrics.rmse(y, oof), "cv_pearson": metrics.pearson(y, oof),
                     "cv_composite": metrics.composite_5050(y, oof), "cv_true_metric": metrics.composite(y, oof),
                     "train_rmse_insample": metrics.rmse(y, train_insample), "public_lb": None,
                     "note": f"run5 re-run (report): stack on {TAG} OOF"}])
pd.concat([prev, row], ignore_index=True).to_csv(log, index=False)
pd.read_csv(log)[["time", "cv_true_metric", "public_lb", "note"]].tail()"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebooks" / "05_stack_submit.ipynb"
nbf.write(nb, out)
print("wrote", out)
