import json, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

R = sys.argv[1]
S = json.load(open(f"{R}/summary.json"))
rep = pd.read_csv(f"{R}/repeated_runs.csv")
var = pd.read_csv(f"{R}/smote_variants.csv")
out = {}

def fmt(tag):
    rows = []
    for m in ["DT", "RF", "KNN", "DKR-Net"]:
        d = S[tag][m]
        rows.append([m] + [round(d["train"][k], 2) for k in ["Accuracy", "Precision", "Recall", "F1"]] +
                    [round(d["test"][k], 2) for k in ["Accuracy", "Precision", "Recall", "F1"]] +
                    [round(d["train"]["AUC"], 3), round(d["test"]["AUC"], 3)] + [d["cm"]])
    return rows
for t in ["orig_70", "smote_70", "smote_80"]:
    out[t] = fmt(t)
out["cv_orig"] = {k: [round(x, 2) for x in v] + [round(np.mean(v), 2)] for k, v in S["cv_orig"].items()}
out["cv_smote"] = {k: [round(x, 2) for x in v] + [round(np.mean(v), 2)] for k, v in S["cv_smote"].items()}
out["configs"] = S["configs"]
out["shap"] = S.get("shap")

order = ["DKR-Net", "RF", "DT", "KNN", "LR", "SVM", "AdaBoost", "XGBoost", "LightGBM"]
stat = []
for m in order:
    d = rep[rep.model == m]
    row = {"model": m}
    for k in ["Accuracy", "Precision", "Recall", "F1", "AUC", "RMSE"]:
        row[k + "_mean"] = round(d[k].mean(), 3 if k in ("AUC", "RMSE") else 2)
        row[k + "_std"] = round(d[k].std(ddof=1), 3 if k in ("AUC", "RMSE") else 2)
    stat.append(row)
out["repeated"] = stat
# Wilcoxon DKR-Net vs others (accuracy & AUC)
wx = {}
base = rep[rep.model == "DKR-Net"].sort_values("run")
for m in order[1:]:
    o = rep[rep.model == m].sort_values("run")
    wx[m] = {k: float(wilcoxon(base[k].values, o[k].values).pvalue) for k in ["Accuracy", "AUC", "F1"]}
out["wilcoxon"] = wx
out["n_runs"] = int(rep.run.nunique())
vs = []
for v in ["No resampling", "SMOTE", "Borderline-SMOTE", "ADASYN", "SMOTE-Tomek", "SMOTE-ENN"]:
    d = var[var.variant == v]
    vs.append({"variant": v, **{k + "_mean": round(d[k].mean(), 2 if k not in ("AUC",) else 3) for k in ["Accuracy", "Recall", "F1", "AUC"]},
               **{k + "_std": round(d[k].std(ddof=1), 2 if k not in ("AUC",) else 3) for k in ["Accuracy", "Recall", "F1", "AUC"]}})
out["variants"] = vs
json.dump(out, open(f"{R}/tables.json", "w"), indent=1)

# box plots
fig, axs = plt.subplots(1, 4, figsize=(15, 4.6), sharey=False)
show = ["DKR-Net", "RF", "DT", "KNN", "XGBoost", "LightGBM", "SVM", "LR"]
for ax, k in zip(axs, ["Accuracy", "Precision", "Recall", "F1"]):
    data = [rep[rep.model == m][k].values for m in show]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color="black"))
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor("#F4B183" if i == 0 else "#BDD7EE")
    ax.set_xticks(range(1, len(show) + 1)); ax.set_xticklabels(show, rotation=45, ha="right")
    ax.set_title(k + " (%)"); ax.grid(axis="y", alpha=.3)
fig.suptitle(f"Test-set performance over {out['n_runs']} independent 70:30 runs (SMOTE applied to training data only)")
fig.tight_layout(); fig.savefig(f"{R}/boxplots.png", dpi=200); plt.close(fig)
print(json.dumps({k: out[k] for k in ["repeated", "variants", "cv_orig", "cv_smote", "wilcoxon"]}, indent=0)[:6000])
for t in ["orig_70", "smote_70", "smote_80"]:
    print(t); [print(r) for r in out[t]]
print(json.dumps(out["configs"], indent=0)[:3000]); print(out["shap"])
