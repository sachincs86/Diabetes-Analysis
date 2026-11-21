""""An Intelligent Soft Voting Ensemble Learning Framework for Accurate and Robust Diabetes Prediction".

Pipeline (applied independently inside every training set / CV fold):
  1. Stratified train-test split (70:30 or 80:20)      -> test set is never touched again
  2. Physiologically impossible zeros (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
     are set to NaN and imputed with the TRAINING-set median
  3. z-score standardisation fitted on the TRAINING set only
  4. SMOTE applied to the TRAINING set only (k = 5)
  5. GridSearchCV (5-fold stratified, SMOTE inside each fold via imblearn Pipeline) for DT, RF, KNN
  6. DKR-Net = soft-voting ensemble of the 5 best DT, 5 best RF and 3 best KNN configurations
     (13 weak learners) taken from the ranked cross-validation grid
  7. Evaluation on the untouched test set

Usage:  python dkrnet_experiments.py --data diabetes.csv --out results/ [--runs 30]
        optional: --external frankfurt_diabetes.csv (2000-sample Frankfurt Hospital dataset)
"""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age", "Outcome"]

GRID = {
    "DT": (DecisionTreeClassifier(random_state=0),
           {"max_depth": [3, 4, 5, 6, 8, 10, 12, None], "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"]}),
    "RF": (RandomForestClassifier(random_state=0, n_jobs=1),
           {"n_estimators": [100, 200, 300], "max_depth": [5, 10, None], "min_samples_split": [2, 5]}),
    "KNN": (KNeighborsClassifier(),
            {"n_neighbors": list(range(3, 20, 2)), "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}),
}
TOPK = {"DT": 5, "RF": 5, "KNN": 3}  # 13 weak learners


def load(path):
    df = pd.read_csv(path)
    if "Outcome" not in df.columns:
        df = pd.read_csv(path, header=None, names=COLS)
    return df[COLS[:-1]].copy(), df["Outcome"].values.astype(int)


def preprocess(Xtr, Xte):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in ZERO_AS_MISSING:
        Xtr.loc[Xtr[c] == 0, c] = np.nan
        Xte.loc[Xte[c] == 0, c] = np.nan
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte)), (imp, sc)


def metrics(y, yp, p):
    return {"Accuracy": accuracy_score(y, yp) * 100, "Precision": precision_score(y, yp, zero_division=0) * 100,
            "Recall": recall_score(y, yp) * 100, "F1": f1_score(y, yp) * 100,
            "AUC": roc_auc_score(y, p), "RMSE": float(np.sqrt(np.mean((y - p) ** 2)))}


def tune(name, Xtr, ytr, use_smote, seed):
    est, grid = GRID[name]
    steps = ([("smote", SMOTE(k_neighbors=5, random_state=seed))] if use_smote else []) + [("clf", clone(est))]
    gs = GridSearchCV(ImbPipeline(steps), {f"clf__{k}": v for k, v in grid.items()}, scoring="accuracy",
                      cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1)
    gs.fit(Xtr, ytr)
    res = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
    configs = [{k.replace("clf__", ""): v for k, v in p.items()} for p in res["params"]]
    return configs, res


def build_models(configs, seed):
    best = {n: clone(GRID[n][0]).set_params(**configs[n][0]) for n in GRID}
    members = []
    for n in GRID:
        for i, cfg in enumerate(configs[n][:TOPK[n]]):
            m = clone(GRID[n][0]).set_params(**cfg)
            if "random_state" in m.get_params():
                m.set_params(random_state=seed + i)
            members.append((f"{n}{i+1}", m))
    dkr = VotingClassifier(members, voting="soft", n_jobs=1)
    return best, dkr


def run_split(X, y, test_size, seed, use_smote=True, sampler=None, extra=False, tuned=None):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    Xtr, Xte, prep = preprocess(Xtr, Xte)
    if tuned is None:
        configs = {n: tune(n, Xtr, ytr, use_smote, seed)[0] for n in GRID}
    else:
        configs = tuned
    if use_smote:
        smp = sampler if sampler is not None else SMOTE(k_neighbors=5, random_state=seed)
        Xfit, yfit = smp.fit_resample(Xtr, ytr)
    else:
        Xfit, yfit = Xtr, ytr
    best, dkr = build_models(configs, seed)
    models = dict(best); models["DKR-Net"] = dkr
    if extra:
        models.update({
            "LR": LogisticRegression(max_iter=2000),
            "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=seed),
            "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=seed),
            "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                                     colsample_bytree=0.9, eval_metric="logloss", random_state=seed, n_jobs=1),
            "LightGBM": LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, random_state=seed,
                                       verbose=-1, n_jobs=1)})
    out = {}
    for n, m in models.items():
        m.fit(Xfit, yfit)
        ptr, pte = m.predict_proba(Xfit)[:, 1], m.predict_proba(Xte)[:, 1]
        out[n] = {"train": metrics(yfit, (ptr >= .5).astype(int), ptr), "test": metrics(yte, (pte >= .5).astype(int), pte),
                  "yte": yte, "pte": pte, "ytr": yfit, "ptr": ptr, "model": m}
    return out, configs, (Xfit, yfit, Xte, yte, prep)


def cv_folds(X, y, use_smote, configs, seed=42):
    """5-fold stratified CV; imputation, scaling and SMOTE fitted inside each training fold only."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    res = {n: [] for n in ["DT", "RF", "KNN", "DKR-Net"]}
    for k, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte, _ = preprocess(X.iloc[tr], X.iloc[te])
        ytr, yte = y[tr], y[te]
        if use_smote:
            Xtr, ytr = SMOTE(k_neighbors=5, random_state=seed + k).fit_resample(Xtr, ytr)
        best, dkr = build_models(configs, seed + k)
        allm = dict(best); allm["DKR-Net"] = dkr
        for n, m in allm.items():
            m.fit(Xtr, ytr)
            res[n].append(accuracy_score(yte, m.predict(Xte)) * 100)
    return res


def plot_roc_cm(out, tag, outdir):
    names = ["DT", "RF", "KNN", "DKR-Net"]
    titles = ["(a) Decision Tree", "(b) Random Forest", "(c) KNN", "(d) Proposed DKR-Net"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.2))
    for ax, n, t in zip(axs.ravel(), names, titles):
        for part, ls, col in [("tr", "--", "#1f77b4"), ("te", "-", "#d62728")]:
            yy, pp = out[n]["y" + part], out[n]["p" + part]
            fpr, tpr, _ = roc_curve(yy, pp)
            ax.plot(fpr, tpr, ls, color=col, lw=2,
                    label=f"{'Training' if part == 'tr' else 'Testing'} AUC = {roc_auc_score(yy, pp):.3f}")
        ax.plot([0, 1], [0, 1], ":", color="grey")
        ax.set_title(t); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{outdir}/roc_{tag}.png", dpi=200); plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(9, 8.2))
    for ax, n, t in zip(axs.ravel(), names, titles):
        cm = confusion_matrix(out[n]["yte"], (out[n]["pte"] >= .5).astype(int))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-diabetic (0)", "Diabetic (1)"]); ax.set_yticklabels(["Non-diabetic (0)", "Diabetic (1)"])
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(t)
    fig.tight_layout(); fig.savefig(f"{outdir}/cm_{tag}.png", dpi=200); plt.close(fig)


def external_validation(X, y, ext_path, outdir, seed=42):
    """Train DKR-Net on the complete PIDD (training-only preprocessing + SMOTE) and test on ALL external records."""
    Xe, ye = load(ext_path)
    Xtr, Xte, prep = preprocess(X, Xe)
    configs = {n: tune(n, Xtr, y, True, seed)[0] for n in GRID}
    Xfit, yfit = SMOTE(k_neighbors=5, random_state=seed).fit_resample(Xtr, y)
    _, dkr = build_models(configs, seed)
    dkr.fit(Xfit, yfit)
    p = dkr.predict_proba(Xte)[:, 1]
    res = metrics(ye, (p >= .5).astype(int), p)
    res["cm"] = confusion_matrix(ye, (p >= .5).astype(int)).tolist()
    json.dump(res, open(f"{outdir}/external_validation.json", "w"), indent=1)
    print("external", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--runs", type=int, default=30); ap.add_argument("--external", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    X, y = load(a.data)
    summary = {}
    t0 = time.time()

    # ---- Experiment 1-3: single reference split (seed 42) ----
    exp = {"orig_70": (0.30, False), "smote_70": (0.30, True), "smote_80": (0.20, True)}
    cfg_store = {}
    for tag, (ts, sm) in exp.items():
        out, cfgs, _ = run_split(X, y, ts, 42, use_smote=sm)
        cfg_store[tag] = cfgs
        summary[tag] = {n: {"train": v["train"], "test": v["test"],
                            "cm": confusion_matrix(v["yte"], (v["pte"] >= .5).astype(int)).tolist()}
                        for n, v in out.items()}
        plot_roc_cm(out, tag, a.out)
        print(tag, {n: round(v["test"]["Accuracy"], 2) for n, v in out.items()}, f"{time.time()-t0:.0f}s", flush=True)
        if tag == "smote_70":
            ref_out, ref_data = out, _
    summary["configs"] = {t: {n: c[:TOPK[n]] for n, c in v.items()} for t, v in cfg_store.items()}

    # ---- 5-fold CV (original vs SMOTE-in-fold) ----
    summary["cv_orig"] = cv_folds(X, y, False, cfg_store["orig_70"])
    summary["cv_smote"] = cv_folds(X, y, True, cfg_store["smote_70"])
    print("cv done", f"{time.time()-t0:.0f}s", flush=True)

    # ---- ablation: ensemble without SMOTE already = orig_70; SMOTE individual = smote_70 ----
    # ---- SHAP on reference DKR-Net ----
    try:
        import shap
        Xfit, yfit, Xte, yte, prep = ref_data
        feat = COLS[:-1]
        model = ref_out["DKR-Net"]["model"]
        bg = shap.kmeans(Xfit, 20)
        expl = shap.KernelExplainer(lambda d: model.predict_proba(d)[:, 1], bg)
        rng = np.random.RandomState(0); idx = rng.choice(len(Xte), 120, replace=False)
        sv = expl.shap_values(Xte[idx], nsamples=300, silent=True)
        np.save(f"{a.out}/shap_values.npy", sv); np.save(f"{a.out}/shap_X.npy", Xte[idx])
        Xraw = pd.DataFrame(prep[1].inverse_transform(Xte[idx]), columns=feat)
        plt.figure(); shap.summary_plot(sv, Xraw, show=False, plot_size=(8, 5)); plt.tight_layout()
        plt.savefig(f"{a.out}/shap_beeswarm.png", dpi=200); plt.close("all")
        plt.figure(); shap.summary_plot(sv, Xraw, plot_type="bar", show=False, plot_size=(8, 5)); plt.tight_layout()
        plt.savefig(f"{a.out}/shap_bar.png", dpi=200); plt.close("all")
        # local explanation: a correctly predicted diabetic test patient with highest probability
        p = model.predict_proba(Xte[idx])[:, 1]
        cand = [i for i in range(len(idx)) if yte[idx][i] == 1 and p[i] >= 0.5]
        j = cand[int(np.argsort([p[i] for i in cand])[len(cand) // 2])] if cand else int(np.argmax(p))
        ex = shap.Explanation(values=sv[j], base_values=expl.expected_value, data=Xraw.iloc[j].values, feature_names=feat)
        plt.figure(); shap.plots.waterfall(ex, show=False); plt.tight_layout()
        plt.savefig(f"{a.out}/shap_waterfall.png", dpi=200, bbox_inches="tight"); plt.close("all")
        summary["shap"] = {"mean_abs": dict(zip(feat, np.abs(sv).mean(0).round(4).tolist())),
                           "local": {"prob": float(p[j]), "base": float(expl.expected_value),
                                     "values": dict(zip(feat, np.round(sv[j], 4).tolist())),
                                     "x": dict(zip(feat, Xraw.iloc[j].round(3).tolist()))}}
        print("shap done", f"{time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print("SHAP failed:", e)

    # ---- repeated independent runs (different random splits), incl. comparative models ----
    rows = []
    for r in range(a.runs):
        out, _, _ = run_split(X, y, 0.30, 100 + r, use_smote=True, extra=True)
        for n, v in out.items():
            rows.append({"run": r + 1, "model": n, **v["test"]})
        print("run", r + 1, round(out["DKR-Net"]["test"]["Accuracy"], 2), f"{time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{a.out}/repeated_runs.csv", index=False)

    # ---- SMOTE-variant sensitivity (10 runs, hyper-parameters re-tuned in every run with SMOTE) ----
    vrows = []
    for r in range(10):
        seed = 500 + r
        Xtr0, _, ytr0, _ = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
        Xtr0p, _, _ = preprocess(Xtr0, Xtr0)
        cfgs = {n: tune(n, Xtr0p, ytr0, True, seed)[0] for n in GRID}
        variants = {"No resampling": None, "SMOTE": SMOTE(random_state=seed),
                    "Borderline-SMOTE": BorderlineSMOTE(random_state=seed), "ADASYN": ADASYN(random_state=seed),
                    "SMOTE-Tomek": SMOTETomek(random_state=seed), "SMOTE-ENN": SMOTEENN(random_state=seed)}
        for vn, smp in variants.items():
            out, _, _ = run_split(X, y, 0.30, seed, use_smote=smp is not None, sampler=smp, tuned=cfgs)
            vrows.append({"run": r + 1, "variant": vn, **out["DKR-Net"]["test"]})
        print("variant run", r + 1, f"{time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(vrows).to_csv(f"{a.out}/smote_variants.csv", index=False)

    if a.external:
        summary["external"] = external_validation(X, y, a.external, a.out)

    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("ALL DONE", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
