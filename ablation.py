"""Ablation over 10 independent 70:30 splits: individual classifiers vs DKR-Net, with and without SMOTE (training data only)."""
import sys, pandas as pd
sys.path.insert(0, sys.argv[1])
from dkrnet_experiments import load, run_split
X, y = load(sys.argv[2]); rows = []
for r in range(10):
    for sm in (False, True):
        out, _, _ = run_split(X, y, 0.30, 500 + r, use_smote=sm)
        for n, v in out.items():
            rows.append({"run": r + 1, "smote": sm, "model": n, **v["test"]})
    print("ablation run", r + 1, flush=True)
    pd.DataFrame(rows).to_csv(sys.argv[3], index=False)
