Supplementary Material S1 - DKR-Net source code
Requirements: python>=3.10, scikit-learn 1.9, imbalanced-learn 0.14, xgboost 3.4, lightgbm 4.7, shap 0.54, pandas, matplotlib, seaborn, scipy
1) python dkrnet_experiments.py --data diabetes.csv --out results --runs 30 [--external frankfurt_diabetes.csv]
   -> Tables 6-11, Figures 5-13, results/summary.json, repeated_runs.csv, smote_variants.csv (+ external_validation.json)
2) python ablation.py <folder containing dkrnet_experiments.py> diabetes.csv results/ablation.csv   -> Table 12
3) python analyse.py results   -> tables.json, Wilcoxon tests, box plots (Figure 11)
diabetes.csv = Pima Indians Diabetes Dataset (768 rows; header Pregnancies,...,Outcome).
