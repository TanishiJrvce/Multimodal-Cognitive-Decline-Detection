"""Train baseline classifiers for group prediction (dementia vs. comparison).

Ladder of baselines:
  B0 – Majority class (sanity check)
  B1 – 2-feature rule: pause_ratio + speech_rate (domain-knowledge baseline)
  B2 – Logistic Regression on acoustic features only
  B3 – Logistic Regression on linguistic features only
  B4 – Random Forest on acoustic + linguistic (fused handcrafted)
  B5 – SVM (RBF) on acoustic + linguistic
  B6 – GBM (HistGBM) on acoustic + linguistic

All trained with:
  - 5-fold StratifiedGroupKFold (grouped by speaker)
  - 5 random seeds → report mean ± std AUC and balanced accuracy
  - Scaler fit on training fold, applied to validation fold (no leakage)

Must beat quality-only RF baseline (AUC 0.625).

Outputs:
  data/results/baselines.csv      – per-fold / per-seed metrics
  data/results/baselines_summary.csv – mean ± std per model
"""
import os, json
# Prevent joblib from spawning subprocesses (blocked in sandbox)
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5
QUALITY_BASELINE_AUC = 0.625  # AUC of quality-only RF (shortcut floor)

# ── Load ───────────────────────────────────────────────────────────────────────
feat_list = json.load(open("data/processed/feature_list.json"))
df = pd.read_csv("data/processed/features_screened.csv")
# 'label' is already 0/1 int (0=nodementia, 1=dementia)

# Keep rows with no NaN in feature columns
df_clean = df.dropna(subset=feat_list).copy()
print(f"Recordings after dropping NaN rows: {len(df_clean)} / {len(df)}")

X = df_clean[feat_list].values.astype(float)
y = df_clean["label"].values.astype(int)
groups = df_clean["speaker"].values

# Two-feature baseline columns (may not survive screening; fall back gracefully)
RULE_FEATS = [f for f in ["pause_ratio", "speech_rate"] if f in feat_list]
X_rule = df_clean[RULE_FEATS].values if RULE_FEATS else None

# Acoustic vs. linguistic split
feat_a = json.load(open("data/processed/feature_list.json"))  # re-read (screened)
feat_acoustic_names  = [f for f in feat_a if f in
    pd.read_csv("data/processed/features_acoustic.csv").columns.tolist()]
feat_linguistic_names = [f for f in feat_a if f not in feat_acoustic_names]
X_aco = df_clean[feat_acoustic_names].values  if feat_acoustic_names  else X
X_lin = df_clean[feat_linguistic_names].values if feat_linguistic_names else X

# ── Helpers ────────────────────────────────────────────────────────────────────
def evaluate_model(clf_fn, X_tr, y_tr, X_va, y_va, scale=True):
    """Fit clf on (X_tr, y_tr) → evaluate on (X_va, y_va)."""
    if scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_va = sc.transform(X_va)
    clf = clf_fn()
    clf.fit(X_tr, y_tr)
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_va)[:, 1]
    else:
        prob = clf.decision_function(X_va)
    pred = clf.predict(X_va)
    auc  = roc_auc_score(y_va, prob)
    bac  = balanced_accuracy_score(y_va, pred)
    return auc, bac

# ── CV Loop ────────────────────────────────────────────────────────────────────
Path("data/results").mkdir(parents=True, exist_ok=True)
records = []

for seed in SEEDS:
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
        y_tr, y_va = y[tr_idx], y[va_idx]
        g_tr = groups[tr_idx]

        # B0 – majority class
        majority = int(np.bincount(y_tr).argmax())
        preds_b0 = np.full(len(y_va), majority)
        probs_b0 = np.full(len(y_va), float(majority))
        auc_b0 = roc_auc_score(y_va, probs_b0) if len(np.unique(y_va)) > 1 else 0.5
        bac_b0 = balanced_accuracy_score(y_va, preds_b0)
        records.append(dict(model="B0_majority", seed=seed, fold=fold,
                            auc=auc_b0, bac=bac_b0))

        # B1 – 2-feature rule (LR on pause_ratio + speech_rate)
        if X_rule is not None and len(RULE_FEATS) == 2:
            auc_b1, bac_b1 = evaluate_model(
                lambda: LogisticRegression(max_iter=500, C=0.1, random_state=seed),
                X_rule[tr_idx], y_tr, X_rule[va_idx], y_va)
        else:
            auc_b1, bac_b1 = np.nan, np.nan
        records.append(dict(model="B1_pause_rate_rule", seed=seed, fold=fold,
                            auc=auc_b1, bac=bac_b1))

        # B2 – LR acoustic only
        if len(feat_acoustic_names) > 0:
            auc_b2, bac_b2 = evaluate_model(
                lambda: LogisticRegression(max_iter=1000, C=0.1, random_state=seed),
                X_aco[tr_idx], y_tr, X_aco[va_idx], y_va)
        else:
            auc_b2, bac_b2 = np.nan, np.nan
        records.append(dict(model="B2_LR_acoustic", seed=seed, fold=fold,
                            auc=auc_b2, bac=bac_b2))

        # B3 – LR linguistic only
        if len(feat_linguistic_names) > 0:
            auc_b3, bac_b3 = evaluate_model(
                lambda: LogisticRegression(max_iter=1000, C=0.1, random_state=seed),
                X_lin[tr_idx], y_tr, X_lin[va_idx], y_va)
        else:
            auc_b3, bac_b3 = np.nan, np.nan
        records.append(dict(model="B3_LR_linguistic", seed=seed, fold=fold,
                            auc=auc_b3, bac=bac_b3))

        # B4 – RF fused handcrafted
        auc_b4, bac_b4 = evaluate_model(
            lambda: RandomForestClassifier(n_estimators=200, random_state=seed,
                                           n_jobs=1),
            X[tr_idx], y_tr, X[va_idx], y_va, scale=False)
        records.append(dict(model="B4_RF_fused", seed=seed, fold=fold,
                            auc=auc_b4, bac=bac_b4))

        # B5 – SVM RBF fused handcrafted
        auc_b5, bac_b5 = evaluate_model(
            lambda: SVC(kernel="rbf", C=1.0, gamma="scale",
                        probability=True, random_state=seed),
            X[tr_idx], y_tr, X[va_idx], y_va)
        records.append(dict(model="B5_SVM_fused", seed=seed, fold=fold,
                            auc=auc_b5, bac=bac_b5))

        # B6 – HistGBM fused handcrafted (handles NaN natively)
        auc_b6, bac_b6 = evaluate_model(
            lambda: HistGradientBoostingClassifier(max_iter=200, random_state=seed),
            X[tr_idx], y_tr, X[va_idx], y_va, scale=False)
        records.append(dict(model="B6_GBM_fused", seed=seed, fold=fold,
                            auc=auc_b6, bac=bac_b6))

    print(f"Seed {seed} done")

# ── Summarise ──────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(records)
res_df.to_csv("data/results/baselines.csv", index=False)

summary = (res_df.groupby("model")[["auc", "bac"]]
           .agg(["mean", "std"]).round(4))
summary.columns = ["auc_mean", "auc_std", "bac_mean", "bac_std"]
summary = summary.sort_values("auc_mean", ascending=False)
summary.to_csv("data/results/baselines_summary.csv")

print("\n── Baseline results (mean ± std over 5 seeds × 5 folds) ──")
print(summary.to_string())
print(f"\nQuality-only floor: AUC = {QUALITY_BASELINE_AUC}")
best_auc = summary["auc_mean"].max()
print(f"Best handcrafted AUC: {best_auc:.4f} "
      f"({'✓ beats' if best_auc > QUALITY_BASELINE_AUC else '✗ does NOT beat'} quality baseline)")
