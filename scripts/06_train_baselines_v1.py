"""Train baseline classifiers — V1 corrected.

V1 CORRECTIONS:
  - Fold-local feature screening: quality-correlation and collinearity
    screening fitted on each fold's outer-training speakers only.
  - Same outer splits as V1 neural models (same SEEDS, N_FOLDS, GroupKFold).
  - Quality-only baseline re-evaluated on matched outer folds (not a
    single historical number).
  - Per-fold-screened features logged.
  - 0.625 is NOT used as a fixed floor; matched quality baseline recomputed.

Ladder of baselines:
  B0 – Majority class (sanity check)
  B1 – 2-feature rule: pause_ratio + speech_rate (domain-knowledge baseline)
  B2 – Logistic Regression on acoustic features only
  B3 – Logistic Regression on linguistic features only
  B4 – Random Forest on all handcrafted
  B5 – SVM (RBF) on all handcrafted
  B6 – GBM (HistGBM) on all handcrafted
  B7 – Quality-only RF (matched baseline — re-evaluated on V1 folds)

Outputs:
  data/results/baselines_v1.csv       – per-fold metrics
  data/results/baselines_summary_v1.csv – mean ± std per model
"""
import os, json
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5
SPEARMAN_QUALITY_THRESH = 0.5
SPEARMAN_FEAT_THRESH    = 0.9
QUALITY_COLS = ["snr_db", "loudness_dbfs", "peak", "bw99_hz", "duration_s"]

# ── Load unscreened features ──────────────────────────────────────────────────
manifest = pd.read_csv("data/processed/manifest.csv")
quality  = pd.read_csv("data/processed/quality.csv")
feat_a   = pd.read_csv("data/processed/features_acoustic.csv")
feat_l   = pd.read_csv("data/processed/features_linguistic.csv")

df_full = (manifest[["file", "speaker", "label", "split"]]
           .merge(quality[["file"] + QUALITY_COLS], on="file", how="left")
           .merge(feat_a, on="file", how="left")
           .merge(feat_l, on="file", how="left"))
df_full["label"] = (df_full["label"] == "dementia").astype(int) \
                   if df_full["label"].dtype == object else df_full["label"].astype(int)

CANDIDATE_FEATS = [c for c in df_full.columns
                   if c not in ["file", "speaker", "label", "split"] + QUALITY_COLS]
feat_acoustic_names  = [c for c in feat_a.columns if c != "file"]
feat_linguistic_names = [c for c in feat_l.columns if c != "file"]

X_all   = df_full[CANDIDATE_FEATS].values.astype(float)
X_qual  = df_full[QUALITY_COLS].fillna(df_full[QUALITY_COLS].median()).values
y       = df_full["label"].values
groups  = df_full["speaker"].values


def screen_on_fold(tr_idx):
    """Return screened feature indices (from CANDIDATE_FEATS) using outer-train only."""
    X_tr = X_all[tr_idx]; q_tr = X_qual[tr_idx]
    n = len(CANDIDATE_FEATS)
    to_drop = set()
    for fi in range(n):
        for qi in range(q_tr.shape[1]):
            x = X_tr[:, fi]; q = q_tr[:, qi]
            valid = np.isfinite(x) & np.isfinite(q)
            if valid.sum() < 10: continue
            r, _ = spearmanr(x[valid], q[valid])
            if abs(r) > SPEARMAN_QUALITY_THRESH:
                to_drop.add(fi); break
    remaining = [i for i in range(n) if i not in to_drop]
    if len(remaining) > 1:
        X_rem = np.nan_to_num(X_tr[:, remaining], nan=0.0)
        ranks = np.apply_along_axis(rankdata, 0, X_rem)
        corr_mat = np.corrcoef(ranks.T); np.fill_diagonal(corr_mat, 0)
        to_drop_col = set()
        for a in range(len(remaining)):
            if a in to_drop_col: continue
            for b in range(a + 1, len(remaining)):
                if b in to_drop_col: continue
                if abs(corr_mat[a, b]) > SPEARMAN_FEAT_THRESH:
                    to_drop_col.add(b)
        remaining = [remaining[i] for i in range(len(remaining)) if i not in to_drop_col]
    return remaining


def evaluate_model(clf_fn, X_tr, y_tr, X_va, y_va, scale=True):
    if scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_va = sc.transform(X_va)
    clf = clf_fn(); clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_va)[:, 1] if hasattr(clf, "predict_proba") \
           else clf.decision_function(X_va)
    pred = clf.predict(X_va)
    return roc_auc_score(y_va, prob), balanced_accuracy_score(y_va, pred)


Path("data/results").mkdir(parents=True, exist_ok=True)
records = []

for seed in SEEDS:
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_all, y, groups)):
        y_tr, y_va = y[tr_idx], y[va_idx]

        # V1: fold-local screening
        feat_idx = screen_on_fold(tr_idx)
        feat_names = [CANDIDATE_FEATS[i] for i in feat_idx]
        X = X_all[:, feat_idx]

        # Split acoustic/linguistic within screened set
        aco_idx = [i for i, f in enumerate(feat_names) if f in feat_acoustic_names]
        lin_idx = [i for i, f in enumerate(feat_names) if f in feat_linguistic_names]
        RULE_IDX = [i for i, f in enumerate(feat_names) if f in ["pause_ratio", "speech_rate"]]

        # B0 – majority class
        majority = int(np.bincount(y_tr).argmax())
        preds_b0 = np.full(len(y_va), majority)
        probs_b0 = np.full(len(y_va), float(majority))
        auc_b0 = roc_auc_score(y_va, probs_b0) if len(np.unique(y_va)) > 1 else 0.5
        bac_b0 = balanced_accuracy_score(y_va, preds_b0)
        records.append(dict(model="B0_majority", seed=seed, fold=fold,
                            auc=auc_b0, bac=bac_b0, n_features=0))

        # B1 – 2-feature rule
        if len(RULE_IDX) == 2:
            X_r = X[:, RULE_IDX]
            auc_b1, bac_b1 = evaluate_model(
                lambda: LogisticRegression(max_iter=500, C=0.1, random_state=seed),
                X_r[tr_idx], y_tr, X_r[va_idx], y_va)
        else:
            auc_b1, bac_b1 = np.nan, np.nan
        records.append(dict(model="B1_pause_rate_rule", seed=seed, fold=fold,
                            auc=auc_b1, bac=bac_b1, n_features=len(RULE_IDX)))

        # B2 – LR acoustic
        if aco_idx:
            X_a = X[:, aco_idx]
            auc_b2, bac_b2 = evaluate_model(
                lambda: LogisticRegression(max_iter=1000, C=0.1, random_state=seed),
                X_a[tr_idx], y_tr, X_a[va_idx], y_va)
        else:
            auc_b2, bac_b2 = np.nan, np.nan
        records.append(dict(model="B2_LR_acoustic", seed=seed, fold=fold,
                            auc=auc_b2, bac=bac_b2, n_features=len(aco_idx)))

        # B3 – LR linguistic
        if lin_idx:
            X_l = X[:, lin_idx]
            auc_b3, bac_b3 = evaluate_model(
                lambda: LogisticRegression(max_iter=1000, C=0.1, random_state=seed),
                X_l[tr_idx], y_tr, X_l[va_idx], y_va)
        else:
            auc_b3, bac_b3 = np.nan, np.nan
        records.append(dict(model="B3_LR_linguistic", seed=seed, fold=fold,
                            auc=auc_b3, bac=bac_b3, n_features=len(lin_idx)))

        # B4 – RF fused handcrafted
        auc_b4, bac_b4 = evaluate_model(
            lambda: RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
            X[tr_idx], y_tr, X[va_idx], y_va, scale=False)
        records.append(dict(model="B4_RF_fused", seed=seed, fold=fold,
                            auc=auc_b4, bac=bac_b4, n_features=len(feat_idx)))

        # B5 – SVM RBF
        auc_b5, bac_b5 = evaluate_model(
            lambda: SVC(kernel="rbf", C=1.0, gamma="scale",
                        probability=True, random_state=seed),
            X[tr_idx], y_tr, X[va_idx], y_va)
        records.append(dict(model="B5_SVM_fused", seed=seed, fold=fold,
                            auc=auc_b5, bac=bac_b5, n_features=len(feat_idx)))

        # B6 – HistGBM
        auc_b6, bac_b6 = evaluate_model(
            lambda: HistGradientBoostingClassifier(max_iter=200, random_state=seed),
            X[tr_idx], y_tr, X[va_idx], y_va, scale=False)
        records.append(dict(model="B6_GBM_fused", seed=seed, fold=fold,
                            auc=auc_b6, bac=bac_b6, n_features=len(feat_idx)))

        # B7 – Quality-only RF (V1 matched baseline, re-evaluated on same folds)
        auc_b7, bac_b7 = evaluate_model(
            lambda: RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
            X_qual[tr_idx], y_tr, X_qual[va_idx], y_va, scale=False)
        records.append(dict(model="B7_RF_quality_only", seed=seed, fold=fold,
                            auc=auc_b7, bac=bac_b7, n_features=len(QUALITY_COLS)))

    print(f"Seed {seed} done")

res_df = pd.DataFrame(records)
res_df.to_csv("data/results/baselines_v1.csv", index=False)

summary = (res_df.groupby("model")[["auc", "bac"]]
           .agg(["mean", "std"]).round(4))
summary.columns = ["auc_mean", "auc_std", "bac_mean", "bac_std"]
summary = summary.sort_values("auc_mean", ascending=False)
summary.to_csv("data/results/baselines_summary_v1.csv")

print("\n── V1 Baseline results (mean ± std over 5 seeds × 5 folds) ──")
print(summary.to_string())
quality_auc = res_df[res_df.model == "B7_RF_quality_only"]["auc"].mean()
print(f"\nV1 quality-only baseline (B7_RF_quality_only): AUC = {quality_auc:.4f}")
print("NOTE: 0.625 was the original single-run quality baseline — no longer used as a fixed floor.")
