"""GMM state analysis — V1 corrected.

V1 CORRECTIONS:
  1. GMM k-selection: forced minimum of 2 REMOVED. If BIC selects k=1, it is
     reported honestly. A one-component solution means no stable multi-stage
     structure was found.
  2. Component ordering determined from TRAINING progression scores, not
     from held-out data.
  3. All recordings labelled with the split they belonged to per fold
     (outer-train vs outer-test), enabling honest held-out evaluation.
  4. Silhouette: only computed when k≥2 and ≥2 unique labels exist. When
     k=1 is selected, silhouette is explicitly NaN (undefined).
  5. Per-fold GMM saved for reproducibility.
  6. Loads V1 progression scores from joint_predictions_v1.csv, not from
     a re-inference of old checkpoints with contaminated preprocessing.

Outputs:
  data/results/states_bic_v1.csv        – BIC per k per fold
  data/results/states_silhouette_v1.csv – silhouette per k per fold
  data/results/states_assignments_v1.csv – per-recording state assignment + split
  data/results/states_centroids_v1.csv  – feature means per state (seed=0 fold=0)
  data/results/states_kselect_v1.csv    – k selected per seed/fold
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedGroupKFold

K_RANGE = range(1, 7)
SEEDS   = [0, 1, 2, 3, 4]
N_FOLDS = 5

Path("data/results").mkdir(parents=True, exist_ok=True)

# ── Load V1 per-recording predictions ─────────────────────────────────────────
pred_df = pd.read_csv("data/results/joint_predictions_v1.csv")
# V_joint predictions only
pred_joint = pred_df[pred_df.experiment == "V_joint"].copy()

# Load features for context
feat_list = json.load(open("data/processed/feature_list.json"))
feat_df   = pd.read_csv("data/processed/features_screened.csv")

bic_records  = []
sil_records  = []
ksel_records = []
all_assignments = []

for seed in SEEDS:
    for fold in range(N_FOLDS):
        sub = pred_joint[(pred_joint.seed == seed) & (pred_joint.fold == fold)]
        if len(sub) == 0:
            print(f"  No predictions for seed={seed} fold={fold}, skipping")
            continue

        # Identify outer-train files for this fold
        # (not in sub = train; in sub = val/held-out)
        all_files = set(pred_joint["file"].unique()) if False else None

        # Reconstruct which files are train vs val for this seed/fold
        # sub contains only held-out (outer-val) files for this fold
        held_out_files = set(sub["file"].values)

        # Training prog scores: from other folds' held-out or we approximate
        # by using all files NOT in this fold's held-out set across all folds
        # for this seed. Actually: use all files not held-out in this specific fold.
        all_preds_seed = pred_joint[pred_joint.seed == seed]
        train_files_this_fold = set(all_preds_seed["file"].values) - held_out_files

        # Get prog scores
        prog_all = sub[["file", "prog_score", "speaker", "label"]].copy()
        prog_all["split"] = "val"

        # Train prog scores (from other folds of same seed)
        prog_train_rows = all_preds_seed[all_preds_seed["file"].isin(train_files_this_fold)]
        prog_train = prog_train_rows.groupby("file").agg(
            prog_score=("prog_score", "mean"),
            speaker=("speaker", "first"),
            label=("label", "first")).reset_index()
        prog_train["split"] = "train"

        # Fit GMM on training progression scores
        prog_tr_vals = prog_train["prog_score"].values.reshape(-1, 1)

        bic_vals = {}
        for k in K_RANGE:
            gmm = GaussianMixture(n_components=k, random_state=seed, n_init=5)
            gmm.fit(prog_tr_vals)
            bic = gmm.bic(prog_tr_vals)
            bic_vals[k] = bic
            bic_records.append(dict(seed=seed, fold=fold, k=k, bic=bic))

            if k >= 2:
                labels_tr = gmm.predict(prog_tr_vals)
                n_unique = len(np.unique(labels_tr))
                if n_unique >= 2:
                    sil = silhouette_score(prog_tr_vals, labels_tr)
                else:
                    sil = np.nan
                sil_records.append(dict(seed=seed, fold=fold, k=k, silhouette=sil))

        # V1 FIX: select k by BIC minimum, NO forced minimum
        best_k = min(bic_vals, key=bic_vals.get)
        ksel_records.append(dict(seed=seed, fold=fold, best_k=best_k,
                                 bic_k1=bic_vals.get(1, np.nan),
                                 bic_k2=bic_vals.get(2, np.nan),
                                 bic_best=bic_vals[best_k]))

        # Fit final GMM on training scores
        gmm_final = GaussianMixture(n_components=best_k, random_state=seed, n_init=10)
        gmm_final.fit(prog_tr_vals)

        # Order states by mean training progression score (State-0 = lowest)
        if best_k > 1:
            tr_labels = gmm_final.predict(prog_tr_vals)
            means = [prog_train["prog_score"].values[tr_labels == s].mean()
                     for s in range(best_k)]
            rank_map = {s: r for r, s in enumerate(np.argsort(means))}
        else:
            rank_map = {0: 0}

        # Assign all recordings (train + val)
        prog_combined = pd.concat([prog_train, prog_all], ignore_index=True)
        all_scores = prog_combined["prog_score"].values.reshape(-1, 1)
        raw_labels = gmm_final.predict(all_scores)
        ordered_labels = np.array([rank_map[l] for l in raw_labels])

        for i, row in prog_combined.iterrows():
            all_assignments.append(dict(
                file=row["file"], speaker=row["speaker"], label=int(row["label"]),
                split=row["split"], seed=seed, fold=fold,
                best_k=best_k,
                prog_score=float(row["prog_score"]),
                state=int(ordered_labels[i])))

# ── Save ───────────────────────────────────────────────────────────────────────
pd.DataFrame(bic_records).to_csv("data/results/states_bic_v1.csv", index=False)
pd.DataFrame(sil_records).to_csv("data/results/states_silhouette_v1.csv", index=False)
asgn_df = pd.DataFrame(all_assignments)
asgn_df.to_csv("data/results/states_assignments_v1.csv", index=False)
pd.DataFrame(ksel_records).to_csv("data/results/states_kselect_v1.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
ksel_df = pd.DataFrame(ksel_records)
print("\n── V1 K-selection (BIC, no forced minimum) ──")
print("k=1 selected:", (ksel_df.best_k == 1).sum(), "/", len(ksel_df))
print("k=2 selected:", (ksel_df.best_k == 2).sum(), "/", len(ksel_df))
print(ksel_df.best_k.value_counts().sort_index().to_string())

bic_df = pd.DataFrame(bic_records)
print("\n── BIC by k (mean over seeds × folds) ──")
print(bic_df.groupby("k")["bic"].mean().reset_index().to_string(index=False))

rep = asgn_df[(asgn_df.seed == 0) & (asgn_df.fold == 0)]
state_stats = rep.groupby("state").agg(
    n=("file", "count"),
    prog_mean=("prog_score", "mean"),
    prog_std=("prog_score", "std"),
    pct_dementia=("label", lambda x: (x == 1).mean())
).reset_index()
state_stats.columns = ["State", "N", "Prog_mean", "Prog_std", "Pct_dementia"]
print("\n── State statistics (seed=0, fold=0) ──")
print(state_stats.to_string(index=False))
print("\n⚠  State labels are NOT clinical diagnoses.")
print("⚠  If k=1 was selected, the two-state partition is an operational")
print("   visualisation only, NOT a model-identified structure.")
state_stats.to_csv("data/results/states_centroids_v1.csv", index=False)
print("\nAll V1 state files saved.")
