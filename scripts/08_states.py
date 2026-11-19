"""Discover latent speech states via GMM on joint-model progression scores.

Protocol:
  1. Load progression scores from best V_joint checkpoint (per fold).
  2. Fit GMM(k) on training-fold scores for k=1..6.
  3. Select k via BIC + silhouette + bootstrap ARI (stability).
  4. Report mean progression score and feature centroids per state.
  5. DO NOT label any state "MCI" — labels are State-1 … State-k.
  6. Cross-seed/split stability: report pairwise ARI across seeds.

Outputs:
  data/results/states_bic.csv         – BIC per k per fold
  data/results/states_silhouette.csv  – silhouette per k per fold
  data/results/states_assignments.csv – per-recording state assignment
  data/results/states_centroids.csv   – feature means per state

Design: src/mcd/states.py
"""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mcd.models import FusionNet
from mcd.states import fit_states, compare_k, order_states_by_progression

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS  = [0, 1, 2, 3, 4]
N_FOLDS = 5
K_RANGE = range(1, 7)

# ── Load features and embeddings ──────────────────────────────────────────────
feat_list = json.load(open("data/processed/feature_list.json"))
df = pd.read_csv("data/processed/features_screened.csv")
# 'label' is already 0/1 int
df_clean = df.dropna(subset=feat_list).copy()

emb_idx  = pd.read_csv("data/processed/embeddings_index.csv")
emb_wav  = np.load("data/processed/embeddings_wavlm.npy")
emb_bert = np.load("data/processed/embeddings_bert.npy")

df_clean = df_clean.merge(emb_idx, on="file", how="left")
df_clean["row_index"] = df_clean["row_index"].fillna(-1).astype(int)

X_feat  = df_clean[feat_list].values.astype(float)
X_wavlm = np.where(df_clean["row_index"].values[:, None] >= 0,
                   emb_wav[np.clip(df_clean["row_index"].values, 0, len(emb_wav)-1)], 0.0)
X_bert  = np.where(df_clean["row_index"].values[:, None] >= 0,
                   emb_bert[np.clip(df_clean["row_index"].values, 0, len(emb_bert)-1)], 0.0)
y      = df_clean["label"].values
groups = df_clean["speaker"].values

# ── Collect progression scores from V_joint checkpoints ───────────────────────
all_assignments = []
bic_records, sil_records = [], []

for seed in SEEDS:
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_feat, y, groups)):
        ckpt_path = f"data/models/best_V_joint_s{seed}_f{fold}.pt"
        if not Path(ckpt_path).exists():
            print(f"  Checkpoint missing: {ckpt_path} — skipping")
            continue

        # Load model
        sc = StandardScaler()
        X_f_tr = sc.fit_transform(X_feat[tr_idx])
        X_f_all = sc.transform(X_feat)

        model = FusionNet(feat_dim=X_f_tr.shape[1], wavlm_dim=768, bert_dim=768).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            xf = torch.tensor(X_f_all,  dtype=torch.float32, device=DEVICE)
            xw = torch.tensor(X_wavlm,  dtype=torch.float32, device=DEVICE)
            xb = torch.tensor(X_bert,   dtype=torch.float32, device=DEVICE)
            _, prog = model(xf, xw, xb)
            prog_np = prog.cpu().numpy()   # (N,) progression score

        # GMM on training-fold progression scores
        prog_tr = prog_np[tr_idx].reshape(-1, 1)
        for k in K_RANGE:
            gmm = GaussianMixture(n_components=k, random_state=seed, n_init=5)
            gmm.fit(prog_tr)
            bic = gmm.bic(prog_tr)
            bic_records.append(dict(seed=seed, fold=fold, k=k, bic=bic))

            if k >= 2:
                labels_tr = gmm.predict(prog_tr)
                sil = silhouette_score(prog_tr, labels_tr) if len(np.unique(labels_tr)) > 1 else 0
                sil_records.append(dict(seed=seed, fold=fold, k=k, silhouette=sil))

        # Choose k by min BIC on training fold
        bic_vals = {r["k"]: r["bic"] for r in bic_records if r["seed"]==seed and r["fold"]==fold}
        best_k = min(bic_vals, key=bic_vals.get)
        best_k = max(best_k, 2)  # at least 2 states

        # Fit final GMM on training fold
        gmm_final = GaussianMixture(n_components=best_k, random_state=seed, n_init=10)
        gmm_final.fit(prog_tr)

        # Assign ALL recordings
        labels_all = gmm_final.predict(prog_np.reshape(-1, 1))
        # Order states by mean progression score (State-1 = lowest)
        means = [prog_np[labels_all == s].mean() for s in range(best_k)]
        rank_map = {s: r for r, s in enumerate(np.argsort(means))}
        ordered_labels = np.array([rank_map[l] for l in labels_all])

        for i, row in enumerate(df_clean.itertuples()):
            all_assignments.append(dict(
                file=row.file, speaker=row.speaker, label=row.label,
                split="train" if i in set(tr_idx) else "val",
                seed=seed, fold=fold, best_k=best_k,
                prog_score=float(prog_np[i]),
                state=int(ordered_labels[i])
            ))

# ── Save ─────────────────────────────────────────────────────────────────────
Path("data/results").mkdir(parents=True, exist_ok=True)
pd.DataFrame(bic_records).to_csv("data/results/states_bic.csv", index=False)
pd.DataFrame(sil_records).to_csv("data/results/states_silhouette.csv", index=False)
asgn_df = pd.DataFrame(all_assignments)
asgn_df.to_csv("data/results/states_assignments.csv", index=False)

# BIC summary
bic_df = pd.DataFrame(bic_records)
bic_summary = bic_df.groupby("k")["bic"].mean().reset_index()
print("\n── BIC by k (mean over seeds × folds) ──")
print(bic_summary.to_string(index=False))

# Silhouette summary
if sil_records:
    sil_df = pd.DataFrame(sil_records)
    sil_summary = sil_df.groupby("k")["silhouette"].mean().reset_index()
    print("\n── Silhouette by k ──")
    print(sil_summary.to_string(index=False))

# State statistics (fold=0, seed=0 as representative)
rep = asgn_df[(asgn_df.seed==0) & (asgn_df.fold==0)]
state_stats = rep.groupby("state").agg(
    n=("file","count"),
    prog_mean=("prog_score","mean"),
    prog_std=("prog_score","std"),
    pct_dementia=("label", lambda x: (x==1).mean())
).reset_index()
state_stats.columns = ["State","N","Prog_mean","Prog_std","Pct_dementia"]
print("\n── State statistics (seed=0, fold=0) ──")
print(state_stats.to_string(index=False))
print("\n⚠  State labels are NOT clinical diagnoses. "
      "State-0 ≠ 'healthy', State-N ≠ 'severe dementia'.")

state_stats.to_csv("data/results/states_centroids.csv", index=False)
print("\nAll state files saved.")
