"""Screen handcrafted features, then compute within-person delta features.

Steps
-----
1. Load features_acoustic.csv + features_linguistic.csv + manifest (with split).
2. Identify TRAINING speakers (split == 'train').
3. Quality covariates from quality.csv: snr_db, loudness_lufs, peak_dbfs,
   bandwidth_99_hz, duration_s.
4. Feature screening (fit on TRAIN only):
   a. Drop any feature with |Spearman| > 0.5 against quality covariates
      (conservative: 0.9 is too lenient for our shortcut-prone data).
   b. Drop one of any pair with |Spearman| > 0.9 (keep lower-index feature).
5. Save screened feature list to data/processed/feature_list.json.
6. Within-person delta: for speakers with ≥ 2 defensibly ordered recordings
   (from recording_order.csv), subtract earliest recording's feature vector.
7. Save data/processed/features_delta.csv.

Outputs:
  data/processed/feature_list.json        – list of retained feature names
  data/processed/features_screened.csv    – all recordings, screened features only
  data/processed/features_delta.csv       – delta features (speakers with ≥ 2 ordered recs)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

QUALITY_COLS = ["snr_db", "loudness_dbfs", "peak", "bw99_hz", "duration_s"]
SPEARMAN_QUALITY_THRESH = 0.5   # drop if |corr| with any quality covariate > this
SPEARMAN_FEAT_THRESH    = 0.9   # drop one of any feature pair with |corr| > this

# ── Load ───────────────────────────────────────────────────────────────────────
manifest = pd.read_csv("data/processed/manifest.csv")
quality  = pd.read_csv("data/processed/quality.csv")
feat_a   = pd.read_csv("data/processed/features_acoustic.csv")
feat_l   = pd.read_csv("data/processed/features_linguistic.csv")
order    = pd.read_csv("data/processed/recording_order.csv")

# Merge everything on 'file'
df = (manifest[["file", "speaker", "label", "split"]]
      .merge(quality[["file"] + QUALITY_COLS], on="file", how="left")
      .merge(feat_a, on="file", how="left")
      .merge(feat_l, on="file", how="left"))

feat_cols = [c for c in df.columns if c not in
             ["file", "speaker", "group", "split"] + QUALITY_COLS]

print(f"Total features before screening: {len(feat_cols)}")
print(f"Total recordings: {len(df)}")

# ── Split ──────────────────────────────────────────────────────────────────────
train_mask = df["split"] == "train"
df_train   = df[train_mask]
print(f"Train recordings: {train_mask.sum()}")

# ── Quality-correlation screening (fit on train) ───────────────────────────────
to_drop_quality = set()
for fc in feat_cols:
    for qc in QUALITY_COLS:
        x = df_train[[fc, qc]].dropna()
        if len(x) < 10:
            continue
        r, _ = spearmanr(x[fc], x[qc])
        if abs(r) > SPEARMAN_QUALITY_THRESH:
            to_drop_quality.add(fc)
            break  # no need to check other quality cols

print(f"\nDropped for quality correlation (|Spearman| > {SPEARMAN_QUALITY_THRESH}): "
      f"{len(to_drop_quality)}")
print(sorted(to_drop_quality))

remaining = [c for c in feat_cols if c not in to_drop_quality]

# ── Inter-feature collinearity screening (fit on train) ───────────────────────
X_train = df_train[remaining].values.astype(float)
n_feats = len(remaining)

# Compute pairwise Spearman on training set
# (use numpy/scipy ranking for speed on ~275 samples)
from scipy.stats import rankdata
ranks = np.apply_along_axis(rankdata, 0, np.nan_to_num(X_train, nan=0.0))
corr_mat = np.corrcoef(ranks.T)
np.fill_diagonal(corr_mat, 0)

to_drop_collinear = set()
for i in range(n_feats):
    if remaining[i] in to_drop_collinear:
        continue
    for j in range(i + 1, n_feats):
        if remaining[j] in to_drop_collinear:
            continue
        if abs(corr_mat[i, j]) > SPEARMAN_FEAT_THRESH:
            to_drop_collinear.add(remaining[j])  # drop higher-index (j)

print(f"\nDropped for inter-feature collinearity (|Spearman| > {SPEARMAN_FEAT_THRESH}): "
      f"{len(to_drop_collinear)}")
print(sorted(to_drop_collinear))

final_feats = [c for c in remaining if c not in to_drop_collinear]
print(f"\nRetained features: {len(final_feats)}")
print(final_feats)

# ── Save screened feature list & screened features ────────────────────────────
Path("data/processed").mkdir(parents=True, exist_ok=True)
json.dump(final_feats, open("data/processed/feature_list.json", "w"), indent=2)
df[["file", "speaker", "label", "split"] + final_feats].to_csv(
    "data/processed/features_screened.csv", index=False)

print("\nSaved: data/processed/feature_list.json")
print("Saved: data/processed/features_screened.csv")

# ── Within-person delta features ──────────────────────────────────────────────
# Only speakers with ≥ 2 recordings with DISTINCT ordered bucket ranks
# recording_order.csv uses column 'recording' (not 'file')
order_cols = ["recording", "speaker", "rank"]
ord_df = order[order_cols].copy().rename(columns={"recording": "file"})
ord_df = ord_df.merge(df[["file"] + final_feats], on="file", how="inner")

# Keep only rows where rank is not NaN (defensible ordering)
ord_df = ord_df.dropna(subset=["rank"])
ord_df["rank"] = ord_df["rank"].astype(int)

# For each speaker: subtract earliest-ranked recording's features
delta_rows = []
for spk, grp in ord_df.groupby("speaker"):
    if grp["rank"].nunique() < 2:
        continue
    grp = grp.sort_values("rank")
    earliest = grp.iloc[0][final_feats].values.astype(float)
    for _, row in grp.iloc[1:].iterrows():
        delta = row[final_feats].values.astype(float) - earliest
        rec = {"file": row["file"], "speaker": spk,
               "rank": row["rank"], "ref_rank": grp.iloc[0]["rank"]}
        for fn, dv in zip(final_feats, delta):
            rec[f"delta_{fn}"] = dv
        delta_rows.append(rec)

delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv("data/processed/features_delta.csv", index=False)
print(f"\nWithin-person delta: {len(delta_rows)} recording pairs "
      f"across {delta_df['speaker'].nunique()} speakers")
print("Saved: data/processed/features_delta.csv")
