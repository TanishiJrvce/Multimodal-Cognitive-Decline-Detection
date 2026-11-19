"""Final evaluation: speaker-level metrics, subgroup analysis, SHAP explanations.

Outputs:
  data/results/eval_speaker.csv       – per-speaker AUC/BAC (best checkpoint)
  data/results/eval_subgroup.csv      – AUC/BAC by gender, ethnicity, subtype, quality tertile
  data/results/eval_pair_accuracy.csv – temporal ordering accuracy (held-out pairs)
  data/results/shap_values.csv        – SHAP values for handcrafted feature baseline
  figures/shap_beeswarm.png           – SHAP beeswarm plot
  figures/trajectory_<speaker>.png   – per-speaker progression score over time (selected speakers)
"""
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import torch

import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mcd.evaluate import speaker_metrics, pair_accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS  = [0, 1, 2, 3, 4]
N_FOLDS = 5
Path("data/results").mkdir(parents=True, exist_ok=True)
Path("figures").mkdir(parents=True, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
feat_list = json.load(open("data/processed/feature_list.json"))
df = pd.read_csv("data/processed/features_screened.csv")
# 'label' is already 0/1 int
quality = pd.read_csv("data/processed/quality.csv")
manifest = pd.read_csv("data/processed/manifest.csv")

# Merge subgroup info
df = df.merge(manifest[["file","gender","ethnicity","dementia type"]],
              on="file", how="left", suffixes=("","_m"))
df = df.merge(quality[["file","snr_db","loudness_dbfs"]],
              on="file", how="left", suffixes=("","_q"))
df["quality_tertile"] = pd.qcut(df["snr_db"].fillna(df["snr_db"].median()),
                                q=3, labels=["low","mid","high"])
df_clean = df.dropna(subset=feat_list).copy()
X = df_clean[feat_list].values.astype(float)
y = df_clean["label"].values
groups = df_clean["speaker"].values

# ── 1. SHAP on best handcrafted baseline (RF, seed=0) ─────────────────────────
print("Computing SHAP values on RF baseline …")
cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
tr_idx, va_idx = next(cv.split(X, y, groups))

rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1)
rf.fit(X[tr_idx], y[tr_idx])

explainer = shap.TreeExplainer(rf)
shap_vals = explainer.shap_values(X[va_idx])
# shap 0.52 returns ndarray (samples, features, classes) or list; normalise to (samples, features) for class 1
if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
    sv = shap_vals[:, :, 1]
elif isinstance(shap_vals, list):
    sv = shap_vals[1]
else:
    sv = shap_vals

shap_df = pd.DataFrame(sv, columns=feat_list)
shap_df["file"] = df_clean["file"].values[va_idx]
shap_df.to_csv("data/results/shap_values.csv", index=False)

# Beeswarm plot
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X[va_idx], feature_names=feat_list,
                  show=False, max_display=20)
plt.tight_layout()
plt.savefig("figures/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP beeswarm saved → figures/shap_beeswarm.png")

# ── 2. Speaker-level evaluation on V_joint ─────────────────────────────────────
print("\nComputing speaker-level metrics on V_joint …")
from mcd.models import FusionNet

emb_idx  = pd.read_csv("data/processed/embeddings_index.csv")
emb_wav  = np.load("data/processed/embeddings_wavlm.npy")
emb_bert = np.load("data/processed/embeddings_bert.npy")
df_clean2 = df_clean.merge(emb_idx, on="file", how="left")
df_clean2["row_index"] = df_clean2["row_index"].fillna(-1).astype(int)
X_feat   = df_clean2[feat_list].values.astype(float)
X_wavlm  = np.where(df_clean2["row_index"].values[:,None]>=0,
                    emb_wav[np.clip(df_clean2["row_index"].values,0,len(emb_wav)-1)], 0.0)
X_bert   = np.where(df_clean2["row_index"].values[:,None]>=0,
                    emb_bert[np.clip(df_clean2["row_index"].values,0,len(emb_bert)-1)], 0.0)

speaker_records = []
for seed in SEEDS:
    cv2 = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx2, va_idx2) in enumerate(cv2.split(X_feat, y, groups)):
        ckpt = f"data/models/best_V_joint_s{seed}_f{fold}.pt"
        if not Path(ckpt).exists():
            continue
        sc2 = StandardScaler()
        X_f_tr = sc2.fit_transform(X_feat[tr_idx2])
        X_f_va = sc2.transform(X_feat[va_idx2])
        model = FusionNet(feat_dim=X_f_tr.shape[1], wavlm_dim=768, bert_dim=768).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            xf = torch.tensor(X_f_va, dtype=torch.float32, device=DEVICE)
            xw = torch.tensor(X_wavlm[va_idx2], dtype=torch.float32, device=DEVICE)
            xb = torch.tensor(X_bert[va_idx2], dtype=torch.float32, device=DEVICE)
            logit, prog = model(xf, xw, xb)
            probs = torch.sigmoid(logit).cpu().numpy()
            prog_np = prog.cpu().numpy()

        y_va = y[va_idx2]
        spk_va = groups[va_idx2]
        files_va = df_clean2["file"].values[va_idx2]

        # Speaker-level aggregation: mean probability per speaker
        spk_df = pd.DataFrame({"speaker": spk_va, "prob": probs,
                               "prog": prog_np, "label": y_va, "file": files_va})
        spk_agg = spk_df.groupby("speaker").agg(
            prob_mean=("prob","mean"), prog_mean=("prog","mean"),
            label=("label","first")).reset_index()
        auc_spk = roc_auc_score(spk_agg["label"], spk_agg["prob_mean"]) \
                  if len(spk_agg["label"].unique()) > 1 else np.nan
        bac_spk = balanced_accuracy_score(spk_agg["label"],
                                           (spk_agg["prob_mean"]>0.5).astype(int))
        speaker_records.append(dict(seed=seed, fold=fold,
                                    n_speakers=len(spk_agg),
                                    auc_speaker=auc_spk, bac_speaker=bac_spk))

spk_res = pd.DataFrame(speaker_records)
spk_res.to_csv("data/results/eval_speaker.csv", index=False)
print(f"  Speaker-level AUC: {spk_res['auc_speaker'].mean():.4f} ± "
      f"{spk_res['auc_speaker'].std():.4f}")

# ── 3. Temporal pair accuracy ──────────────────────────────────────────────────
print("\nComputing temporal pair accuracy …")
order_df = pd.read_csv("data/processed/recording_order.csv").dropna(subset=["rank"])
order_df = order_df.rename(columns={"recording": "file"})  # align with features/assignments naming
# Use progression scores from V_joint fold=0, seed=0
asgn = pd.read_csv("data/results/states_assignments.csv")
asgn_rep = asgn[(asgn.seed==0)&(asgn.fold==0)][["file","prog_score"]]
ord_prog = order_df.merge(asgn_rep, on="file", how="inner")

correct, total = 0, 0
for spk, grp in ord_prog.groupby("speaker"):
    if grp["rank"].nunique() < 2:
        continue
    grp = grp.sort_values("rank")
    # check that prog_score increases (weakly) with rank
    scores = grp["prog_score"].values
    for i in range(len(scores)-1):
        total += 1
        if scores[i] <= scores[i+1]:
            correct += 1

pair_acc = correct / total if total > 0 else np.nan
pd.DataFrame([{"pair_accuracy": pair_acc, "n_pairs": total,
               "n_correct": correct}]).to_csv(
    "data/results/eval_pair_accuracy.csv", index=False)
print(f"  Temporal pair accuracy: {pair_acc:.3f} ({correct}/{total} pairs)")
print("  ⚠ This is an internal consistency check, NOT independent clinical validation.")

# ── 4. Subgroup analysis (from RF baseline, fold=0 seed=0) ────────────────────
print("\nSubgroup analysis …")
sub_records = []
cv3 = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
tr3, va3 = next(cv3.split(X, y, groups))
rf3 = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1)
rf3.fit(X[tr3], y[tr3])
probs3 = rf3.predict_proba(X[va3])[:, 1]
preds3 = (probs3 > 0.5).astype(int)
meta_va = df_clean.iloc[va3][["gender","ethnicity","dementia type","quality_tertile","label"]].copy()
meta_va = meta_va.rename(columns={"dementia type": "dementia_type"})
meta_va["prob"] = probs3
meta_va["pred"] = preds3

for col in ["gender","ethnicity","quality_tertile"]:
    for val, grp2 in meta_va.groupby(col):
        if len(grp2["label"].unique()) < 2 or len(grp2) < 5:
            continue
        auc_g = roc_auc_score(grp2["label"], grp2["prob"])
        bac_g = balanced_accuracy_score(grp2["label"], grp2["pred"])
        sub_records.append(dict(subgroup_col=col, subgroup_val=val,
                                n=len(grp2), auc=auc_g, bac=bac_g))

sub_df = pd.DataFrame(sub_records)
sub_df.to_csv("data/results/eval_subgroup.csv", index=False)
print(sub_df.to_string(index=False))

# ── 5. Trajectory plots (selected speakers with ≥2 ordered recordings) ─────────
print("\nGenerating trajectory plots …")
asgn_all = pd.read_csv("data/results/states_assignments.csv")
asgn_rep2 = asgn_all[(asgn_all.seed==0)&(asgn_all.fold==0)]
traj = order_df.merge(asgn_rep2[["file","prog_score"]], on="file", how="inner")
traj = traj.sort_values(["speaker","rank"])

BUCKET_LABEL = {0:"~0y", 5:"~5y", 10:"~10y", 15:"~15y"}
n_plotted = 0
for spk, grp3 in traj.groupby("speaker"):
    if grp3["rank"].nunique() < 2:
        continue
    if n_plotted >= 6:  # only first 6 speakers with trajectories
        break
    fig, ax = plt.subplots(figsize=(5, 3))
    x = grp3["rank"].values  # use rank (1,2,3...) as x-axis since suffix_bucket not in order_df
    ax.plot(x, grp3["prog_score"].values, "o-", linewidth=2)
    ax.set_xlabel("Recording order (1 = earliest)")
    ax.set_ylabel("Progression score")
    ax.set_title(f"Speaker {spk}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.invert_xaxis()  # earlier time on left
    plt.tight_layout()
    fname = f"figures/trajectory_{spk}.png"
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()
    n_plotted += 1
print(f"  {n_plotted} trajectory plots saved → figures/")

print("\n✓ Evaluation complete. See data/results/ and figures/")
