"""Train the joint FusionNet model with optional ranking supervision.  [V1-corrected]

V1 CORRECTIONS applied here:
  1. Modality masking: each stream zeroed ONLY by its own key (via apply_modality_mask).
     - V_feat: handcrafted only (no WavLM, no BERT)
     - V_audio: WavLM + handcrafted (no BERT)
     - V_text: BERT + handcrafted (no WavLM)
     - V_cls: all streams, classification loss only
     - V_joint: all streams, classification + ranking
  2. Fold-local feature screening: quality-correlation and collinearity screening
     is fitted exclusively on each fold's outer-training speakers, not globally.
  3. Inner validation split for epoch/checkpoint selection (20% of outer-train,
     stratified, speaker-grouped). After epoch count is selected, model refits on
     full outer-train for the same number of epochs, then evaluates outer-test once.
  4. Rank-pair safety: same-bucket pairs (rank[a] == rank[b]) are excluded.
     Pair counts and gradient contributions are logged.
  5. Preprocessing (StandardScaler) fitted only on outer-train speakers.
  6. Per-recording outer-held-out predictions saved with full metadata.
  7. GMM k-selection: forced minimum of 2 removed in 08_states.py (see that script).

Variants:
  V_feat  – handcrafted features only; no WavLM, no BERT
  V_audio – WavLM + handcrafted; no BERT
  V_text  – BERT + handcrafted; no WavLM
  V_cls   – all streams, classification objective only
  V_joint – all streams, classification + ranking

Training protocol:
  - 5-fold StratifiedGroupKFold (speaker-grouped)
  - 5 seeds
  - Inner validation: StratifiedGroupKFold(2) on outer-train (≈ 20% inner-val)
  - Adam optimizer, lr=1e-4, weight_decay=1e-4
  - Early stopping on inner-val loss (patience=10 epochs)
  - After epoch selection: refit on full outer-train, train for selected epochs
  - Evaluate outer-test speakers once (never used during training)
  - Max 100 epochs

Outputs:
  data/results/joint_results_v1.csv       – per-fold metrics per variant
  data/results/joint_summary_v1.csv       – mean ± std summary
  data/results/joint_predictions_v1.csv   – per-recording held-out predictions
  data/results/joint_fold_features_v1/    – fold-specific selected feature lists
  data/models/v1/best_<variant>_s<s>_f<f>.pt – best checkpoints
  data/results/joint_provenance_v1.json   – run configuration, versions, seeds
"""
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import platform, sys, importlib

# ── Import project modules ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mcd.models import FusionNet, joint_loss, apply_modality_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS       = [0, 1, 2, 3, 4]
N_FOLDS     = 5
LR          = 1e-4
WD          = 1e-4
MAX_EPOCHS  = 100
PATIENCE    = 10
LAMBDA_RANK = 0.5
MOD_DROP_P  = 0.2   # modality dropout during training

# V1: quality threshold for fold-local screening (same thresholds as 05b)
SPEARMAN_QUALITY_THRESH = 0.5
SPEARMAN_FEAT_THRESH    = 0.9
QUALITY_COLS = ["snr_db", "loudness_dbfs", "peak", "bw99_hz", "duration_s"]

Path("data/models/v1").mkdir(parents=True, exist_ok=True)
Path("data/results").mkdir(parents=True, exist_ok=True)
Path("data/results/joint_fold_features_v1").mkdir(parents=True, exist_ok=True)

# ── Load all candidate features (before screening) ───────────────────────────
# V1 FIX: load full unscreened columns, then screen inside each fold.
# We load from the merged acoustic+linguistic+quality table.
manifest = pd.read_csv("data/processed/manifest.csv")
quality  = pd.read_csv("data/processed/quality.csv")
feat_a   = pd.read_csv("data/processed/features_acoustic.csv")
feat_l   = pd.read_csv("data/processed/features_linguistic.csv")
order    = pd.read_csv("data/processed/recording_order.csv")

# Merge to full table
df_full = (manifest[["file", "speaker", "label", "split"]]
           .merge(quality[["file"] + QUALITY_COLS], on="file", how="left")
           .merge(feat_a, on="file", how="left")
           .merge(feat_l, on="file", how="left"))

# Convert label to 0/1
df_full["label"] = (df_full["label"] == "dementia").astype(int) \
                   if df_full["label"].dtype == object else df_full["label"].astype(int)

# All candidate feature columns (exclude metadata, quality, label)
CANDIDATE_FEATS = [c for c in df_full.columns
                   if c not in ["file", "speaker", "label", "split"] + QUALITY_COLS]
print(f"Candidate features before fold-local screening: {len(CANDIDATE_FEATS)}")

df_full = df_full.dropna(subset=["speaker", "label"]).reset_index(drop=True)

# Load embeddings
emb_idx  = pd.read_csv("data/processed/embeddings_index.csv")
emb_wav  = np.load("data/processed/embeddings_wavlm.npy")  # (N_emb, 768)
emb_bert = np.load("data/processed/embeddings_bert.npy")   # (N_emb, 768)

# Align: left-join features df with embeddings index
df_full = df_full.merge(emb_idx, on="file", how="left")
df_full["row_index"] = df_full["row_index"].fillna(-1).astype(int)

X_cand   = df_full[CANDIDATE_FEATS].values.astype(float)
X_wavlm  = np.where(df_full["row_index"].values[:, None] >= 0,
                    emb_wav[np.clip(df_full["row_index"].values, 0, len(emb_wav)-1)],
                    0.0)
X_bert   = np.where(df_full["row_index"].values[:, None] >= 0,
                    emb_bert[np.clip(df_full["row_index"].values, 0, len(emb_bert)-1)],
                    0.0)
y       = df_full["label"].values
groups  = df_full["speaker"].values
files_all = df_full["file"].values

# Recording order for ranking pairs
order = order.dropna(subset=["rank"])


# ── V1: Fold-local feature screening ─────────────────────────────────────────
def screen_features_on_fold(tr_idx, candidate_cols, X_full, quality_full):
    """Fit feature screening on outer-training rows only. Returns retained column indices."""
    X_tr = X_full[tr_idx]
    q_tr = quality_full[tr_idx]
    n = len(candidate_cols)

    # Step 1: quality correlation
    to_drop = set()
    for fi, fc in enumerate(candidate_cols):
        for qi in range(q_tr.shape[1]):
            x = X_tr[:, fi]; q = q_tr[:, qi]
            valid = np.isfinite(x) & np.isfinite(q)
            if valid.sum() < 10:
                continue
            r, _ = spearmanr(x[valid], q[valid])
            if abs(r) > SPEARMAN_QUALITY_THRESH:
                to_drop.add(fi)
                break

    remaining = [i for i in range(n) if i not in to_drop]

    # Step 2: inter-feature collinearity
    if len(remaining) > 1:
        X_rem = np.nan_to_num(X_tr[:, remaining], nan=0.0)
        ranks = np.apply_along_axis(rankdata, 0, X_rem)
        corr_mat = np.corrcoef(ranks.T)
        np.fill_diagonal(corr_mat, 0)
        to_drop_col = set()
        for a in range(len(remaining)):
            if a in to_drop_col:
                continue
            for b in range(a + 1, len(remaining)):
                if b in to_drop_col:
                    continue
                if abs(corr_mat[a, b]) > SPEARMAN_FEAT_THRESH:
                    to_drop_col.add(b)
        remaining = [remaining[i] for i in range(len(remaining)) if i not in to_drop_col]

    return remaining  # list of column indices into CANDIDATE_FEATS


X_quality = df_full[QUALITY_COLS].fillna(df_full[QUALITY_COLS].median()).values


# ── V1: Rank-pair builder (excludes same-bucket pairs) ───────────────────────
def make_rank_pairs_v1(files_tr, order_df):
    """Return index pairs (i, j) where recording i is strictly earlier than j.
    Excludes pairs from the same temporal bucket (rank[a] == rank[b]).
    Returns: list of (local_i, local_j), pair_count, same_bucket_excluded_count
    """
    file_to_idx = {f: i for i, f in enumerate(files_tr)}
    pairs = []
    excluded_same_bucket = 0
    for spk, grp in order_df.groupby("speaker"):
        grp = grp.sort_values("rank")
        recs = [(row["recording"], row["rank"])
                for _, row in grp.iterrows()
                if row["recording"] in file_to_idx]
        for k in range(len(recs)):
            for m in range(k + 1, len(recs)):
                fa, ra = recs[k]
                fb, rb = recs[m]
                if ra == rb:
                    excluded_same_bucket += 1
                    continue  # V1 FIX: skip same-bucket pairs
                # ra < rb guaranteed because grp is sorted by rank
                pairs.append((file_to_idx[fa], file_to_idx[fb]))
    return pairs, len(pairs), excluded_same_bucket


# ── Train one fold ────────────────────────────────────────────────────────────
def run_fold_v1(feat_idx, outer_tr_idx, outer_va_idx,
                mod_names, lambda_rank, seed, fold):
    """
    V1 protocol:
      1. Screen features on outer-train only → feat_idx subset
      2. Scale handcrafted on outer-train
      3. Carve inner split from outer-train for epoch selection
      4. Train on inner-train, select best epoch on inner-val
      5. Refit scaler + fresh model on full outer-train, train for selected epochs
      6. Evaluate on outer-val (held-out) ONCE
    """
    # --- Extract fold-screened features ---
    X_feat_all = X_cand[:, feat_idx]

    # --- Outer train/val split ---
    X_f_otr = X_feat_all[outer_tr_idx]
    X_f_ova = X_feat_all[outer_va_idx]
    X_w_otr = X_wavlm[outer_tr_idx]
    X_w_ova = X_wavlm[outer_va_idx]
    X_b_otr = X_bert[outer_tr_idx]
    X_b_ova = X_bert[outer_va_idx]
    y_otr = y[outer_tr_idx]
    y_ova = y[outer_va_idx]
    g_otr = groups[outer_tr_idx]

    # --- Inner split for epoch selection (StratifiedGroupKFold, 2 splits → ~20% inner val) ---
    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    # Use first inner split
    inner_tr_idx, inner_va_idx = next(inner_cv.split(X_f_otr, y_otr, g_otr))

    # Scaler for inner split (fit on inner-train only)
    sc_inner = StandardScaler()
    X_f_itr = sc_inner.fit_transform(X_f_otr[inner_tr_idx])
    X_f_iva = sc_inner.transform(X_f_otr[inner_va_idx])
    X_w_itr = X_w_otr[inner_tr_idx]; X_w_iva = X_w_otr[inner_va_idx]
    X_b_itr = X_b_otr[inner_tr_idx]; X_b_iva = X_b_otr[inner_va_idx]
    y_itr = y_otr[inner_tr_idx]; y_iva = y_otr[inner_va_idx]

    # Apply modality mask (V1 FIX: independent per stream)
    X_f_itr, X_w_itr, X_b_itr = apply_modality_mask(X_f_itr, X_w_itr, X_b_itr, mod_names)
    X_f_iva, X_w_iva, X_b_iva = apply_modality_mask(X_f_iva, X_w_iva, X_b_iva, mod_names)

    # Ranking pairs from inner-train only
    files_itr = files_all[outer_tr_idx][inner_tr_idx]
    pairs_itr, n_pairs, n_excl = make_rank_pairs_v1(files_itr, order)

    # --- Phase 1: train on inner-train, select epoch ---
    torch.manual_seed(seed)
    model = FusionNet(feat_dim=X_f_itr.shape[1], wavlm_dim=768, bert_dim=768).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_inner_val_loss = float("inf")
    best_epoch = 0
    pat = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = np.random.RandomState(epoch + seed * 1000).permutation(len(y_itr))
        for b in range(0, len(perm), 32):
            bi = perm[b:b+32]
            xf = torch.tensor(X_f_itr[bi], dtype=torch.float32, device=DEVICE)
            xw = torch.tensor(X_w_itr[bi], dtype=torch.float32, device=DEVICE)
            xb = torch.tensor(X_b_itr[bi], dtype=torch.float32, device=DEVICE)
            yt = torch.tensor(y_itr[bi],   dtype=torch.float32, device=DEVICE)
            # Modality dropout (only for included streams)
            if np.random.rand() < MOD_DROP_P and "audio" in mod_names:
                xw = torch.zeros_like(xw)
            if np.random.rand() < MOD_DROP_P and "text" in mod_names:
                xb = torch.zeros_like(xb)
            logit, prog = model(xf, xw, xb)
            loss = joint_loss(logit, prog, yt, pairs=pairs_itr,
                              batch_indices=bi.tolist(), lambda_rank=lambda_rank)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            xf = torch.tensor(X_f_iva, dtype=torch.float32, device=DEVICE)
            xw = torch.tensor(X_w_iva, dtype=torch.float32, device=DEVICE)
            xb = torch.tensor(X_b_iva, dtype=torch.float32, device=DEVICE)
            yt = torch.tensor(y_iva,   dtype=torch.float32, device=DEVICE)
            logit, prog = model(xf, xw, xb)
            inner_val_loss = joint_loss(logit, prog, yt, pairs=[], batch_indices=[],
                                        lambda_rank=0).item()
        if inner_val_loss < best_inner_val_loss:
            best_inner_val_loss = inner_val_loss
            best_epoch = epoch + 1
            pat = 0
        else:
            pat += 1
        if pat >= PATIENCE:
            break

    # --- Phase 2: refit on full outer-train for best_epoch epochs ---
    sc_full = StandardScaler()
    X_f_otr_sc = sc_full.fit_transform(X_f_otr)
    X_f_ova_sc = sc_full.transform(X_f_ova)
    X_w_otr_m, X_w_iva_m, X_b_otr_m = X_w_otr, X_w_ova, X_b_otr  # naming fix
    X_f_otr_m, X_w_otr_m, X_b_otr_m = apply_modality_mask(X_f_otr_sc, X_w_otr, X_b_otr, mod_names)
    X_f_ova_m, X_w_ova_m, X_b_ova_m = apply_modality_mask(X_f_ova_sc, X_w_ova, X_b_ova, mod_names)

    files_otr = files_all[outer_tr_idx]
    pairs_otr, n_pairs_otr, n_excl_otr = make_rank_pairs_v1(files_otr, order)

    torch.manual_seed(seed)
    model2 = FusionNet(feat_dim=X_f_otr_m.shape[1], wavlm_dim=768, bert_dim=768).to(DEVICE)
    opt2 = optim.Adam(model2.parameters(), lr=LR, weight_decay=WD)

    for epoch in range(best_epoch):
        model2.train()
        perm = np.random.RandomState(epoch + seed * 1000).permutation(len(y_otr))
        for b in range(0, len(perm), 32):
            bi = perm[b:b+32]
            xf = torch.tensor(X_f_otr_m[bi], dtype=torch.float32, device=DEVICE)
            xw = torch.tensor(X_w_otr_m[bi], dtype=torch.float32, device=DEVICE)
            xb = torch.tensor(X_b_otr_m[bi], dtype=torch.float32, device=DEVICE)
            yt = torch.tensor(y_otr[bi],     dtype=torch.float32, device=DEVICE)
            if np.random.rand() < MOD_DROP_P and "audio" in mod_names:
                xw = torch.zeros_like(xw)
            if np.random.rand() < MOD_DROP_P and "text" in mod_names:
                xb = torch.zeros_like(xb)
            logit, prog = model2(xf, xw, xb)
            loss = joint_loss(logit, prog, yt, pairs=pairs_otr,
                              batch_indices=bi.tolist(), lambda_rank=lambda_rank)
            opt2.zero_grad(); loss.backward(); opt2.step()

    # --- Evaluate on outer-val (once) ---
    model2.eval()
    with torch.no_grad():
        xf = torch.tensor(X_f_ova_m, dtype=torch.float32, device=DEVICE)
        xw = torch.tensor(X_w_ova_m, dtype=torch.float32, device=DEVICE)
        xb = torch.tensor(X_b_ova_m, dtype=torch.float32, device=DEVICE)
        logit, prog_out = model2(xf, xw, xb)
        probs = torch.sigmoid(logit).cpu().numpy()
        prog_np = prog_out.cpu().numpy()

    preds = (probs > 0.5).astype(int)
    auc = roc_auc_score(y_ova, probs) if len(np.unique(y_ova)) > 1 else 0.5
    bac = balanced_accuracy_score(y_ova, preds)

    # Save checkpoint
    ckpt_path = f"data/models/v1/best_{mod_names_key(mod_names)}_s{seed}_f{fold}.pt"
    torch.save({"state_dict": model2.state_dict(),
                "scaler_mean": sc_full.mean_.tolist(),
                "scaler_scale": sc_full.scale_.tolist(),
                "feat_indices": feat_idx,
                "feat_names": [CANDIDATE_FEATS[i] for i in feat_idx],
                "mod_names": list(mod_names),
                "best_epoch": best_epoch,
                "n_pairs": n_pairs_otr,
                "n_same_bucket_excluded": n_excl_otr}, ckpt_path)

    return auc, bac, probs, prog_np, best_epoch, n_pairs_otr, n_excl_otr, \
           [CANDIDATE_FEATS[i] for i in feat_idx], sc_full


def mod_names_key(mod_names):
    """Canonical experiment name from mod_names set."""
    for name, (mods, _) in experiments.items():
        if mods == mod_names:
            return name
    return "_".join(sorted(mod_names))


# ── Experiment grid (V1: explicit, correct stream assignments) ────────────────
experiments = {
    "V_feat"  : ({"feat"},               0.0),   # handcrafted only
    "V_audio" : ({"feat", "audio"},      0.0),   # WavLM + handcrafted
    "V_text"  : ({"feat", "text"},       0.0),   # BERT + handcrafted
    "V_cls"   : ({"feat", "audio", "text"}, 0.0),   # all, cls only
    "V_joint" : ({"feat", "audio", "text"}, LAMBDA_RANK),  # all, joint
}

records = []
pred_records = []

for exp_name, (mod_names, lam) in experiments.items():
    print(f"\n── {exp_name} (mods={sorted(mod_names)}, λ={lam}) ──")
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (outer_tr_idx, outer_va_idx) in enumerate(cv.split(X_cand, y, groups)):
            # V1: fold-local feature screening
            feat_idx = screen_features_on_fold(outer_tr_idx, CANDIDATE_FEATS,
                                               X_cand, X_quality)
            n_feats = len(feat_idx)
            feat_names = [CANDIDATE_FEATS[i] for i in feat_idx]

            # Save fold-specific feature list
            feat_path = f"data/results/joint_fold_features_v1/{exp_name}_s{seed}_f{fold}.json"
            json.dump(feat_names, open(feat_path, "w"), indent=2)

            auc, bac, probs, prog_np, best_epoch, n_pairs, n_excl, _, sc = run_fold_v1(
                feat_idx, outer_tr_idx, outer_va_idx,
                mod_names=mod_names, lambda_rank=lam,
                seed=seed, fold=fold)

            records.append(dict(experiment=exp_name, seed=seed, fold=fold,
                                auc=auc, bac=bac, lambda_rank=lam,
                                n_features=n_feats, best_epoch=best_epoch,
                                n_rank_pairs=n_pairs,
                                n_same_bucket_excluded=n_excl))

            # Save per-recording held-out predictions
            for i, va_i in enumerate(outer_va_idx):
                pred_records.append(dict(
                    experiment=exp_name, seed=seed, fold=fold,
                    file=files_all[va_i],
                    speaker=groups[va_i],
                    label=int(y[va_i]),
                    group_prob=float(probs[i]),
                    prog_score=float(prog_np[i]),
                    pred=(int(probs[i] > 0.5))))

            print(f"  s{seed} f{fold}: AUC={auc:.4f} BAC={bac:.4f} "
                  f"feats={n_feats} epoch={best_epoch} pairs={n_pairs}(excl={n_excl})",
                  flush=True)

res_df = pd.DataFrame(records)
res_df.to_csv("data/results/joint_results_v1.csv", index=False)

pred_df = pd.DataFrame(pred_records)
pred_df.to_csv("data/results/joint_predictions_v1.csv", index=False)

summary = (res_df.groupby("experiment")[["auc", "bac"]]
           .agg(["mean", "std"]).round(4))
summary.columns = ["auc_mean", "auc_std", "bac_mean", "bac_std"]
summary = summary.sort_values("auc_mean", ascending=False)
summary.to_csv("data/results/joint_summary_v1.csv")

# ── Speaker-level aggregation ─────────────────────────────────────────────────
spk_records = []
for exp_name in experiments:
    sub = pred_df[pred_df.experiment == exp_name]
    for (seed, fold), g in sub.groupby(["seed", "fold"]):
        spk = g.groupby("speaker").agg(
            prob_mean=("group_prob", "mean"),
            prog_mean=("prog_score", "mean"),
            label=("label", "first")).reset_index()
        auc_spk = roc_auc_score(spk["label"], spk["prob_mean"]) \
                  if spk["label"].nunique() > 1 else np.nan
        bac_spk = balanced_accuracy_score(spk["label"],
                                          (spk["prob_mean"] > 0.5).astype(int))
        spk_records.append(dict(experiment=exp_name, seed=seed, fold=fold,
                                n_speakers=len(spk),
                                auc_speaker=auc_spk, bac_speaker=bac_spk))
pd.DataFrame(spk_records).to_csv("data/results/eval_speaker_v1.csv", index=False)

# ── Provenance record ─────────────────────────────────────────────────────────
import torch as _torch, sklearn as _sk
prov = {
    "script": "07_train_joint_v1.py",
    "branch": "V1-corrected",
    "python": sys.version,
    "pytorch": _torch.__version__,
    "sklearn": _sk.__version__,
    "device": DEVICE,
    "seeds": SEEDS,
    "n_folds": N_FOLDS,
    "lr": LR, "wd": WD, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
    "lambda_rank": LAMBDA_RANK,
    "mod_drop_p": MOD_DROP_P,
    "quality_spearman_thresh": SPEARMAN_QUALITY_THRESH,
    "feat_spearman_thresh": SPEARMAN_FEAT_THRESH,
    "asr_model": "UNKNOWN — not stored in transcript JSON; config declares small.en, script defaults medium.en",
    "wavlm_model": "microsoft/wavlm-base-plus",
    "bert_model": "bert-base-uncased",
    "n_recordings": int(len(df_full)),
    "n_speakers": int(pd.Series(groups).nunique()),
    "v1_fixes": [
        "Modality masking decoupled: each stream zeroed only by its own key",
        "Fold-local feature screening: fitted on outer-train only",
        "Inner validation split for epoch selection",
        "Same-bucket rank pairs excluded from training",
        "Per-recording held-out predictions saved",
        "Checkpoint includes scaler, feature indices, pair counts"
    ]
}
json.dump(prov, open("data/results/joint_provenance_v1.json", "w"), indent=2)

print("\n── V1 Joint model results ──")
print(summary.to_string())
print("\nAll outputs saved to data/results/")
