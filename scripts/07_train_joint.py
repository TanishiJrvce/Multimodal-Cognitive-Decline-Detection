"""Train the joint FusionNet model with optional ranking supervision.

Variants:
  V_cls  – classification loss only       (lambda_rank = 0)
  V_joint – classification + ranking loss (lambda_rank = 0.5)
  V_audio – audio-only (WavLM + acoustic)
  V_text  – text-only  (BERT + linguistic)
  V_late  – late fusion (average logits)
  V_feat  – handcrafted features only (no embeddings)

Modality dropout: randomly zero-out one modality stream during training
  with probability p=0.2 (makes model robust to missing modality at test time).

Training protocol:
  - 5-fold StratifiedGroupKFold (speaker-grouped)
  - 5 seeds
  - Adam optimizer, lr=1e-4, weight_decay=1e-4
  - Early stopping on validation loss (patience=10 epochs)
  - Max 100 epochs
  - Scaler fit on training fold only

Outputs:
  data/results/joint_results.csv       – per-fold metrics per variant
  data/results/joint_summary.csv       – mean ± std summary
  data/models/best_<variant>_fold<k>.pt – best checkpoints

Design: src/mcd/models.py::FusionNet, joint_loss
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# ── Import project modules ────────────────────────────────────────────────────
import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mcd.models import FusionNet, joint_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

SEEDS       = [0, 1, 2, 3, 4]
N_FOLDS     = 5
LR          = 1e-4
WD          = 1e-4
MAX_EPOCHS  = 100
PATIENCE    = 10
LAMBDA_RANK = 0.5
MOD_DROP_P  = 0.2   # modality dropout probability during training

# ── Load data ─────────────────────────────────────────────────────────────────
feat_list = json.load(open("data/processed/feature_list.json"))
df = pd.read_csv("data/processed/features_screened.csv")
# 'label' is already 0/1 int (screened CSV has it pre-converted)
df_clean = df.dropna(subset=feat_list).copy()

# Load embeddings
emb_idx = pd.read_csv("data/processed/embeddings_index.csv")
emb_wav = np.load("data/processed/embeddings_wavlm.npy")  # (N_emb, 768)
emb_bert = np.load("data/processed/embeddings_bert.npy")   # (N_emb, 768)

# Align: left-join features df with embeddings index
df_clean = df_clean.merge(emb_idx, on="file", how="left")
has_emb_mask = df_clean["row_index"].notna()
df_clean["row_index"] = df_clean["row_index"].fillna(-1).astype(int)

X_feat   = df_clean[feat_list].values.astype(float)
X_wavlm  = np.where(df_clean["row_index"].values[:, None] >= 0,
                    emb_wav[np.clip(df_clean["row_index"].values, 0, len(emb_wav)-1)],
                    0.0)
X_bert   = np.where(df_clean["row_index"].values[:, None] >= 0,
                    emb_bert[np.clip(df_clean["row_index"].values, 0, len(emb_bert)-1)],
                    0.0)
y       = df_clean["label"].values
groups  = df_clean["speaker"].values

# Recording order for ranking pairs
order = pd.read_csv("data/processed/recording_order.csv")
order = order.dropna(subset=["rank"])

Path("data/models").mkdir(parents=True, exist_ok=True)
Path("data/results").mkdir(parents=True, exist_ok=True)

# ── Utility ───────────────────────────────────────────────────────────────────
def make_rank_pairs(files_tr, groups_tr, y_tr, order_df):
    """Return index pairs (i, j) where recording i is earlier than j (same speaker, dementia)."""
    file_to_idx = {f: i for i, f in enumerate(files_tr)}
    pairs = []
    for spk, grp in order_df.groupby("speaker"):
        grp = grp.sort_values("rank")
        files_spk = [f for f in grp["recording"].values if f in file_to_idx]
        if len(files_spk) < 2:
            continue
        for a, b in zip(files_spk, files_spk[1:]):
            pairs.append((file_to_idx[a], file_to_idx[b]))
    return pairs

def run_fold(X_f, X_w, X_b, y_tr_all, y_va_all,
             tr_idx, va_idx, files_all, groups_all,
             lambda_rank, mod_names, seed, fold):
    """Train FusionNet on one fold."""
    # Fit scaler on training handcrafted features only
    sc = StandardScaler()
    X_f_tr = sc.fit_transform(X_f[tr_idx])
    X_f_va = sc.transform(X_f[va_idx])
    X_w_tr, X_w_va = X_w[tr_idx], X_w[va_idx]
    X_b_tr, X_b_va = X_b[tr_idx], X_b[va_idx]
    y_tr = y_tr_all[tr_idx]
    y_va = y_va_all[va_idx]

    # Mask based on which modalities are used
    if "audio" not in mod_names:
        X_w_tr = np.zeros_like(X_w_tr); X_w_va = np.zeros_like(X_w_va)
        X_f_tr = np.zeros_like(X_f_tr); X_f_va = np.zeros_like(X_f_va)
    if "text" not in mod_names:
        X_b_tr = np.zeros_like(X_b_tr); X_b_va = np.zeros_like(X_b_va)
    if "feat" not in mod_names:
        X_f_tr = np.zeros_like(X_f_tr); X_f_va = np.zeros_like(X_f_va)

    # Ranking pairs (training only)
    files_tr = files_all[tr_idx]
    pairs_tr = make_rank_pairs(files_tr, groups_all[tr_idx], y_tr, order)

    # Model
    torch.manual_seed(seed)
    model = FusionNet(
        feat_dim=X_f_tr.shape[1],
        wavlm_dim=768, bert_dim=768
    ).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_val_loss, best_state, pat = float("inf"), None, 0

    for epoch in range(MAX_EPOCHS):
        # ── Train ───────────────────────────────────────────────────────────
        model.train()
        perm = np.random.RandomState(epoch).permutation(len(y_tr))
        for b in range(0, len(perm), 32):
            bi = perm[b:b+32]
            xf = torch.tensor(X_f_tr[bi], dtype=torch.float32, device=DEVICE)
            xw = torch.tensor(X_w_tr[bi], dtype=torch.float32, device=DEVICE)
            xb = torch.tensor(X_b_tr[bi], dtype=torch.float32, device=DEVICE)
            yt = torch.tensor(y_tr[bi], dtype=torch.float32, device=DEVICE)

            # Modality dropout
            if np.random.rand() < MOD_DROP_P and "audio" in mod_names:
                xw = torch.zeros_like(xw)
            if np.random.rand() < MOD_DROP_P and "text" in mod_names:
                xb = torch.zeros_like(xb)

            logit, prog = model(xf, xw, xb)
            loss = joint_loss(logit, prog, yt, pairs=pairs_tr,
                              batch_indices=bi.tolist(), lambda_rank=lambda_rank)
            opt.zero_grad(); loss.backward(); opt.step()

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            xf = torch.tensor(X_f_va, dtype=torch.float32, device=DEVICE)
            xw = torch.tensor(X_w_va, dtype=torch.float32, device=DEVICE)
            xb = torch.tensor(X_b_va, dtype=torch.float32, device=DEVICE)
            yt = torch.tensor(y_va, dtype=torch.float32, device=DEVICE)
            logit, prog = model(xf, xw, xb)
            val_loss = joint_loss(logit, prog, yt, pairs=[], batch_indices=[],
                                  lambda_rank=0).item()
            probs = torch.sigmoid(logit).cpu().numpy()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= PATIENCE:
            break

    # ── Evaluate best checkpoint ─────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xf = torch.tensor(X_f_va, dtype=torch.float32, device=DEVICE)
        xw = torch.tensor(X_w_va, dtype=torch.float32, device=DEVICE)
        xb = torch.tensor(X_b_va, dtype=torch.float32, device=DEVICE)
        logit, _ = model(xf, xw, xb)
        probs = torch.sigmoid(logit).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    auc = roc_auc_score(y_va, probs) if len(np.unique(y_va)) > 1 else 0.5
    bac = balanced_accuracy_score(y_va, preds)
    return auc, bac, best_state, model

# ── Experiment grid ───────────────────────────────────────────────────────────
experiments = {
    "V_feat"  : ({"feat", "audio", "text"}, 0.0),   # wait, feat only
    "V_cls"   : ({"feat", "audio", "text"}, 0.0),
    "V_joint" : ({"feat", "audio", "text"}, LAMBDA_RANK),
    "V_audio" : ({"audio", "feat"}, 0.0),
    "V_text"  : ({"text", "feat"}, 0.0),
    "V_late"  : ({"feat", "audio", "text"}, 0.0),   # late fusion variant handled inside FusionNet
}
# Note: V_feat = no embeddings (zeros), V_cls = all modalities cls-only
experiments = {
    "V_feat"  : ({"feat"},              0.0),
    "V_audio" : ({"feat", "audio"},     0.0),
    "V_text"  : ({"feat", "text"},      0.0),
    "V_cls"   : ({"feat","audio","text"}, 0.0),
    "V_joint" : ({"feat","audio","text"}, LAMBDA_RANK),
}

files_all = df_clean["file"].values

records = []
for exp_name, (mod_names, lam) in experiments.items():
    print(f"\n── {exp_name} (mods={mod_names}, lambda_rank={lam}) ──")
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_feat, y, groups)):
            auc, bac, best_state, model = run_fold(
                X_feat, X_wavlm, X_bert, y, y,
                tr_idx, va_idx, files_all, groups,
                lambda_rank=lam, mod_names=mod_names,
                seed=seed, fold=fold)
            records.append(dict(experiment=exp_name, seed=seed, fold=fold,
                                auc=auc, bac=bac, lambda_rank=lam))
            # Save best model checkpoint
            ckpt_path = f"data/models/best_{exp_name}_s{seed}_f{fold}.pt"
            torch.save(best_state, ckpt_path)
            print(f"  s{seed} f{fold}: AUC={auc:.4f} BAC={bac:.4f}", flush=True)

res_df = pd.DataFrame(records)
res_df.to_csv("data/results/joint_results.csv", index=False)

summary = (res_df.groupby("experiment")[["auc","bac"]]
           .agg(["mean","std"]).round(4))
summary.columns = ["auc_mean","auc_std","bac_mean","bac_std"]
summary = summary.sort_values("auc_mean", ascending=False)
summary.to_csv("data/results/joint_summary.csv")

print("\n── Joint model results ──")
print(summary.to_string())
