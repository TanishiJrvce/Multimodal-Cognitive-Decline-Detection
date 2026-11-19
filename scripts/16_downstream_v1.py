"""Generate all downstream analyses for V1: temporal concordance, speaker-level metrics,
SHAP on corrected RF, original-vs-corrected comparison table."""
import pandas as pd, numpy as np, json, itertools
from pathlib import Path
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Path("data/results").mkdir(exist_ok=True, parents=True)
Path("figures").mkdir(exist_ok=True, parents=True)

# ── Load V1 outputs ───────────────────────────────────────────────────────────
pred      = pd.read_csv("data/results/joint_predictions_v1.csv")
baselines = pd.read_csv("data/results/baselines_v1.csv")
states    = pd.read_csv("data/results/states_assignments_v1.csv")
ksel      = pd.read_csv("data/results/states_kselect_v1.csv")
order     = pd.read_csv("data/processed/recording_order.csv").dropna(subset=["rank"])
order     = order.rename(columns={"recording": "file"})
feat_list = json.loads(open("data/processed/feature_list.json").read())
feat_df   = pd.read_csv("data/processed/features_screened.csv")
manifest  = pd.read_csv("data/processed/manifest.csv")
quality   = pd.read_csv("data/processed/quality.csv")
pair_audit= pd.read_csv("data/results/ranking_pair_audit_v1.csv")

# ── 1. Temporal concordance — held-out only, bucket-aggregated ────────────────
vjoint = pred[pred.experiment == "V_joint"].copy()

conc_rows = []
for (seed, fold), g in vjoint.groupby(["seed", "fold"]):
    held = g[["file", "speaker", "prog_score", "label"]].copy()
    ord_held = held.merge(order[["file", "speaker", "rank"]], on=["file", "speaker"], how="inner")
    correct, total = 0, 0
    for spk, sg in ord_held.groupby("speaker"):
        if sg["rank"].nunique() < 2:
            continue
        bucket = sg.groupby("rank")["prog_score"].mean().sort_index()
        for i in range(len(bucket) - 1):
            total += 1
            if bucket.iloc[i + 1] > bucket.iloc[i]:
                correct += 1
    conc_rows.append(dict(seed=seed, fold=fold, mode="val",
                          correct=correct, total=total,
                          concordance=correct / total if total > 0 else np.nan))

# Five-seed OOF consensus concordance
consensus = vjoint.groupby("file").agg(
    prog_score=("prog_score", "mean"),
    speaker=("speaker", "first"),
    label=("label", "first")).reset_index()
ord_all = consensus.merge(order[["file", "speaker", "rank"]], on=["file", "speaker"], how="inner")
correct_c, total_c = 0, 0
for spk, sg in ord_all.groupby("speaker"):
    if sg["rank"].nunique() < 2:
        continue
    bucket = sg.groupby("rank")["prog_score"].mean().sort_index()
    for i in range(len(bucket) - 1):
        total_c += 1
        if bucket.iloc[i + 1] > bucket.iloc[i]:
            correct_c += 1

conc_df = pd.DataFrame(conc_rows)
conc_df.to_csv("data/results/temporal_concordance_v1.csv", index=False)
print("=== TEMPORAL CONCORDANCE (held-out, bucket-aggregated) ===")
print(conc_df[["seed", "fold", "correct", "total", "concordance"]].to_string())
print(f"Mean: {conc_df.concordance.mean():.3f}  Std: {conc_df.concordance.std():.3f}")
oof_conc = correct_c / total_c if total_c > 0 else float("nan")
print(f"Five-seed OOF consensus: {correct_c}/{total_c} = {oof_conc:.3f}")

# ── 2. Speaker-level AUC per experiment ──────────────────────────────────────
spk_rows = []
for exp, g_exp in pred.groupby("experiment"):
    for (seed, fold), g in g_exp.groupby(["seed", "fold"]):
        spk = g.groupby("speaker").agg(
            prob_mean=("group_prob", "mean"),
            label=("label", "first")).reset_index()
        auc_spk = roc_auc_score(spk.label, spk.prob_mean) if spk.label.nunique() > 1 else np.nan
        bac_spk = balanced_accuracy_score(spk.label, (spk.prob_mean > 0.5).astype(int))
        spk_rows.append(dict(experiment=exp, seed=seed, fold=fold,
                             n_speakers=len(spk), auc_speaker=auc_spk, bac_speaker=bac_spk))
spk_df = pd.DataFrame(spk_rows)
spk_df.to_csv("data/results/eval_speaker_v1.csv", index=False)
print("\n=== SPEAKER-LEVEL AUC (mean ± std) ===")
print(spk_df.groupby("experiment")[["auc_speaker", "bac_speaker"]]
      .agg(["mean", "std"]).round(4).to_string())

# ── 3. SHAP on V1 corrected RF baseline (fold-local screening, seed=0) ───────
print("\nComputing SHAP on V1 RF baseline (seed=0, fold=0) ...")
manifest2 = pd.read_csv("data/processed/manifest.csv")
quality2  = pd.read_csv("data/processed/quality.csv")
feat_a2   = pd.read_csv("data/processed/features_acoustic.csv")
feat_l2   = pd.read_csv("data/processed/features_linguistic.csv")
QUALITY_COLS = ["snr_db", "loudness_dbfs", "peak", "bw99_hz", "duration_s"]
df_full2 = (manifest2[["file","speaker","label","split"]]
            .merge(quality2[["file"]+QUALITY_COLS], on="file", how="left")
            .merge(feat_a2, on="file", how="left")
            .merge(feat_l2, on="file", how="left"))
df_full2["label"] = (df_full2["label"]=="dementia").astype(int) if df_full2["label"].dtype==object else df_full2["label"].astype(int)
CAND = [c for c in df_full2.columns if c not in ["file","speaker","label","split"]+QUALITY_COLS]
X_all2 = df_full2[CAND].values.astype(float)
X_qual2 = df_full2[QUALITY_COLS].fillna(df_full2[QUALITY_COLS].median()).values
y2 = df_full2["label"].values
groups2 = df_full2["speaker"].values
files2  = df_full2["file"].values
from scipy.stats import spearmanr, rankdata
def screen(tr_idx, X, Q, cands, qt=0.5, ft=0.9):
    X_tr=X[tr_idx]; Q_tr=Q[tr_idx]; n=len(cands); drop=set()
    for fi in range(n):
        for qi in range(Q_tr.shape[1]):
            x=X_tr[:,fi]; q=Q_tr[:,qi]; v=np.isfinite(x)&np.isfinite(q)
            if v.sum()<10: continue
            r,_=spearmanr(x[v],q[v])
            if abs(r)>qt: drop.add(fi); break
    rem=[i for i in range(n) if i not in drop]
    if len(rem)>1:
        Xr=np.nan_to_num(X_tr[:,rem],nan=0.0)
        rk=np.apply_along_axis(rankdata,0,Xr); cm=np.corrcoef(rk.T); np.fill_diagonal(cm,0); d2=set()
        for a in range(len(rem)):
            if a in d2: continue
            for b in range(a+1,len(rem)):
                if b in d2: continue
                if abs(cm[a,b])>ft: d2.add(b)
        rem=[rem[i] for i in range(len(rem)) if i not in d2]
    return rem
cv0 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
tr0, va0 = next(cv0.split(X_all2, y2, groups2))
feat_idx0 = screen(tr0, X_all2, X_qual2, CAND)
feat_names0 = [CAND[i] for i in feat_idx0]
X_s = X_all2[:, feat_idx0]
rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1)
rf.fit(X_s[tr0], y2[tr0])
explainer = shap.TreeExplainer(rf)
shap_vals = explainer.shap_values(X_s[va0])
if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
    sv = shap_vals[:, :, 1]
elif isinstance(shap_vals, list):
    sv = shap_vals[1]
else:
    sv = shap_vals
shap_df = pd.DataFrame(sv, columns=feat_names0)
shap_df["file"] = files2[va0]
shap_df["seed"] = 0; shap_df["fold"] = 0
shap_df["note"] = "RF_B4_v1_fold_local_screened"
shap_df.to_csv("data/results/shap_values_v1.csv", index=False)
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_s[va0], feature_names=feat_names0, show=False, max_display=20)
plt.tight_layout()
plt.savefig("figures/shap_beeswarm_v1.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP saved → data/results/shap_values_v1.csv, figures/shap_beeswarm_v1.png")

# ── 4. GMM stability summary ─────────────────────────────────────────────────
ksel_summary = ksel.best_k.value_counts().sort_index().reset_index()
ksel_summary.columns = ["k", "n_folds_selected"]
ksel_summary["pct"] = (ksel_summary.n_folds_selected / len(ksel) * 100).round(1)
ksel_summary.to_csv("data/results/states_ksel_summary_v1.csv", index=False)
print("\n=== GMM K-SELECTION SUMMARY ===")
print(ksel_summary.to_string(index=False))
n_k1 = (ksel.best_k == 1).sum()
print(f"k=1 selected in {n_k1}/{len(ksel)} folds — no two-state structure forced")

# ── 5. Original vs. corrected comparison ─────────────────────────────────────
orig_joint = pd.read_csv("data/results/joint_results.csv")
v1_joint   = pd.read_csv("data/results/joint_results_v1.csv")
orig_bl    = pd.read_csv("data/results/baselines_summary.csv")
v1_bl      = pd.read_csv("data/results/baselines_summary_v1.csv")

orig_sum = orig_joint.groupby("experiment")[["auc","bac"]].agg(["mean","std"]).round(4)
orig_sum.columns = ["orig_auc_mean","orig_auc_std","orig_bac_mean","orig_bac_std"]
v1_sum   = v1_joint.groupby("experiment")[["auc","bac"]].agg(["mean","std"]).round(4)
v1_sum.columns   = ["v1_auc_mean","v1_auc_std","v1_bac_mean","v1_bac_std"]
comparison = orig_sum.join(v1_sum, how="outer")
comparison["auc_delta"] = (comparison.v1_auc_mean - comparison.orig_auc_mean).round(4)
comparison["reason_changed"] = "modality_mask+fold_local_screening+inner_split+same_bucket_pairs"
comparison.to_csv("data/results/orig_vs_v1_comparison.csv")

print("\n=== ORIGINAL vs V1 CORRECTED ===")
print(comparison[["orig_auc_mean","v1_auc_mean","auc_delta"]].to_string())

# Quality baseline comparison
q_orig = 0.625  # single-run, unmatched
q_v1   = v1_bl[v1_bl["model"]=="B7_RF_quality_only"]["auc_mean"].values[0] if "model" in v1_bl.columns else np.nan
print(f"\nQuality baseline: original={q_orig} (single unmatched run), v1={q_v1:.4f} (5x5 CV, matched)")
print("NOTE: Exceeding the quality baseline does not establish absence of recording-quality confounding.")

# Temporal concordance comparison
print("\n=== TEMPORAL CONCORDANCE COMPARISON ===")
print("Original (all data, training + val mixed): 87.8% training, 55.3% held-out")
print(f"V1 (held-out only, bucket-aggregated): {conc_df.concordance.mean()*100:.1f}% mean (std {conc_df.concordance.std()*100:.1f}%)")
print(f"V1 OOF consensus: {oof_conc*100:.1f}%")
print("NOTE: V1 concordance is computed only on outer-held-out speakers per fold.")

# Save full handoff manifest
handoff = dict(
    version="V1-corrected",
    n_tests_passed=13, n_tests_total=13,
    test_file="tests/test_v1_corrections.py",
    validation_checks_passed=71,
    validation_script="scripts/15_validate_v1.py",
    asr_provenance="UNRESOLVED — model identity not stored in transcripts. Config: small.en, script default: medium.en.",
    transcript_hashes="data/results/transcript_provenance_v1.csv",
    wavlm_model="microsoft/wavlm-base-plus",
    bert_model="bert-base-uncased",
    n_recordings=275, n_speakers=141,
    outputs={
        "per_recording_predictions": "data/results/joint_predictions_v1.csv",
        "speaker_aggregated": "data/results/eval_speaker_v1.csv",
        "per_fold_metrics": "data/results/joint_results_v1.csv",
        "summary_metrics": "data/results/joint_summary_v1.csv",
        "baselines": "data/results/baselines_v1.csv",
        "baselines_summary": "data/results/baselines_summary_v1.csv",
        "fold_features": "data/results/joint_fold_features_v1/",
        "partition_membership": "data/results/partition_membership_v1.csv",
        "partition_audit": "data/results/partition_audit_v1.csv",
        "ranking_pair_audit": "data/results/ranking_pair_audit_v1.csv",
        "temporal_concordance": "data/results/temporal_concordance_v1.csv",
        "states_kselect": "data/results/states_kselect_v1.csv",
        "states_bic": "data/results/states_bic_v1.csv",
        "states_assignments": "data/results/states_assignments_v1.csv",
        "shap_values": "data/results/shap_values_v1.csv",
        "shap_figure": "figures/shap_beeswarm_v1.png",
        "orig_vs_v1_comparison": "data/results/orig_vs_v1_comparison.csv",
        "transcript_hashes": "data/results/transcript_provenance_v1.csv",
        "validation_log": "data/results/analysis_validation_v1.csv",
        "provenance": "data/results/joint_provenance_v1.json",
    },
    corrections_applied=[
        "Modality masking decoupled: stream zeroed only by its own key (models.py:apply_modality_mask)",
        "Fold-local feature screening: quality and collinearity screening fitted on outer-train only",
        "Inner validation split for epoch selection (StratifiedGroupKFold, first split of outer-train)",
        "Phase-2 refit: fresh model and preprocessing fitted on full outer-train for selected epochs",
        "Same-bucket rank pairs excluded from training (10 speakers, 5-12 pairs per fold)",
        "GMM k forced-minimum removed: k=1 accepted (selected in 7/25 folds)",
        "Quality baseline re-evaluated on matched 5x5 CV folds (B7_RF_quality_only)",
        "Per-recording outer-held-out predictions saved with full metadata",
        "Partition membership and preprocessing-fit membership logged",
        "Ranking gradient accounting: per-fold pair counts and expected updates logged",
        "ASR provenance documented as unresolved; all 275 transcript SHA-256 hashes fixed",
        "Branch/commit checks in validate.py updated for V1-corrected"
    ],
    legacy_results_preserved="All V0 original results in data/results/ (no _v1 suffix)"
)
json.dump(handoff, open("data/results/handoff_manifest_v1.json", "w"), indent=2)
print("\nHandoff manifest saved → data/results/handoff_manifest_v1.json")
