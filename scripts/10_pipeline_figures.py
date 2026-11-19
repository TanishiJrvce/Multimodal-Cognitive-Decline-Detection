"""Generate summary figures for the full pipeline results.

Figures produced:
  figures/fig_model_comparison.png   – AUC bar chart across all model variants + baselines
  figures/fig_state_distribution.png – Progression score distributions by group and state
  figures/fig_feature_importance.png – Top SHAP features (RF baseline, mean |SHAP|)
  figures/fig_temporal_pair_acc.png  – Temporal pair accuracy summary panel
  figures/fig_bic_silhouette.png     – GMM k selection (BIC + silhouette)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

Path("figures").mkdir(exist_ok=True)
GREY   = "#6b7280"
BLUE   = "#2563eb"
TEAL   = "#0d9488"
AMBER  = "#d97706"
RED    = "#dc2626"
GREEN  = "#16a34a"

# ── 1. Model comparison ───────────────────────────────────────────────────────
print("Fig 1: model comparison …")
joint = pd.read_csv("data/results/joint_summary.csv", index_col=0)
baselines = pd.read_csv("data/results/baselines_summary.csv")

LABELS = {
    "V_feat":  "FusionNet\n(features only)",
    "V_text":  "FusionNet\n(BERT + feats)",
    "V_audio": "FusionNet\n(WavLM + feats)",
    "V_cls":   "FusionNet\n(all, cls-only)",
    "V_joint": "FusionNet\n(all + ranking)",
}
order   = ["V_feat", "V_text", "V_audio", "V_cls", "V_joint"]
colors  = [GREY, GREY, TEAL, BLUE, GREEN]
means   = [joint.loc[e, "auc_mean"] for e in order]
stds    = [joint.loc[e, "auc_std"]  for e in order]
labels  = [LABELS[e] for e in order]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(order))
bars = ax.bar(x, means, yerr=stds, capsize=4,
              color=colors, edgecolor="white", linewidth=0.8, zorder=3)
# Quality-only floor
ax.axhline(0.625, color=RED,   ls="--", lw=1.6, label="Quality-only baseline (0.625)", zorder=4)
ax.axhline(0.500, color=GREY,  ls=":",  lw=1.2, label="Chance (0.500)", zorder=4)
# Speaker-level AUC annotation for V_joint
ax.annotate("Speaker-level\nAUC = 0.688", xy=(4, 0.665), xytext=(3.4, 0.74),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
            color=GREEN, fontsize=8.5, ha="center")
for b, m in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, m+0.005, f"{m:.3f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.42, 0.78)
ax.set_ylabel("AUC (mean ± std across 25 runs)", fontsize=10)
ax.set_title("Recording-level AUC — speaker-grouped cross-validation\n(5 seeds × 5 folds, DementiaNet)", fontsize=11)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.35, zorder=0)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/fig_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/fig_model_comparison.png")

# ── 2. State + group distribution ────────────────────────────────────────────
print("Fig 2: state distribution …")
asgn = pd.read_csv("data/results/states_assignments.csv")
rep  = asgn[(asgn.seed==0) & (asgn.fold==0)].copy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: progression score by group
ax = axes[0]
for lbl, col, name in [(1, RED, "Dementia-associated"), (0, BLUE, "Comparison")]:
    vals = rep[rep.label==lbl]["prog_score"].values
    ax.hist(vals, bins=25, alpha=0.6, color=col, label=f"{name} (n={len(vals)})", density=True)
ax.set_xlabel("Progression score"); ax.set_ylabel("Density")
ax.set_title("Progression score by group\n(seed 0, fold 0)")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

# Right: progression score by state
ax = axes[1]
state_colors = [TEAL, AMBER]
for s in sorted(rep.state.unique()):
    vals = rep[rep.state==s]["prog_score"].values
    pct_dem = (rep[(rep.state==s) & (rep.label==1)].shape[0] / len(rep[rep.state==s]))*100
    ax.hist(vals, bins=20, alpha=0.65, color=state_colors[s],
            label=f"State-{s}  (n={len(vals)}, {pct_dem:.0f}% dementia)", density=True)
ax.set_xlabel("Progression score"); ax.set_ylabel("Density")
ax.set_title("Progression score by latent state\n(GMM k=2, seed 0 fold 0)")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Progression score distributions — ⚠ States are not clinical diagnoses", fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig("figures/fig_state_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/fig_state_distribution.png")

# ── 3. Feature importance (top SHAP) ─────────────────────────────────────────
print("Fig 3: feature importance …")
shap_df = pd.read_csv("data/results/shap_values.csv")
feat_cols = [c for c in shap_df.columns if c != "file"]
mean_abs  = shap_df[feat_cols].abs().mean().sort_values(ascending=False).head(20)

TIMING_FEATS = {"speech_rate","pause_mean_s","pause_ratio","articulation_rate"}
LING_FEATS   = {"mattr50","filler_rate","immediate_rep_rate","bigram_rep","mean_sent_len",
                "type_token_ratio","word_rate","mean_word_freq","lexical_density","pronoun_rate","n_words"}

def feat_color(f):
    if f in TIMING_FEATS: return AMBER
    if f in LING_FEATS:   return TEAL
    return BLUE

bar_colors = [feat_color(f) for f in mean_abs.index]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(len(mean_abs)), mean_abs.values, color=bar_colors, edgecolor="white")
ax.set_yticks(range(len(mean_abs))); ax.set_yticklabels(mean_abs.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP| value", fontsize=10)
ax.set_title("Top 20 features by mean |SHAP| value\n(RF baseline, fold 0, seed 0)", fontsize=11)
patches = [mpatches.Patch(color=AMBER, label="Timing"),
           mpatches.Patch(color=TEAL,  label="Linguistic"),
           mpatches.Patch(color=BLUE,  label="Acoustic/spectral")]
ax.legend(handles=patches, fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/fig_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/fig_feature_importance.png")

# ── 4. GMM k selection (BIC + silhouette) ────────────────────────────────────
print("Fig 4: GMM k selection …")
bic = pd.read_csv("data/results/states_bic.csv")
sil = pd.read_csv("data/results/states_silhouette.csv")
bic_m = bic.groupby("k")["bic"].mean()
sil_m = sil.groupby("k")["silhouette"].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.plot(bic_m.index, bic_m.values, "o-", color=BLUE, lw=2)
ax1.set_xlabel("Number of states (k)"); ax1.set_ylabel("Mean BIC (lower = better)")
ax1.set_title("GMM model selection — BIC\n(mean over 25 seed×fold runs)")
ax1.set_xticks(bic_m.index); ax1.spines[["top","right"]].set_visible(False)
ax1.axvline(2, color=RED, ls="--", lw=1.3, label="k=2 selected")
ax1.legend(fontsize=9)

ax2.plot(sil_m.index, sil_m.values, "s-", color=TEAL, lw=2)
ax2.set_xlabel("Number of states (k)"); ax2.set_ylabel("Mean silhouette score")
ax2.set_title("GMM model selection — Silhouette\n(mean over 25 seed×fold runs)")
ax2.set_xticks(sil_m.index); ax2.spines[["top","right"]].set_visible(False)
ax2.axvline(2, color=RED, ls="--", lw=1.3, label="k=2 selected")
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("figures/fig_bic_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/fig_bic_silhouette.png")

# ── 5. Temporal pair accuracy summary ────────────────────────────────────────
print("Fig 5: temporal pair accuracy …")
pair = pd.read_csv("data/results/eval_pair_accuracy.csv")
correct = int(pair["n_correct"].iloc[0])
total   = int(pair["n_pairs"].iloc[0])
wrong   = total - correct
pair_acc= float(pair["pair_accuracy"].iloc[0])

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Correctly ordered\npairs", "Incorrectly ordered\npairs"],
       [correct, wrong], color=[GREEN, RED], edgecolor="white", width=0.5)
ax.set_ylabel("Number of recording pairs")
ax.set_title(f"Temporal pair accuracy = {pair_acc:.1%}  ({correct}/{total} pairs)\n"
             f"V_joint progression scores vs. defensible within-person ordering\n"
             f"⚠ Internal consistency check — not independent clinical validation",
             fontsize=9)
for rect, val in zip(ax.patches, [correct, wrong]):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.3, str(val),
            ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(0, total+5)
plt.tight_layout()
plt.savefig("figures/fig_temporal_pair_acc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/fig_temporal_pair_acc.png")

print("\n✓ All 5 pipeline figures saved to figures/")
