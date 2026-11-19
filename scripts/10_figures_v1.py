"""Generate V1-corrected main result figures.

Produces:
  figures/fig_model_comparison_v1.png   - AUC bar chart with CIs (V0 vs V1)
  figures/fig_temporal_v1.png           - Held-out temporal concordance distribution
  figures/fig_gmm_v1.png                - GMM k-selection BIC + silhouette
  figures/fig_probes_v1.png             - Speaker-level logistic probe AUCs
  figures/fig_group_effects_v1.png      - Top group-difference features (Hedges g)
  figures/fig_longitudinal_v1.png       - Longitudinal endpoint effects
  figures/fig_rate_decomp_v1.png        - Rate decomposition (endpoint speakers)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RES  = BASE / 'data/results'
FIGS = BASE / 'figures'
FIGS.mkdir(exist_ok=True)

BLUE   = '#2563EB'
ORANGE = '#EA580C'
GREY   = '#6B7280'
RED    = '#DC2626'
GREEN  = '#16A34A'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})

# ── 1. Model comparison (V0 vs V1) ──────────────────────────────────────────
def fig_model_comparison():
    v1 = pd.read_csv(RES / 'joint_summary_v1.csv')
    # V0 numbers from original results
    v0 = pd.DataFrame({
        'experiment': ['V_feat','V_audio','V_text','V_cls','V_joint'],
        'auc_mean':   [0.5000,  0.6359,   0.6087,  0.6562,  0.6654],
        'auc_std':    [0.000,   0.098,    0.088,   0.097,   0.088],
    })
    labels = {'V_feat':'Features only','V_audio':'WavLM+features',
              'V_text':'BERT+features','V_cls':'All (cls only)','V_joint':'All + ranking'}
    order  = ['V_feat','V_audio','V_text','V_cls','V_joint']
    v1 = v1.set_index('experiment').loc[order]
    v0 = v0.set_index('experiment').loc[order]
    x  = np.arange(len(order))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars0 = ax.bar(x - w/2, v0.auc_mean, w, yerr=v0.auc_std, color=GREY,
                   capsize=4, label='V0 (original, contaminated)', alpha=0.75)
    bars1 = ax.bar(x + w/2, v1.auc_mean, w, yerr=v1.auc_std, color=BLUE,
                   capsize=4, label='V1 (corrected)', alpha=0.9)
    ax.axhline(0.5, ls='--', lw=1, color='black', label='Chance (0.50)')
    ax.axhline(0.6099, ls=':', lw=1.2, color=ORANGE,
               label='Quality baseline V1 (0.610)')
    ax.set_xticks(x)
    ax.set_xticklabels([labels[e] for e in order], rotation=15, ha='right')
    ax.set_ylabel('AUC (ROC)')
    ax.set_ylim(0.42, 0.75)
    ax.set_title('Model AUC: V0 vs V1-corrected (5×5 speaker-grouped CV)', pad=10)
    ax.legend(fontsize=8, loc='upper left')
    # Annotate V1 bars
    for bar, exp in zip(bars1, order):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color=BLUE)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_model_comparison_v1.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_model_comparison_v1.png')

# ── 2. Held-out temporal concordance distribution ────────────────────────────
def fig_temporal():
    tc = pd.read_csv(RES / 'temporal_person_concordance_v1.csv')
    fig, ax = plt.subplots(figsize=(6, 3.8))
    vals = tc['concordance'].values
    ax.hist(vals, bins=np.linspace(0, 1, 11), color=BLUE, alpha=0.8, edgecolor='white')
    ax.axvline(vals.mean(), color=RED, lw=2, ls='--',
               label=f'Mean = {vals.mean():.3f}')
    ax.axvline(0.5, color=GREY, lw=1.2, ls=':', label='Chance (0.50)')
    ax.set_xlabel('Concordance per speaker per fold')
    ax.set_ylabel('Count')
    ax.set_title('Held-out temporal concordance (V1, prog_score)\nn = %d speaker×fold pairs' % len(tc))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_temporal_v1.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_temporal_v1.png')

# ── 3. GMM k-selection ───────────────────────────────────────────────────────
def fig_gmm():
    bic  = pd.read_csv(RES / 'states_bic_v1.csv')
    ksel = pd.read_csv(RES / 'states_ksel_summary_v1.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    # BIC by k
    ax = axes[0]
    for (seed, fold), g in bic.groupby(['seed','fold']):
        ax.plot(g.k, g.bic, color=GREY, alpha=0.25, lw=0.8)
    mean_bic = bic.groupby('k').bic.mean()
    ax.plot(mean_bic.index, mean_bic.values, color=BLUE, lw=2.5, label='Mean BIC')
    ax.set_xlabel('Number of components (k)')
    ax.set_ylabel('BIC')
    ax.set_title('GMM BIC by k across 25 folds')
    ax.legend(fontsize=8)
    # k selection histogram
    ax = axes[1]
    counts = ksel.set_index('best_k')['count'] if 'count' in ksel.columns else \
             pd.read_csv(RES / 'states_kselect_v1.csv').best_k.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=BLUE, alpha=0.85, edgecolor='white')
    ax.set_xlabel('Selected k (BIC-optimal)')
    ax.set_ylabel('Number of folds (out of 25)')
    ax.set_title('GMM component selection\n(no forced minimum; k=1 allowed)')
    ax.set_xticks(sorted(counts.index))
    for x_, y_ in zip(counts.index, counts.values):
        ax.text(x_, y_ + 0.3, str(y_), ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_gmm_v1.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_gmm_v1.png')

# ── 4. Logistic probe AUCs ───────────────────────────────────────────────────
def fig_probes():
    pr = pd.read_csv(RES / 'supplementary_probe_summary_v1.csv')
    order = ['Timing3','Language5','Voice6','Spectral23','TimingLanguage8',
             'All37','Quality6','TimingQuality9','AllQuality43','Gender1']
    pr = pr[pr.model.isin(order)].set_index('model').loc[order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [BLUE if 'Quality' not in m and m != 'Gender1' else ORANGE
              for m in order]
    y = np.arange(len(order))
    ax.barh(y, pr.auc, xerr=[(pr.auc - pr.auc_low), (pr.auc_high - pr.auc)],
            color=colors, alpha=0.85, capsize=3, height=0.6)
    ax.axvline(0.5, ls='--', lw=1, color='black')
    ax.axvline(0.6099, ls=':', lw=1.2, color=ORANGE, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel('AUC (speaker-level 5×5 CV)')
    ax.set_title('Fixed-spec logistic probe AUCs (V1)')
    p1 = mpatches.Patch(color=BLUE,   label='Speech features')
    p2 = mpatches.Patch(color=ORANGE, label='Quality / gender')
    ax.legend(handles=[p1, p2], fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_probes_v1.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_probes_v1.png')

# ── 5. Group effects (top features by Hedges g) ──────────────────────────────
def fig_group_effects():
    ge = pd.read_csv(RES / 'speaker_group_effects_v1.csv')
    top = ge.reindex(ge.hedges_g.abs().sort_values(ascending=False).index).head(15)
    top = top.sort_values('hedges_g')
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [GREEN if g > 0 else RED for g in top.hedges_g]
    ax.barh(top.feature, top.hedges_g, color=colors, alpha=0.8)
    ax.axvline(0, color='black', lw=0.8)
    # CI lines
    for i, (_, row) in enumerate(top.iterrows()):
        ci_lo = row.ci_low / (row.mean1 - row.mean0 + 1e-9) * row.hedges_g \
                if abs(row.mean1 - row.mean0) > 1e-9 else row.hedges_g * 0.8
        ci_hi = row.ci_high / (row.mean1 - row.mean0 + 1e-9) * row.hedges_g \
                if abs(row.mean1 - row.mean0) > 1e-9 else row.hedges_g * 1.2
        # simpler: just show hedges_g with no CI recompute
    ax.set_xlabel("Hedges' g (dementia vs control, speaker means)")
    ax.set_title('Top 15 group-difference features (V1, n=141 speakers)')
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_group_effects_v1.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_group_effects_v1.png')

# ── 6. Longitudinal endpoint effects ─────────────────────────────────────────
def fig_longitudinal():
    le = pd.read_csv(RES / 'longitudinal_endpoint_effects_v1.csv')
    timing_ling = ['speech_rate','pause_mean_s','pause_ratio','mattr50',
                   'filler_rate','immediate_rep_rate','bigram_rep','mean_sent_len',
                   'articulation_rate']
    le = le[le.feature.isin(timing_ling)].set_index('feature').loc[
         [f for f in timing_ling if f in le.feature.values]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(le))
    colors = [GREEN if m > 0 else RED for m in le.mean_delta]
    ax.barh(y, le.mean_delta, color=colors, alpha=0.8, height=0.6)
    ax.errorbar(le.mean_delta, y, xerr=[le.mean_delta - le.mean_low,
                                         le.mean_high - le.mean_delta],
                fmt='none', color='black', capsize=3, lw=1.2)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(le.index)
    ax.set_xlabel('Mean change (last − first epoch, speaker units)')
    ax.set_title('Longitudinal endpoint effects (V1, n=31 speakers with ≥2 epochs)')
    # Annotate sign-flip p
    for i, (_, row) in enumerate(le.iterrows()):
        ax.text(le.mean_delta.max() * 1.05, i,
                f'p={row.signflip_p:.3f}', va='center', fontsize=7, color=GREY)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_longitudinal_v1.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_longitudinal_v1.png')

# ── 7. Rate decomposition ────────────────────────────────────────────────────
def fig_rate_decomp():
    rd = pd.read_csv(RES / 'rate_decomposition_v1.csv')
    labels = {'log_r':'Overall word rate','log_a':'Pause-excluded rate','log_nonpause':'Non-pause fraction'}
    rd['label'] = rd.component.map(labels)
    rd['pct']   = rd.geometric_percent_change
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = [GREEN if p > 0 else RED for p in rd.pct]
    ax.barh(rd.label, rd.pct, color=colors, alpha=0.85, height=0.5)
    ax.errorbar(rd.pct, np.arange(len(rd)),
                xerr=[rd.pct - (np.expm1(rd.low) * 100),
                      (np.expm1(rd.high) * 100) - rd.pct],
                fmt='none', color='black', capsize=3, lw=1.2)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Geometric % change (latest − earliest epoch)')
    ax.set_title('Speech rate decomposition (V1, n=31 endpoint speakers)')
    for i, (_, row) in enumerate(rd.iterrows()):
        ax.text(rd.pct.min() - 0.5, i, f"{row.pct:.1f}%",
                va='center', ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / 'fig_rate_decomp_v1.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_rate_decomp_v1.png')

if __name__ == '__main__':
    fig_model_comparison()
    fig_temporal()
    fig_gmm()
    fig_probes()
    fig_group_effects()
    fig_longitudinal()
    fig_rate_decomp()
    print('\nAll V1 figures saved to figures/')
