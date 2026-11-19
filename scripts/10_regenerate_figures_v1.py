"""Regenerate all stale figures with V1-corrected data.

Overwrites in place:
  figures/fig_model_comparison.png
  figures/fig_state_distribution.png
  figures/fig_feature_importance.png
  figures/fig_temporal_pair_acc.png
  figures/fig_bic_silhouette.png
  figures/shap_beeswarm.png
  figures/trajectory_<speaker>.png  (6 speakers)
  figures/01_speaker_effects.png/.pdf
  figures/02_original_paired_gains.png/.pdf
  figures/03_rate_decomposition.png/.pdf
  figures/04_temporal_generalization.png/.pdf
  figures/05_attribution_normalization.png/.pdf
  figures/06_score_associations.png/.pdf
  figures/07_supplementary_probes.png/.pdf
  figures/08_state_structure.png/.pdf
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RES  = BASE / 'data/results'
PROC = BASE / 'data/processed'
FIGS = BASE / 'figures'

GREY   = '#6b7280'
BLUE   = '#2563eb'
TEAL   = '#0d9488'
AMBER  = '#d97706'
RED    = '#dc2626'
GREEN  = '#16a34a'
ORANGE = '#EA580C'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})

def savefig(fig, name, pdf=False):
    path = FIGS / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  → {name}')
    if pdf:
        fig.savefig(FIGS / name.replace('.png', '.pdf'), bbox_inches='tight')
        print(f'  → {name.replace(".png", ".pdf")}')
    plt.close(fig)

# ── 1. fig_model_comparison.png ─────────────────────────────────────────────
def fig_model_comparison():
    js = pd.read_csv(RES / 'joint_summary_v1.csv').set_index('experiment')
    bs = pd.read_csv(RES / 'baselines_summary_v1.csv')
    qb = bs[bs.model == 'B7_RF_quality_only'].iloc[0]

    LABELS = {
        'V_feat':  'FusionNet\n(features only)',
        'V_text':  'FusionNet\n(BERT + feats)',
        'V_audio': 'FusionNet\n(WavLM + feats)',
        'V_cls':   'FusionNet\n(all, cls-only)',
        'V_joint': 'FusionNet\n(all + ranking)',
    }
    order  = ['V_feat', 'V_text', 'V_audio', 'V_cls', 'V_joint']
    colors = [GREY, GREY, TEAL, BLUE, GREEN]
    means  = [js.loc[e, 'auc_mean'] for e in order]
    stds   = [js.loc[e, 'auc_std']  for e in order]
    spk_auc = 0.648  # V1 speaker-level AUC for V_joint

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=colors, edgecolor='white', linewidth=0.8, zorder=3)
    ax.axhline(qb.auc_mean, color=RED, ls='--', lw=1.6,
               label=f'Quality-only baseline ({qb.auc_mean:.3f} ± {qb.auc_std:.3f})', zorder=4)
    ax.axhline(0.500, color=GREY, ls=':', lw=1.2, label='Chance (0.500)', zorder=4)
    ax.annotate(f'Speaker-level\nAUC = {spk_auc:.3f}',
                xy=(4, means[4]), xytext=(3.4, 0.74),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
                color=GREEN, fontsize=8.5, ha='center')
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.005, f'{m:.3f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[e] for e in order], fontsize=9)
    ax.set_ylim(0.42, 0.78)
    ax.set_ylabel('AUC (mean ± std across 25 runs)', fontsize=10)
    ax.set_title('Recording-level AUC – speaker-grouped cross-validation\n'
                 '(5 seeds × 5 folds, DementiaNet) [V1-corrected]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.35, zorder=0)
    savefig(fig, 'fig_model_comparison.png')

# ── 2. fig_state_distribution.png ───────────────────────────────────────────
def fig_state_distribution():
    sa  = pd.read_csv(RES / 'states_assignments_v1.csv')
    rep = sa[(sa.seed == 0) & (sa.fold == 0)].copy()
    best_k = int(rep.best_k.iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for lbl, col, name in [(1, RED, 'Dementia-associated'), (0, BLUE, 'Comparison')]:
        vals = rep[rep.label == lbl]['prog_score'].values
        ax.hist(vals, bins=25, alpha=0.6, color=col,
                label=f'{name} (n={len(vals)})', density=True)
    ax.set_xlabel('Progression score')
    ax.set_ylabel('Density')
    ax.set_title('Progression score by group\n(seed 0, fold 0) [V1-corrected]')
    ax.legend(fontsize=9)

    ax = axes[1]
    state_colors = [TEAL, AMBER, GREY]
    for s in sorted(rep.state.unique()):
        vals = rep[rep.state == s]['prog_score'].values
        pct = rep[(rep.state == s) & (rep.label == 1)].shape[0] / len(rep[rep.state == s]) * 100
        ax.hist(vals, bins=20, alpha=0.65, color=state_colors[s],
                label=f'State-{s}  (n={len(vals)}, {pct:.0f}% dementia)', density=True)
    ax.set_xlabel('Progression score')
    ax.set_ylabel('Density')
    ax.set_title(f'Progression score by state (k={best_k}, BIC-selected)\n'
                 f'(seed 0, fold 0) [V1-corrected; labels are operational only]')
    ax.legend(fontsize=9)

    fig.tight_layout()
    savefig(fig, 'fig_state_distribution.png')

# ── 3. fig_feature_importance.png (SHAP) ────────────────────────────────────
def fig_feature_importance():
    shap_df = pd.read_csv(RES / 'shap_values_v1.csv')
    feat_cols = list(shap_df.select_dtypes(include='number').columns)
    # exclude any numeric metadata columns that aren't feature SHAP values
    feat_cols = [c for c in feat_cols if c not in ('seed', 'fold')]
    mean_abs = shap_df[feat_cols].abs().mean().sort_values(ascending=False).head(20)

    TIMING = {'speech_rate', 'pause_mean_s', 'pause_ratio', 'articulation_rate'}
    LING   = {'mattr50', 'filler_rate', 'immediate_rep_rate', 'bigram_rep', 'mean_sent_len'}

    def feat_color(f):
        if f in TIMING: return AMBER
        if f in LING:   return TEAL
        return BLUE

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_colors = [feat_color(f) for f in mean_abs.index]
    ax.barh(range(len(mean_abs)), mean_abs.values, color=bar_colors, edgecolor='white')
    ax.set_yticks(range(len(mean_abs)))
    ax.set_yticklabels(mean_abs.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP| value', fontsize=10)
    ax.set_title('Top 20 features by mean |SHAP| value\n(V_joint, V1-corrected)', fontsize=11)
    patches = [mpatches.Patch(color=AMBER, label='Timing'),
               mpatches.Patch(color=TEAL,  label='Linguistic'),
               mpatches.Patch(color=BLUE,  label='Acoustic/spectral')]
    ax.legend(handles=patches, fontsize=9)
    savefig(fig, 'fig_feature_importance.png')

# ── 4. fig_temporal_pair_acc.png ────────────────────────────────────────────
def fig_temporal_pair_acc():
    tc = pd.read_csv(RES / 'temporal_person_concordance_v1.csv')
    n_total = len(tc)
    # Count speaker×fold pairs where concordance > 0.5 (correct majority)
    n_correct = int((tc.concordance > 0.5).sum())
    n_tie     = int((tc.concordance == 0.5).sum())
    n_wrong   = n_total - n_correct - n_tie
    mean_conc = tc.concordance.mean()

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(['Concordance\n> 0.5', 'Concordance\n= 0.5', 'Concordance\n< 0.5'],
                  [n_correct, n_tie, n_wrong],
                  color=[GREEN, AMBER, RED], edgecolor='white', width=0.5)
    ax.set_ylabel('Speaker × fold pairs')
    ax.set_title(f'Held-out temporal concordance (V1-corrected)\n'
                 f'mean = {mean_conc:.3f}  |  n = {n_total} speaker×fold pairs\n'
                 f'Outer-test only; same-bucket pairs excluded from ordering', fontsize=9)
    for rect, val in zip(bars, [n_correct, n_tie, n_wrong]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.set_ylim(0, max(n_correct, n_wrong) + 10)
    savefig(fig, 'fig_temporal_pair_acc.png')

# ── 5. fig_bic_silhouette.png ────────────────────────────────────────────────
def fig_bic_silhouette():
    bic = pd.read_csv(RES / 'states_bic_v1.csv')
    sil = pd.read_csv(RES / 'states_silhouette_v1.csv')
    bic_m = bic.groupby('k')['bic'].mean()
    sil_m = sil.groupby('k')['silhouette'].mean()
    best_k_counts = pd.read_csv(RES / 'states_kselect_v1.csv').best_k.value_counts().sort_index()
    modal_k = int(best_k_counts.idxmax())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # BIC
    for (seed, fold), g in bic.groupby(['seed', 'fold']):
        ax1.plot(g.k, g.bic, color=GREY, alpha=0.2, lw=0.7)
    ax1.plot(bic_m.index, bic_m.values, 'o-', color=BLUE, lw=2.5, label='Mean BIC')
    ax1.axvline(modal_k, color=RED, ls='--', lw=1.3,
                label=f'Modal k={modal_k} ({int(best_k_counts[modal_k])}/25 folds)')
    ax1.set_xlabel('Number of states (k)')
    ax1.set_ylabel('BIC (lower = better)')
    ax1.set_title('GMM model selection – BIC\n(mean over 25 seed×fold runs) [V1-corrected]')
    ax1.set_xticks(bic_m.index)
    ax1.legend(fontsize=9)

    # Silhouette
    sil_valid = sil_m.dropna()
    if len(sil_valid) > 0:
        ax2.plot(sil_valid.index, sil_valid.values, 's-', color=TEAL, lw=2.5)
        ax2.axvline(modal_k, color=RED, ls='--', lw=1.3, label=f'Modal k={modal_k}')
        ax2.legend(fontsize=9)
    ax2.set_xlabel('Number of states (k)')
    ax2.set_ylabel('Mean silhouette score')
    ax2.set_title('GMM model selection – Silhouette\n(undefined at k=1; shown where available)')
    ax2.set_xticks(bic_m.index)

    # k-selection inset as bar
    ax2b = ax2.inset_axes([0.55, 0.55, 0.42, 0.40])
    ax2b.bar(best_k_counts.index, best_k_counts.values, color=BLUE, alpha=0.7)
    ax2b.set_xlabel('k', fontsize=7); ax2b.set_ylabel('folds', fontsize=7)
    ax2b.set_title('k selected', fontsize=7)
    ax2b.tick_params(labelsize=6)
    ax2b.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    savefig(fig, 'fig_bic_silhouette.png')

# ── 6. shap_beeswarm.png ────────────────────────────────────────────────────
def fig_shap_beeswarm():
    # Reuse the already-generated v1 beeswarm
    import shutil
    src = FIGS / 'shap_beeswarm_v1.png'
    dst = FIGS / 'shap_beeswarm.png'
    if src.exists():
        shutil.copy2(src, dst)
        print(f'  → shap_beeswarm.png (copied from v1)')
    else:
        print('  shap_beeswarm_v1.png not found, skipping')

# ── 7. trajectory_<speaker>.png (6 speakers) ────────────────────────────────
def fig_trajectories():
    pred  = pd.read_csv(RES / 'joint_predictions_v1.csv')
    order = pd.read_csv(PROC / 'recording_order.csv').rename(columns={'recording': 'file'})
    vj    = pred[pred.experiment == 'V_joint']

    speakers = ['Aileen Hernandez', 'Andrew Sachs', 'Antony Flew',
                'B B King', 'Ben Bradlee', 'Bill Buckner']

    for spk in speakers:
        spk_pred = vj[vj.speaker == spk].copy()
        if len(spk_pred) == 0:
            print(f'  No predictions for {spk}, skipping')
            continue

        # Merge with rank order
        spk_order = order[order.speaker == spk][['file', 'rank']].drop_duplicates()
        merged = spk_pred.merge(spk_order, on='file', how='left')

        # Aggregate by seed/fold then by rank bucket
        # Average group_prob across seeds/folds per file, then per rank bucket
        per_file = merged.groupby(['file', 'rank']).agg(
            prob=('group_prob', 'mean'),
            prob_sd=('group_prob', 'std'),
            label=('label', 'first')).reset_index()
        per_rank = per_file.groupby('rank').agg(
            prob=('prob', 'mean'),
            prob_sd=('prob', 'mean'),
            n=('file', 'count')).reset_index().sort_values('rank')

        label = int(spk_pred.label.iloc[0])
        color = RED if label == 1 else BLUE
        group = 'Dementia-associated' if label == 1 else 'Comparison'

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(per_rank['rank'], per_rank['prob'], 'o-', color=color, lw=2, ms=7)
        ax.fill_between(per_rank['rank'],
                        (per_rank['prob'] - per_rank['prob_sd']).clip(0, 1),
                        (per_rank['prob'] + per_rank['prob_sd']).clip(0, 1),
                        alpha=0.18, color=color)
        ax.set_xlabel('Recording epoch (temporal rank)')
        ax.set_ylabel('Group probability (dementia)')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, ls='--', lw=1, color=GREY)
        ax.set_title(f'{spk}\n({group}) [V1-corrected, averaged over 5×5 CV]', fontsize=9)
        ax.set_xticks(per_rank['rank'])
        fig.tight_layout()
        fname = f'trajectory_{spk}.png'
        savefig(fig, fname)

# ── 8. 01_speaker_effects.png/.pdf ──────────────────────────────────────────
def fig_01_speaker_effects():
    ge  = pd.read_csv(RES / 'speaker_group_effects_v1.csv')
    top = ge.reindex(ge.hedges_g.abs().sort_values(ascending=False).index).head(20)
    top = top.sort_values('hedges_g')

    TIMING = {'speech_rate', 'pause_mean_s', 'pause_ratio'}
    LING   = {'mattr50', 'filler_rate', 'immediate_rep_rate', 'bigram_rep', 'mean_sent_len'}

    def feat_color(f):
        if f in TIMING: return AMBER
        if f in LING:   return TEAL
        return BLUE

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (_, row) in enumerate(top.iterrows()):
        c = feat_color(row.feature)
        ax.barh(i, row.hedges_g, color=c, alpha=0.8, height=0.7)
        ax.errorbar(row.difference / (abs(row.difference) + 1e-9) * abs(row.hedges_g),
                    i, xerr=0, fmt='none')  # just bar, no recomputed CI

    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.feature, fontsize=8.5)
    ax.set_xlabel("Hedges' g (dementia vs control, speaker means)")
    ax.set_title("Top 20 group-difference features (V1-corrected)\n"
                 "n=141 speakers; 5,000 bootstrap, 10,000 permutation resamples")
    patches = [mpatches.Patch(color=AMBER, label='Timing'),
               mpatches.Patch(color=TEAL,  label='Linguistic'),
               mpatches.Patch(color=BLUE,  label='Acoustic/spectral')]
    ax.legend(handles=patches, fontsize=9, loc='lower right')
    fig.tight_layout()
    savefig(fig, '01_speaker_effects.png', pdf=True)

# ── 9. 02_original_paired_gains.png/.pdf ────────────────────────────────────
def fig_02_paired_gains():
    mc = pd.read_csv(RES / 'model_contrasts_v1.csv')

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(mc))
    colors = [GREEN if d > 0 else RED for d in mc.mean_difference]
    ax.barh(y, mc.mean_difference, color=colors, alpha=0.8, height=0.6)
    ax.errorbar(mc.mean_difference, y,
                xerr=[mc.mean_difference - mc.corrected_resampling_low,
                      mc.corrected_resampling_high - mc.mean_difference],
                fmt='none', color='black', capsize=4, lw=1.3)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{r.left} vs {r.right}' for _, r in mc.iterrows()], fontsize=9)
    ax.set_xlabel('Mean AUC difference (25 seed×fold runs)')
    ax.set_title('Model contrast: paired AUC gains (V1-corrected)\n'
                 'Bars: mean diff; error bars: corrected resampling 95% CI')
    # Win/loss annotation
    for i, (_, row) in enumerate(mc.iterrows()):
        ax.text(ax.get_xlim()[1] * 0.98, i,
                f'{int(row.wins)}/25 wins', va='center', ha='right', fontsize=7.5, color=GREY)
    fig.tight_layout()
    savefig(fig, '02_original_paired_gains.png', pdf=True)

# ── 10. 03_rate_decomposition.png/.pdf ──────────────────────────────────────
def fig_03_rate_decomp():
    rd = pd.read_csv(RES / 'rate_decomposition_v1.csv')
    labels = {'log_r': 'Overall word rate',
              'log_a': 'Pause-excluded rate',
              'log_nonpause': 'Non-pause fraction'}
    rd['label'] = rd.component.map(labels)
    rd['pct']   = rd.geometric_percent_change
    ci_low  = (np.expm1(rd['low']) * 100).values
    ci_high = (np.expm1(rd['high']) * 100).values

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    colors = [GREEN if p > 0 else RED for p in rd.pct]
    ax.barh(rd.label, rd.pct, color=colors, alpha=0.85, height=0.5)
    ax.errorbar(rd.pct, np.arange(len(rd)),
                xerr=[rd.pct.values - ci_low, ci_high - rd.pct.values],
                fmt='none', color='black', capsize=3, lw=1.2)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Geometric % change (latest − earliest epoch)')
    ax.set_title(f'Speech rate decomposition (V1-corrected)\n'
                 f'n={int(rd.n_speakers.iloc[0])} endpoint speakers with ≥2 ordered epochs')
    for i, (pct, lbl) in enumerate(zip(rd.pct, rd.label)):
        ax.text(-0.5, i, f'{pct:.1f}%', va='center', ha='right', fontsize=9)
    fig.tight_layout()
    savefig(fig, '03_rate_decomposition.png', pdf=True)

# ── 11. 04_temporal_generalization.png/.pdf ──────────────────────────────────
def fig_04_temporal():
    tc = pd.read_csv(RES / 'temporal_person_concordance_v1.csv')
    rr = pd.read_csv(RES / 'relative_rate_robustness_v1.csv')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: concordance distribution
    ax = axes[0]
    ax.hist(tc.concordance, bins=np.linspace(0, 1, 11),
            color=BLUE, alpha=0.8, edgecolor='white')
    ax.axvline(tc.concordance.mean(), color=RED, lw=2, ls='--',
               label=f'Mean = {tc.concordance.mean():.3f}')
    ax.axvline(0.5, color=GREY, lw=1.2, ls=':', label='Chance (0.50)')
    ax.set_xlabel('Concordance per speaker per fold')
    ax.set_ylabel('Count')
    ax.set_title(f'Held-out temporal concordance (V1-corrected)\n'
                 f'n={len(tc)} speaker×fold pairs; outer-test only')
    ax.legend(fontsize=8)

    # Right: relative rate robustness (sign-flip p)
    ax = axes[1]
    colors = [GREEN if m < 0 else RED for m in rr['mean']]
    ax.barh(rr.feature, rr['mean'], color=colors, alpha=0.8, height=0.6)
    ax.errorbar(rr['mean'], np.arange(len(rr)),
                xerr=[rr['mean'] - rr.min_leave_one_out_mean,
                      rr.max_leave_one_out_mean - rr['mean']],
                fmt='none', color='black', capsize=3, lw=1)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Mean log change (endpoint speakers)')
    ax.set_title('Relative rate robustness (LOO range)\n(V1-corrected)')
    for i, (_, row) in enumerate(rr.iterrows()):
        ax.text(ax.get_xlim()[1] * 1.02, i,
                f'q={row.q_bh_4:.3f}', va='center', fontsize=7.5, color=GREY)

    fig.tight_layout()
    savefig(fig, '04_temporal_generalization.png', pdf=True)

# ── 12. 05_attribution_normalization.png/.pdf ────────────────────────────────
def fig_05_attribution():
    sb  = pd.read_csv(RES / 'shap_speaker_bootstrap_v1.csv')
    sf  = pd.read_csv(RES / 'shap_family_uncertainty_v1.csv')
    top = sb.head(15).sort_values('mean_abs_shap_speaker')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: top features by bootstrapped speaker mean |SHAP|
    ax = axes[0]
    TIMING = {'speech_rate', 'pause_mean_s', 'pause_ratio'}
    LING   = {'mattr50', 'filler_rate', 'immediate_rep_rate', 'bigram_rep', 'mean_sent_len'}
    def fcolor(f):
        if f in TIMING: return AMBER
        if f in LING:   return TEAL
        return BLUE
    colors = [fcolor(f) for f in top.feature]
    ax.barh(top.feature, top.mean_abs_shap_speaker, color=colors, alpha=0.85, height=0.65)
    ax.errorbar(top.mean_abs_shap_speaker, np.arange(len(top)),
                xerr=[top.mean_abs_shap_speaker - top.low,
                      top.high - top.mean_abs_shap_speaker],
                fmt='none', color='black', capsize=3, lw=1)
    ax.set_xlabel('Mean |SHAP| (speaker-averaged, 5000 bootstrap)')
    ax.set_title('Feature attribution – top 15 (V1-corrected)')
    patches = [mpatches.Patch(color=AMBER, label='Timing'),
               mpatches.Patch(color=TEAL,  label='Linguistic'),
               mpatches.Patch(color=BLUE,  label='Acoustic')]
    ax.legend(handles=patches, fontsize=8)

    # Right: family share pie/bar
    ax = axes[1]
    fam_colors = {'Timing3': AMBER, 'Language5': TEAL, 'Voice6': RED, 'Spectral23': BLUE}
    ax.barh(sf.family, sf.share * 100,
            xerr=[(sf.share - sf.share_low) * 100, (sf.share_high - sf.share) * 100],
            color=[fam_colors.get(f, GREY) for f in sf.family],
            alpha=0.85, capsize=4, height=0.5)
    ax.set_xlabel('Attribution share (%) with 95% bootstrap CI')
    ax.set_title('Feature family attribution share (V1-corrected)')
    for i, (_, row) in enumerate(sf.iterrows()):
        ax.text(row.share * 100 + 0.5, i, f'{row.share*100:.1f}%',
                va='center', fontsize=8.5)

    fig.tight_layout()
    savefig(fig, '05_attribution_normalization.png', pdf=True)

# ── 13. 06_score_associations.png/.pdf ──────────────────────────────────────
def fig_06_score_associations():
    # Spearman of held-out progression score vs speaker-averaged features
    # Using within_speaker_rank_associations for rank-based effects
    ws = pd.read_csv(RES / 'within_speaker_rank_associations_v1.csv')
    unadj = ws[~ws.adjusted].copy().set_index('feature')
    adj   = ws[ws.adjusted].copy().set_index('feature')

    # Focus on prespecified features
    TIMING = ['speech_rate', 'pause_mean_s', 'pause_ratio']
    LING   = ['mattr50', 'filler_rate', 'immediate_rep_rate', 'bigram_rep', 'mean_sent_len']
    focus  = [f for f in TIMING + LING if f in unadj.index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: unadjusted within-speaker rank beta
    ax = axes[0]
    vals = unadj.loc[focus, 'beta_rank']
    colors = [GREEN if b > 0 else RED for b in vals]
    y = np.arange(len(focus))
    ax.barh(y, vals, color=colors, alpha=0.8, height=0.6)
    ax.errorbar(vals, y,
                xerr=[vals - unadj.loc[focus, 'low'],
                      unadj.loc[focus, 'high'] - vals],
                fmt='none', color='black', capsize=3, lw=1.1)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(focus)
    ax.set_xlabel('β (rank ~ feature, within-speaker FE)')
    ax.set_title('Within-speaker longitudinal associations\n(unadjusted) [V1-corrected]')
    for i, f in enumerate(focus):
        q = unadj.loc[f, 'q_bh_37']
        ax.text(ax.get_xlim()[1] * 0.98, i, f'q={q:.3f}',
                va='center', ha='right', fontsize=7.5, color=GREY)

    # Right: quality-adjusted
    ax = axes[1]
    vals_a = adj.loc[focus, 'beta_rank']
    colors_a = [GREEN if b > 0 else RED for b in vals_a]
    ax.barh(y, vals_a, color=colors_a, alpha=0.8, height=0.6)
    ax.errorbar(vals_a, y,
                xerr=[vals_a - adj.loc[focus, 'low'],
                      adj.loc[focus, 'high'] - vals_a],
                fmt='none', color='black', capsize=3, lw=1.1)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(focus)
    ax.set_xlabel('β (rank ~ feature, within-speaker FE, quality-adjusted)')
    ax.set_title('Within-speaker longitudinal associations\n(quality-adjusted) [V1-corrected]')
    for i, f in enumerate(focus):
        q = adj.loc[f, 'q_bh_37']
        ax.text(ax.get_xlim()[1] * 0.98, i, f'q={q:.3f}',
                va='center', ha='right', fontsize=7.5, color=GREY)

    fig.tight_layout()
    savefig(fig, '06_score_associations.png', pdf=True)

# ── 14. 07_supplementary_probes.png/.pdf ────────────────────────────────────
def fig_07_probes():
    pr = pd.read_csv(RES / 'supplementary_probe_summary_v1.csv')
    order = ['Timing3', 'Language5', 'Voice6', 'Spectral23', 'TimingLanguage8',
             'All37', 'Quality6', 'TimingQuality9', 'AllQuality43', 'Gender1']
    pr = pr[pr.model.isin(order)].set_index('model').loc[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(order))
    colors = [BLUE if 'Quality' not in m and m != 'Gender1' else ORANGE
              for m in order]
    ax.barh(y, pr.auc, xerr=[(pr.auc - pr.auc_low), (pr.auc_high - pr.auc)],
            color=colors, alpha=0.85, capsize=3, height=0.6)
    ax.axvline(0.5, ls='--', lw=1, color='black', label='Chance')
    ax.axvline(0.610, ls=':', lw=1.3, color=ORANGE, alpha=0.7,
               label='Quality baseline V1 (0.610)')
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel('AUC (speaker-level 5-fold CV, 5 seeds)')
    ax.set_title('Fixed-specification logistic probe AUCs (V1-corrected)\n'
                 'n=141 speakers; bars = mean ± 95% bootstrap CI')
    p1 = mpatches.Patch(color=BLUE,   label='Speech features')
    p2 = mpatches.Patch(color=ORANGE, label='Quality / gender')
    ax.legend(handles=[p1, p2, plt.Line2D([0], [0], ls='--', color='black', label='Chance'),
                        plt.Line2D([0], [0], ls=':', color=ORANGE, label='Quality baseline')],
              fontsize=8)
    fig.tight_layout()
    savefig(fig, '07_supplementary_probes.png', pdf=True)

# ── 15. 08_state_structure.png/.pdf ─────────────────────────────────────────
def fig_08_state_structure():
    sa  = pd.read_csv(RES / 'states_assignments_v1.csv')
    bic = pd.read_csv(RES / 'states_bic_v1.csv')
    ks  = pd.read_csv(RES / 'states_kselect_v1.csv')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: k-selection histogram
    ax = axes[0]
    kc = ks.best_k.value_counts().sort_index()
    ax.bar(kc.index, kc.values, color=BLUE, alpha=0.85, edgecolor='white')
    ax.set_xlabel('Selected k (BIC-optimal, no forced minimum)')
    ax.set_ylabel('Number of folds')
    ax.set_title('GMM component selection\nacross 25 seed×fold runs [V1-corrected]')
    ax.set_xticks(sorted(kc.index))
    for x_, y_ in zip(kc.index, kc.values):
        ax.text(x_, y_ + 0.2, str(y_), ha='center', va='bottom', fontsize=9)

    # Panel 2: BIC curves
    ax = axes[1]
    bic_m = bic.groupby('k')['bic'].mean()
    for (seed, fold), g in bic.groupby(['seed', 'fold']):
        ax.plot(g.k, g.bic, color=GREY, alpha=0.2, lw=0.6)
    ax.plot(bic_m.index, bic_m.values, 'o-', color=BLUE, lw=2.5, label='Mean BIC')
    ax.set_xlabel('k'); ax.set_ylabel('BIC')
    ax.set_title('BIC by k\n(grey: individual folds)')
    ax.legend(fontsize=8); ax.set_xticks(bic_m.index)

    # Panel 3: state label purity (dementia fraction per state across folds where k>1)
    ax = axes[2]
    multi = sa[sa.best_k > 1].copy()
    if len(multi) > 0:
        purity = multi.groupby('state').apply(
            lambda g: g.label.mean()).reset_index(name='pct_dementia')
        ax.bar(purity.state, purity.pct_dementia * 100,
               color=[TEAL, AMBER, RED][:len(purity)], alpha=0.85, edgecolor='white')
        ax.axhline(multi.label.mean() * 100, ls='--', color=GREY,
                   label=f'Overall: {multi.label.mean()*100:.0f}%')
        ax.set_xlabel('State (operational partition)')
        ax.set_ylabel('% dementia-associated recordings')
        ax.set_title('Dementia fraction per state\n(folds with k>1; labels = State-0,1 only)')
        ax.legend(fontsize=8)
        ax.set_xticks(purity.state)
        ax.set_xticklabels([f'State-{s}' for s in purity.state])
    else:
        ax.text(0.5, 0.5, 'All folds selected k=1\n(no multi-state structure)',
                ha='center', va='center', transform=ax.transAxes)

    fig.tight_layout()
    savefig(fig, '08_state_structure.png', pdf=True)


if __name__ == '__main__':
    print('Regenerating all stale figures with V1-corrected data...\n')
    fig_model_comparison()
    fig_state_distribution()
    fig_feature_importance()
    fig_temporal_pair_acc()
    fig_bic_silhouette()
    fig_shap_beeswarm()
    fig_trajectories()
    fig_01_speaker_effects()
    fig_02_paired_gains()
    fig_03_rate_decomp()
    fig_04_temporal()
    fig_05_attribution()
    fig_06_score_associations()
    fig_07_probes()
    fig_08_state_structure()
    print('\nAll figures regenerated.')
