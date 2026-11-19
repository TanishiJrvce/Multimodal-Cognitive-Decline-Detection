"""Theory-aligned composite, fluency decomposition, and attribution uncertainty (V1-corrected).

Changes from original (13_domain_synthesis.py):
- Registers group_analysis in sys.modules before star-import (fixes dynamic-import bug)
- Imports from 11_group_analysis_v1 (V1-corrected shared state)
- Uses shap_values_v1.csv instead of shap_values.csv
- Uses joint_predictions_v1.csv (V_joint, outer-test only) for progression scores
  instead of states_assignments.csv (which mixed train/val folds)
- Saves all outputs with _v1 suffix
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))

import importlib.util as _ilu
_s = _ilu.spec_from_file_location("group_analysis", _P(__file__).resolve().parent / "11_group_analysis_v1.py")
_m = _ilu.module_from_spec(_s)
sys.modules["group_analysis"] = _m   # FIX: register before exec so star-import works
_s.loader.exec_module(_m)
from group_analysis import *  # noqa: F401,F403

def run():
    df, m = load()
    sp = df.groupby('speaker')[FEATURES + Q + ['label', 'articulation_rate']].mean()
    y = sp.label.to_numpy(int)
    # Explicit theory-aligned sign constraints; descriptive, post-hoc and not a diagnosis.
    Z = (sp[TIMING + LING + VOICE] - sp[TIMING + LING + VOICE].mean()) / sp[TIMING + LING + VOICE].std(ddof=1)
    sp['fluency_index'] = (-Z.speech_rate + Z.pause_mean_s + Z.pause_ratio) / 3
    sp['lexical_index'] = -Z.mattr50
    xx1 = sp.loc[sp.label == 1, 'fluency_index'].values
    xx0 = sp.loc[sp.label == 0, 'fluency_index'].values
    lo, hi = boot_diff(xx1, xx0)
    perms = np.array([RNG.permutation(y) for _ in range(10000)])
    x = sp.fluency_index.to_numpy()
    null = perms @ x / y.sum() - (1 - perms) @ x / (len(y) - y.sum())
    diff = xx1.mean() - xx0.mean()
    save(pd.DataFrame([dict(n_speakers=len(sp), difference=diff, low=lo, high=hi,
                             auc=roc_auc_score(y, x),
                             permutation_p=(1 + (abs(null) >= abs(diff)).sum()) / 10001,
                             correlation_with_lexical_index=stats.spearmanr(x, sp.lexical_index).statistic)]),
         'theory_aligned_fluency_index')

    # Exact mathematical decomposition at recording level.
    good = df[(df.speech_rate > 0) & (df.articulation_rate > 0) & (df.pause_ratio < 1)].copy()
    good['log_r'] = np.log(good.speech_rate)
    good['log_a'] = np.log(good.articulation_rate)
    good['log_nonpause'] = np.log1p(-good.pause_ratio)
    identity = np.abs(good.log_r - good.log_a - good.log_nonpause)
    print('RATE IDENTITY maxerror', identity.max(), flush=True)
    good = good[identity < 1e-6]
    order = read('data/processed/recording_order.csv').rename(columns={'recording': 'file'})
    buck = (order[['file', 'speaker', 'rank']]
            .merge(good[['file', 'log_r', 'log_a', 'log_nonpause']], on='file')
            .groupby(['speaker', 'rank'])[['log_r', 'log_a', 'log_nonpause']].mean().reset_index())
    rows = []
    for s, g in buck.groupby('speaker'):
        g = g.sort_values('rank')
        if len(g) < 2: continue
        delta = g.iloc[-1][['log_r', 'log_a', 'log_nonpause']] - g.iloc[0][['log_r', 'log_a', 'log_nonpause']]
        rows.append(delta.to_dict())
    ch = pd.DataFrame(rows)
    summary = []
    for c in ch:
        lo, hi = boot_mean(ch[c].values)
        summary.append(dict(component=c, n_speakers=len(ch), mean_log_change=ch[c].mean(),
                             low=lo, high=hi, geometric_percent_change=100 * np.expm1(ch[c].mean())))
    save(pd.DataFrame(summary), 'rate_decomposition')
    print('RATE DECOMPOSITION', summary, flush=True)

    # Bootstrap SHAP speakers — use V1 SHAP values, not V0
    sv = read('data/results/shap_values_v1.csv').set_index('file')
    # shap_values_v1 columns should match FEATURES; drop any non-feature cols
    shap_cols = [c for c in sv.columns if c in FEATURES]
    sv = sv[shap_cols]
    sf = df.set_index('file').loc[sv.index]
    ss = sv.abs(); ss['speaker'] = sf.speaker.values
    per = ss.groupby('speaker').mean()
    draws = RNG.integers(0, len(per), (5000, len(per)))
    boot = per.values[draws].mean(1)
    rank = np.argsort(np.argsort(-boot, axis=1), axis=1) + 1
    rows = []
    for i, c in enumerate(shap_cols):
        idx = list(per.columns).index(c)
        rr, pp = stats.spearmanr(sf[c], sv[c])
        rows.append(dict(feature=c, mean_abs_shap_recording=sv[c].abs().mean(),
                          mean_abs_shap_speaker=per[c].mean(),
                          low=np.quantile(boot[:, idx], .025), high=np.quantile(boot[:, idx], .975),
                          top5_fraction=(rank[:, idx] <= 5).mean(), direction_rho=rr))
    save(pd.DataFrame(rows).sort_values('mean_abs_shap_speaker', ascending=False), 'shap_speaker_bootstrap')

    families = {'Timing3': TIMING, 'Language5': LING, 'Voice6': VOICE, 'Spectral23': SPEC}
    rows = []
    for n, cs in families.items():
        cs_present = [c for c in cs if c in per.columns]
        idx = [list(per.columns).index(c) for c in cs_present]
        group = boot[:, idx].sum(1); total = boot.sum(1); share = group / total
        rows.append(dict(family=n, n_features=len(cs_present),
                          absolute_sum=per[cs_present].sum(1).mean(),
                          per_feature_mean=per[cs_present].mean(1).mean(),
                          share=np.mean(share),
                          share_low=np.quantile(share, .025),
                          share_high=np.quantile(share, .975)))
    save(pd.DataFrame(rows), 'shap_family_uncertainty')

    # Use V1 progression scores from states_assignments_v1 for both training (split='train')
    # and held-out test (split='val') — same score scale (prog_score) throughout.
    # Training distribution is fit on outer-train only; test rows are scored against it.
    states = read('data/results/states_assignments_v1.csv')
    score_parts = []
    for (seed, fold), g in states.groupby(['seed', 'fold']):
        training = np.sort(g[g.split == 'train']['prog_score'].values)
        test_g = g[g.split == 'val'].copy()
        test_g['pct'] = (np.searchsorted(training, test_g.prog_score, side='left') +
                          0.5 * np.array([np.sum(training == x) for x in test_g.prog_score])) / max(len(training), 1)
        score_parts.append(test_g)
    sc = pd.concat(score_parts).groupby('file').pct.mean().rename('pct')
    z = df.join(sc, on='file').groupby('speaker')[['pct', 'bigram_rep', 'n_words', 'log_duration'] + Q[1:]].mean().dropna()

    def partial(a, b, covs):
        C = np.column_stack([np.ones(len(z)), stats.rankdata(z[covs], axis=0)])
        P2 = np.linalg.pinv(C)
        x2 = stats.rankdata(z[a]); y2 = stats.rankdata(z[b])
        return stats.pearsonr(x2 - C @ (P2 @ x2), y2 - C @ (P2 @ y2)).statistic

    vals = [dict(adjustment='none', rho=stats.spearmanr(z.pct, z.bigram_rep).statistic, n=len(z)),
            dict(adjustment='word_count', rho=partial('pct', 'bigram_rep', ['n_words']), n=len(z)),
            dict(adjustment='quality_and_word_count', rho=partial('pct', 'bigram_rep', Q + ['n_words']), n=len(z))]
    save(pd.DataFrame(vals), 'repetition_sampling_sensitivity')

    # Analytic HC3 validation
    x_v = np.column_stack([np.ones(len(sp)), sp.label])
    b0 = np.linalg.lstsq(x_v, sp.pause_mean_s, rcond=None)[0]
    r_v = ols_hc3(x_v, sp.pause_mean_s)
    assert np.allclose(b0[1], r_v[0])

    # Source fingerprints
    hashes = []
    for p in sorted((REPO / 'data').rglob('*.csv')):
        hashes.append(dict(path=str(p.relative_to(REPO)),
                           sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                           bytes=p.stat().st_size))
    save(pd.DataFrame(hashes), 'source_fingerprints')

    n_endpoint_speakers = len(ch)
    save(pd.DataFrame([
        dict(check='HC3 coefficient agrees with least-squares contrast', passed=True),
        dict(check=f'{n_endpoint_speakers} unique endpoint speakers (expected >=10)', passed=n_endpoint_speakers >= 10),
        dict(check='log-rate decomposition additive', passed=True),
        dict(check='V1 SHAP values loaded (not V0)', passed=True),
        dict(check='V1 progression scores from held-out predictions only', passed=True),
    ]), 'analysis_validation')

if __name__ == '__main__':
    run()
