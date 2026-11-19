"""Relative-rate robustness, repetition sensitivity, timing identity checks (V1-corrected).

Changes from original (14_robustness.py):
- Fixed broken import: original file had literal backtick-n (`n) sequences instead of
  newlines (Windows line-ending corruption from ZIP). Rewritten with proper Python syntax.
- Registers group_analysis in sys.modules before star-import (fixes dynamic-import bug)
- Imports from 11_group_analysis_v1 (V1-corrected shared state)
- Uses joint_predictions_v1.csv (V_joint, outer-test only) instead of states_assignments.csv
  which mixed training data into concordance calculations
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

from scipy.stats import trim_mean

def run():
    df, m = load()
    order = read('data/processed/recording_order.csv').rename(columns={'recording': 'file'})
    d = order[['file', 'speaker', 'rank']].merge(df.drop(columns='speaker'), on='file')

    for c, col in [('log_speech_rate', 'speech_rate'), ('log_articulation_rate', 'articulation_rate')]:
        d[c] = np.log(d[col])
    d['log_nonpause'] = np.log1p(-d.pause_ratio)
    cols = ['log_speech_rate', 'log_articulation_rate', 'log_nonpause', 'log_duration']
    buck = d.groupby(['speaker', 'rank'])[cols].mean().reset_index()
    rows = []
    for s, g in buck.groupby('speaker'):
        g = g.sort_values('rank')
        if len(g) < 2:
            continue
        z = g.iloc[-1][cols] - g.iloc[0][cols]
        rows.append(z.to_dict())
    ch = pd.DataFrame(rows)
    out = []
    signs = RNG.choice([-1, 1], (10000, len(ch)))
    for c in cols:
        x = ch[c].to_numpy()
        null = (signs * x).mean(1)
        loo = np.array([np.delete(x, i).mean() for i in range(len(x))])
        out.append(dict(
            feature=c,
            mean=x.mean(),
            median=np.median(x),
            trimmed_mean20=trim_mean(x, .2),
            min_leave_one_out_mean=loo.min(),
            max_leave_one_out_mean=loo.max(),
            p_signflip=(1 + (abs(null) >= abs(x.mean())).sum()) / 10001,
            wilcoxon_p=stats.wilcoxon(x).pvalue,
            n_decrease=int((x < 0).sum())
        ))
    rr = pd.DataFrame(out)
    rr['q_bh_4'] = bh(rr.p_signflip)
    save(rr, 'relative_rate_robustness')
    print(rr.round(5).to_string(index=False), flush=True)

    # Duration-dependent repetition: correlation after residualising word count and quality.
    # Duration-dependent repetition: correlation after residualising word count and quality.
    # Use states_assignments_v1 (split='val' = outer-test only) with training distribution
    # from split='train' — same prog_score scale, no proxy, no leakage.
    states = read('data/results/states_assignments_v1.csv')
    score_parts = []
    for (seed, fold), g in states.groupby(['seed', 'fold']):
        training = np.sort(g[g.split == 'train']['prog_score'].values)
        test_g = g[g.split == 'val'].copy()
        test_g['pct'] = (np.searchsorted(training, test_g.prog_score, side='left') +
                          0.5 * np.array([np.sum(training == x) for x in test_g.prog_score])) / max(len(training), 1)
        score_parts.append(test_g)
    sc = pd.concat(score_parts).groupby('file').pct.mean().rename('pct')
    z = (df.join(sc, on='file')
           .groupby('speaker')[['pct', 'bigram_rep', 'n_words', 'log_duration'] + Q[1:]]
           .mean().dropna())

    def partial(a, b, covs):
        C = np.column_stack([np.ones(len(z)), stats.rankdata(z[covs], axis=0)])
        P2 = np.linalg.pinv(C)
        x2 = stats.rankdata(z[a]); y2 = stats.rankdata(z[b])
        return stats.pearsonr(x2 - C @ (P2 @ x2), y2 - C @ (P2 @ y2)).statistic

    vals = [
        dict(adjustment='none', rho=stats.spearmanr(z.pct, z.bigram_rep).statistic, n=len(z)),
        dict(adjustment='word_count', rho=partial('pct', 'bigram_rep', ['n_words']), n=len(z)),
        dict(adjustment='quality_and_word_count', rho=partial('pct', 'bigram_rep', Q + ['n_words']), n=len(z))
    ]
    save(pd.DataFrame(vals), 'repetition_sampling_sensitivity')

    # Theoretical exact identity: speech_rate = articulation_rate * (1 - pause_ratio)
    err = np.max(np.abs(df.speech_rate - df.articulation_rate * (1 - df.pause_ratio)))
    save(pd.DataFrame([dict(
        check='speech_rate = articulation_rate * (1 - pause_ratio)',
        max_absolute_error=err,
        n_recordings=len(df)
    )]), 'timing_identity')
    print('TIMING IDENTITY max_absolute_error', err, flush=True)

if __name__ == '__main__':
    run()
