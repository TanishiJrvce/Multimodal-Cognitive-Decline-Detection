"""Fixed-specification speaker-level logistic probes and OOF score structure analysis (V1-corrected).

Changes from original (12_model_probes.py):
- Registers group_analysis_v1 in sys.modules before star-import (fixes dynamic-import bug)
- Imports from 11_group_analysis_v1 (V1-corrected shared state)
- Uses states_assignments_v1.csv for score structure analysis
- Saves all outputs with _v1 suffix
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "group_analysis",
    Path(__file__).resolve().parent / "11_group_analysis_v1.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["group_analysis"] = _mod   # FIX: register before exec so star-import works
_spec.loader.exec_module(_mod)
from group_analysis import *  # noqa: F401,F403

from sklearn.metrics import roc_curve
from scipy.stats import rankdata

def auc_boot(y, s, n=5000):
    i1 = np.flatnonzero(y == 1); i0 = np.flatnonzero(y == 0); out = []
    for _ in range(n):
        idx = np.r_[RNG.choice(i1, len(i1)), RNG.choice(i0, len(i0))]
        out.append(roc_auc_score(y[idx], s[idx]))
    return np.quantile(out, [.025, .975])

def predict(X, y, seed):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        model = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LogisticRegression(C=.1, class_weight='balanced', solver='liblinear',
                               max_iter=2000, random_state=seed))
        model.fit(X[tr], y[tr]); p[te] = model.predict_proba(X[te])[:, 1]
    return p

def probes():
    df, _ = load()
    sp = df.groupby('speaker')[FEATURES + Q + ['label', 'female']].mean()
    y = sp.label.values.astype(int)
    families = {
        'Timing3': TIMING, 'Language5': LING, 'Voice6': VOICE, 'Spectral23': SPEC,
        'TimingLanguage8': TIMING + LING, 'All37': FEATURES, 'Quality6': Q,
        'TimingQuality9': TIMING + Q, 'AllQuality43': FEATURES + Q, 'Gender1': ['female']
    }
    preds = {}; rows = []; seeds = []
    for name, cols in families.items():
        X = sp[cols].values
        p = np.stack([predict(X, y, s) for s in range(5)]); preds[name] = p
        for s in range(5):
            seeds.append(dict(model=name, seed=s, auc=roc_auc_score(y, p[s]),
                              bac=balanced_accuracy_score(y, p[s] > .5),
                              brier=brier_score_loss(y, p[s])))
        mean = p.mean(0); lo, hi = auc_boot(y, mean)
        fpr, tpr, t = roc_curve(y, mean)
        idx = np.flatnonzero(fpr <= .2)[-1]
        rows.append(dict(model=name, n_features=len(cols), n_speakers=len(y),
                         auc=roc_auc_score(y, mean), auc_low=lo, auc_high=hi,
                         bac=balanced_accuracy_score(y, mean > .5), brier=brier_score_loss(y, mean),
                         sensitivity_at_empirical_80specificity=tpr[idx],
                         specificity_at_that_threshold=1 - fpr[idx], threshold=t[idx]))
        print('PROBE', name, rows[-1], flush=True)
    save(pd.DataFrame(rows), 'supplementary_probe_summary')
    save(pd.DataFrame(seeds), 'supplementary_probe_runs')
    out = pd.DataFrame({'speaker_id': ['S%03d' % i for i in range(len(y))], 'label': y})
    for name, p in preds.items():
        for s in range(5): out[f'{name}_seed{s}'] = p[s]
    save(out, 'supplementary_oof_predictions')
    # Paired conditional bootstrap difference of two frozen OOF score vectors.
    contrasts = []; i1 = np.flatnonzero(y == 1); i0 = np.flatnonzero(y == 0)
    for l, r in [('All37', 'Quality6'), ('TimingLanguage8', 'Timing3'),
                  ('AllQuality43', 'Quality6'), ('TimingQuality9', 'Quality6')]:
        z = []; left = preds[l].mean(0); right = preds[r].mean(0)
        for _ in range(5000):
            ix = np.r_[RNG.choice(i1, len(i1)), RNG.choice(i0, len(i0))]
            z.append(roc_auc_score(y[ix], left[ix]) - roc_auc_score(y[ix], right[ix]))
        contrasts.append(dict(left=l, right=r,
                              auc_difference=roc_auc_score(y, left) - roc_auc_score(y, right),
                              low=np.quantile(z, .025), high=np.quantile(z, .975)))
    save(pd.DataFrame(contrasts), 'supplementary_probe_contrasts')
    # Full-pipeline permutation: refit scaler and classifier after label shuffle, seed 0 only.
    null = []
    for name in ['Timing3', 'All37']:
        X = sp[families[name]].values; observed = roc_auc_score(y, preds[name][0]); values = []
        for b in range(499):
            yp = RNG.permutation(y); v = roc_auc_score(yp, predict(X, yp, 0))
            values.append(v); null.append(dict(model=name, iteration=b, auc=v))
        pv = (1 + sum(v >= observed for v in values)) / 500
        print('PIPELINE PERMUTATION', name, observed, pv, flush=True)
        save(pd.DataFrame([dict(model=name, observed_seed0_auc=observed, permutation_p=pv,
                                null_mean=np.mean(values), null_q95=np.quantile(values, .95))]),
             'permutation_' + name)
    save(pd.DataFrame(null), 'supplementary_pipeline_permutations')

def score_structure():
    """Score structure analysis using V1 predictions from states_assignments_v1."""
    # states_assignments_v1 has split='train' (outer-train) and split='val' (outer-test),
    # both on the same prog_score scale — no cross-scale mixing, no proxy.
    states = read('data/results/states_assignments_v1.csv')
    df, m = load()
    o = read('data/processed/recording_order.csv').rename(columns={'recording': 'file', 'split': 'original_split'})

    # Percentile rank each outer-test score against its fold's outer-training distribution
    parts = []
    for (seed, fold), g in states.groupby(['seed', 'fold']):
        training = np.sort(g[g.split == 'train']['prog_score'].values)
        test_g = g[g.split == 'val'].copy()
        test_g['score_percentile'] = (
            np.searchsorted(training, test_g.prog_score, side='left') +
            0.5 * np.array([np.sum(training == x) for x in test_g.prog_score])
        ) / max(len(training), 1)
        parts.append(test_g)

    a = pd.concat(parts)
    p = (a.groupby('file').agg(score=('score_percentile', 'mean'),
                                 score_sd=('score_percentile', 'std'),
                                 label=('label', 'first'),
                                 speaker=('speaker', 'first')).reset_index())
    joined = p.merge(df.drop(columns=['speaker', 'label'], errors='ignore'), on='file', how='left')

    sp = joined.groupby('speaker')[['score', 'label']].mean()
    y = sp.label.to_numpy(int); s = sp.score.values
    lo, hi = auc_boot(y, s)
    rows = [dict(level='speaker', auc=roc_auc_score(y, s), low=lo, high=hi, n=len(y))]
    rows.append(dict(level='recording', auc=roc_auc_score(joined.label, joined.score),
                     low=np.nan, high=np.nan, n=len(joined)))
    save(pd.DataFrame(rows), 'progression_score_group_discrimination')
    print('PROGRESSION SCORE AUC', rows, flush=True)

    # Temporal concordance from held-out scores only (prog_score, same scale)
    pairrows = []; personrows = []
    aa = a.merge(o[['speaker', 'file', 'rank']], on=['speaker', 'file'], how='inner')
    bucket = (aa.groupby(['seed', 'fold', 'speaker', 'rank'])
                .agg(score=('prog_score', 'mean')).reset_index())
    for (seed, fold, speaker), g in bucket.groupby(['seed', 'fold', 'speaker']):
        g = g.sort_values('rank')
        if len(g) < 2: continue
        concord = []
        for r1, r2 in itertools.combinations(g.to_dict('records'), 2):
            hit = float(r2['score'] > r1['score']) + .5 * float(r2['score'] == r1['score'])
            concord.append(hit)
            pairrows.append(dict(seed=seed, fold=fold, speaker=speaker, hit=hit,
                                  rankgap=r2['rank'] - r1['rank']))
        personrows.append(dict(seed=seed, fold=fold, speaker=speaker,
                                concordance=np.mean(concord),
                                endpoint_hit=float(g.iloc[-1].score > g.iloc[0].score),
                                n_pairs=len(concord)))
    save(pd.DataFrame(pairrows), 'temporal_pair_concordance')
    save(pd.DataFrame(personrows), 'temporal_person_concordance')
    print('TEMPORAL concordance (heldout)', pd.DataFrame(personrows).concordance.describe().round(3).to_dict(), flush=True)

if __name__ == '__main__':
    probes()
    score_structure()
