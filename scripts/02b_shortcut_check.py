"""Shortcut check: can recording-quality features alone predict group or temporal bucket?"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, spearmanr

m = pd.read_csv("data/processed/manifest.csv").merge(pd.read_csv("data/processed/quality.csv"), on="file")
Q = ["orig_sr", "ch_corr", "duration_s", "loudness_dbfs", "snr_db", "bw99_hz", "peak"]
m["stereo_true"] = (m.ch_corr < 0.99).astype(int)
Q.append("stereo_true")
X = m[Q].fillna(0).values

print("== group (dementia=1) vs quality, medians ==")
print(m.groupby("label")[Q].median().round(2).T)
for q in Q:
    p = mannwhitneyu(m[m.label == 1][q].fillna(0), m[m.label == 0][q].fillna(0)).pvalue
    print(f"  {q:14s} MWU p={p:.4f}")

def cv_auc(y, groups, model, n=5, seeds=5):
    aucs = []
    for s in range(seeds):
        p = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(n, shuffle=True, random_state=s).split(X, y, groups):
            model.fit(X[tr], y[tr]); p[te] = model.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p))
    return np.mean(aucs), np.std(aucs)

lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
rf = RandomForestClassifier(300, min_samples_leaf=3, random_state=0)
y, g = m.label.values, m.speaker.values
print("\nQuality-only GROUP classification, recording-level AUC (speaker-grouped CV):")
print("  logistic: %.3f +- %.3f" % cv_auc(y, g, lr)); print("  forest:   %.3f +- %.3f" % cv_auc(y, g, rf))

d = m[(m.label == 1) & m.suffix.isin([0, 5, 10, 15])].copy()
print("\n== dementia recordings: quality vs time bucket (15=earliest ... 0=after symptoms) ==")
print(d.groupby("suffix")[Q].median().round(2).T)
for q in Q:
    r, p = spearmanr(-d.suffix, d[q].fillna(0)); print(f"  {q:14s} spearman(later, q)={r:+.2f} p={p:.3f}")
early = d[d.suffix.isin([10, 15])]; late = d[d.suffix.isin([0, 5])]
dd = pd.concat([early.assign(t=0), late.assign(t=1)])
X = dd[Q].fillna(0).values
print("Quality-only early(10/15) vs late(0/5): AUC %.3f +- %.3f" % cv_auc(dd.t.values, dd.speaker.values, rf))
