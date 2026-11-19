# Secondary Analysis Package — V0 Deep Dive

Pinned to commit `a391a57e5fe127dd44e211ffe6bd19e0d5777ac0`.  
These scripts perform post-hoc secondary analyses of the V0 study. They **never mutate repository inputs** and require no GPU, raw audio, transcripts, or model retraining.

## What's here

| Script | What it does |
|---|---|
| `analysis.py` | Speaker-group contrasts, longitudinal endpoint effects, original model contrasts |
| `model_probes.py` | Fixed-spec logistic probes (10 feature families × 5 seeds × 5 folds), pipeline permutation tests, OOF score structure |
| `domain_synthesis.py` | Rate decomposition, fluency index synthesis |
| `robustness.py` | Sensitivity checks (subtype, repetition, relative rate) |
| `figures.py` | Reproduce all 8 figure pairs (PNG + PDF) |
| `build_report.py` | Generate the deep-dive PDF report |
| `validate.py` | Source fingerprints + integrity checks |

## Reproduce

From the repo root (after `pip install -r analysis/requirements.txt`):

```bash
python analysis/analysis.py
python analysis/model_probes.py --stage probes
python analysis/model_probes.py --stage scores
python analysis/domain_synthesis.py
python analysis/robustness.py
python analysis/figures.py
python analysis/build_report.py
python analysis/validate.py
```

> Run the two `model_probes` stages as separate Python processes to exactly reproduce recorded random-number streams. Each script seeds from `20261119`. GPU not required — runs in minutes.

## Key outputs

| Output | Location |
|---|---|
| 31 numerical result tables | `analysis/tables/` |
| 8 figure pairs (PNG + PDF) | `analysis/figures/` |
| Table index / manifest | `analysis/TABLE_INDEX.csv` |

## What was newly executed (not in the original V0 scripts)

- **37 speaker-average group contrasts** — 5,000 bootstrap resamples + 10,000 label permutations each; HC3 covariate adjustment (sex + 6 recording covariates)
- **Alzheimer-subset sensitivity** — results restricted to confirmed Alzheimer subtype
- **Within-person endpoint changes** — earliest vs. latest distinct recording bucket for 31 speakers; 38 feature tests
- **Log word-rate decomposition** — exact partition of mean log change into pause-excluded delivery (≈90%) vs. pause-share change (≈10%)
- **Within-person rank associations** — demeaned fixed-effects models with cluster-robust uncertainty
- **Temporal generalisation** — training (87.8%) vs. held-out (55.3%) concordance; five-seed OOF consensus permutation test
- **Held-out score normalisation** — training-fold percentile normalisation before aggregation; group AUC after normalisation
- **Score characterisation** — OOF feature/covariate associations, state entropy, cross-model agreement
- **10 logistic probe families** — 5 folds × 5 seeds = 250 fits; full-pipeline permutation for Timing3 and All37
- **RF SHAP bootstrap** — 5,000 speaker-resampled explanation summaries; per-feature-family normalisation
- **Integrity validation** — SHA-256 source fingerprints, grouping checks, input immutability

## Interpretation boundaries

- Group effects are speaker-level averages; label-permutation p-values describe an exchangeability null, not causal randomisation.
- Endpoint changes are longitudinal comparisons within persons — not independent cross-sectional measurements.
- Temporal concordance on held-out speakers is an internal consistency check, **not independent clinical validation**.
- No feature survives 37-feature BH FDR correction. All analyses are post-hoc and hypothesis-generating.
- No arbitrary stage labels or clinical thresholds have been manufactured. No state is labelled "MCI".
