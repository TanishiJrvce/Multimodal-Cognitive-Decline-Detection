# V0 deep dive: reproducible secondary analyses

This package analyses the final V0 study without changing its implementation.
Original source: https://github.com/TanishiJrvce/Multimodal-Cognitive-Decline-Detection/tree/V0
Pinned commit: `a391a57e5fe127dd44e211ffe6bd19e0d5777ac0`.
Neither main nor the draft was used.

The PDF explains the results and domain interpretation. `Manuscript_Results_Discussion.md`
provides editable manuscript text. `tables/` contains 31 complete numerical result tables;
`figures/` contains eight PNG/PDF figure pairs. A source-input CSV snapshot is supplied as
the sibling directory `cognitive-decline-v0/` inside the archive. It contains public speaker
identifiers needed to link repeated recordings; newly exported person-level outputs use
anonymous IDs and do not make personal diagnostic assertions.

## Reproduce

Extract the archive preserving its two sibling directories. Install the versions listed
in `requirements.txt` if needed. Run these commands from the extracted parent directory:

```bash
python deep_dive/analysis.py
python deep_dive/model_probes.py --stage probes
python deep_dive/model_probes.py --stage scores
python deep_dive/domain_synthesis.py
python deep_dive/robustness.py
python deep_dive/figures.py
python deep_dive/build_report.py
python deep_dive/validate.py
```

Use separate Python processes for the two model_probes stages, as shown, to reproduce the
recorded random-number streams exactly. Each script starts from seed 20261119.
The analysis.py default runs group, longitudinal and original-contrast stages in that order;
running a subset resets its random-number stream and can change Monte Carlo interval endpoints
slightly. The scripts take minutes rather than needing a GPU. No raw audio, transcripts,
pretrained model download or neural checkpoint is needed for these secondary analyses.

Alternatively set `MCD_V0_REPO` to an existing pinned V0 checkout. All scripts write only
inside deep_dive or the adjacent output/pdf directory. The original V0 checkout stays unchanged.

## What was newly executed

- 37 speaker-average feature contrasts, 5,000 bootstrap resamples each, and 10,000 label permutations.
- HC3 robust conditional group models for 37 features, with a separate multiple-testing family.
- Alzheimer-labelled-subset sensitivity summaries.
- One earliest-to-latest distinct-bucket endpoint contrast for each of 31 speakers.
- 38 endpoint feature tests, plus within-person demeaned rank models across 66 buckets.
- Exact log decomposition of word-rate change into pause-excluded word rate and non-pause share.
- Leave-one-speaker-out, trimmed-mean, sign-flip and Wilcoxon sensitivity summaries.
- Training versus held-out temporal concordance using the existing scores from all 25 saved models.
- Held-out score normalisation using training-fold empirical percentiles.
- Progression-coordinate group AUC and feature/covariate characterisation.
- Word-count sensitivity of the repetition association.
- 10 fixed-specification logistic probe families, 5 folds x 5 seeds = 250 primary fits.
- 499 complete-pipeline permutations for each of two probes: 998 permutation statistics and
  4,990 fold refits. Imputation, scaling and classifier fitting are repeated after each label shuffle.
- 5,000 speaker-bootstrap summaries of existing RF explanations from 28 speakers / 55 clips.
- Per-feature-normalised attribution families, state-membership uncertainty and cross-model agreement.
- Numerical and structural validation of grouping, input fingerprints, source immutability and identities.

These are secondary investigations, not replacements for the original V0 neural experiments.
Original embeddings, transcripts, raw audio and model checkpoints were not committed; therefore
no claim of full neural retraining, neural attribution, or new neural counterfactual experiments is made.

## Estimands and interpretation

Group effects average each feature within speaker before comparing 80 dementia-associated and
61 comparison speakers. This prevents clip-rich people from receiving extra weight. Percentile
bootstrap intervals resample speakers within groups. Label-permutation p-values describe an
exchangeability null, not causal randomisation. Hedges g uses the pooled speaker-level SD.

Adjusted group models include sex indicator and six available recording covariates: log duration,
SNR proxy, bandwidth, loudness, peak amplitude and channel correlation. Complete cases are used.
HC3 covariance is computed directly with the leverage-corrected residual sandwich. Adjustment
cannot control unmeasured education, recording age, interview context or clinical disease severity.

Endpoint changes first average recordings sharing a speaker/rank bucket, then compare the last
distinct bucket with the first. This yields 31 equally weighted people, unlike the earlier
later-versus-reference table with repeated people. Rank is ordinal, not elapsed years. Within-person
rank associations demean data per person, weight people equally and use cluster-robust uncertainty.

The word-rate identity holds in all 275 rows. The feature named articulation_rate is interpreted
as a pause-excluded word-rate proxy. The reported 90.2% share is a decomposition of mean log change;
it is not a causal attribution to a motor or cognitive disorder. Relative and absolute changes,
pointwise intervals and multiplicity-adjusted tests are all retained.

Progression scores are mapped to their training-fold empirical percentile before aggregation
across models. Each recording has exactly five held-out appearances. Temporal comparisons are
made within the same model and only across distinct ranks. Runs are averaged within speaker
before resampling people. Dependence from shared neural training data is not eliminated by this
bootstrap, so its intervals are conditional, not an independent external validation.

The new logistic probes use speaker-averaged tables, fixed C=0.1, balanced classes, liblinear,
fold-fitted median imputation and scaling. They are not tuned or selected to beat the original
network. Their bootstrap AUC intervals condition on the generated OOF vectors; full-pipeline
permutation results are separate and use the seed-0 statistic. Operating points are exploratory
ROC descriptions, not independently calibrated clinical thresholds.

RF SHAP bootstraps resample explanation-sample speakers and do not refit the forest. Per-feature
family normalization changes the denominator but does not remove correlations among features.
Associations of input features with a derived model score describe representation behaviour and
cannot serve as independent clinical validation.

Multiple-testing adjustments are Benjamini-Hochberg within the documented families. Every analysis
is post-hoc and hypothesis-generating; none should be described as preregistered confirmatory evidence.
Neither a high model score nor a discovered state is a confirmed MCI or dementia diagnosis.

## Most important numerical interpretation

The final study demonstrates partially generalisable group discrimination, but its temporal head
has much weaker ordering on unseen speakers than on training speakers. Small group pause effects,
relative slowing within estimated speech spans, robust timing-feature attribution, and the duration
structure of the score offer a more precise scientific account than a single accuracy headline.
No individual handcrafted feature survives its 37-feature permutation FDR screen; no arbitrary
stage labels or clinical thresholds have been manufactured.

## Integrity

`tables/source_fingerprints.csv` records the SHA-256 and size of every committed input CSV.
`tables/analysis_validation.csv` records the validation checks. The snapshot marker identifies
the source commit. Running against a real checkout additionally verifies branch, commit and
unchanged source status. Statistical methods and claim boundaries are described in the report.
