# Explainable Multimodal Learning of Cognitive Decline Trajectories from Longitudinal Spontaneous Speech

**Tanishi Jain** · V0 · DementiaNet (MIT licence)

---

## Overview

This repository presents a multimodal framework for learning dementia-associated speech trajectories from longitudinal spontaneous speech. Rather than framing the task as binary classification, the framework jointly optimises a group-discrimination objective and a weak pairwise temporal ranking loss over within-person recording pairs, producing both a group probability and an uncalibrated progression score per recording. All analyses use the publicly available DementiaNet corpus and are conducted without access to clinical severity ratings, MMSE scores, or approval-gated data.

The central question is whether acoustic and linguistic representations, trained with minimal supervision, can capture speech changes associated with dementia and reveal reproducible intermediate patterns of speech organisation — without enforcing hard ordinal labels or clinical staging assumptions.

> **Note on corrected evaluation (V1).** Results reported here supersede an earlier version of this repository. Three implementation errors in the original evaluation have been identified and corrected: (1) modality masking logic that zeroed handcrafted features whenever the audio stream was absent, causing V_feat and V_text to receive all-zero inputs; (2) global feature screening applied before cross-validation folds were defined, which allowed test-speaker information to influence feature selection; and (3) the absence of a held-out inner validation split, which caused the same partition to be used for early stopping and final AUC estimation. All corrected scripts carry a `_v1` suffix. Original results are preserved in files without that suffix. The ASR provenance of the existing transcripts remains unresolved: the extraction configuration names `small.en` while the transcription script defaulted to `medium.en`; SHA-256 hashes of all 275 transcripts are fixed in `data/results/transcript_provenance_v1.csv`.

---

## Results

Evaluation follows strict speaker-grouped cross-validation (5 outer folds × 5 seeds, 275 recordings, 141 speakers). No speaker appears in both training and test partitions at any stage. Feature screening, imputation, and standardisation are fitted exclusively on outer-training speakers within each fold. Early stopping uses a speaker-disjoint inner validation split drawn from the outer-training set; outer-test data never influence preprocessing, epoch selection, or threshold choice.

### Model performance

| Model | Streams | AUC mean ± SD | Balanced accuracy |
|---|---|---|---|
| Quality-only baseline (RF) | SNR, loudness, duration | 0.610 ± 0.068 | — |
| SVM, handcrafted features | 37 handcrafted | 0.598 ± 0.082 | — |
| FusionNet — features only (V_feat) | 37 handcrafted | 0.543 ± 0.074 | 0.509 |
| FusionNet — WavLM + features (V_audio) | WavLM + 37 handcrafted | 0.623 ± 0.086 | 0.581 |
| FusionNet — BERT + features (V_text) | BERT + 37 handcrafted | 0.593 ± 0.089 | 0.554 |
| FusionNet — all modalities, cls only (V_cls) | WavLM + BERT + 37 handcrafted | 0.639 ± 0.084 | 0.582 |
| **FusionNet — all modalities + ranking (V_joint)** | WavLM + BERT + 37 handcrafted | **0.637 ± 0.096** | **0.574** |

Averaging predictions across a speaker's recordings yields a speaker-level AUC of **0.648 ± 0.114** for V_joint.

The quality-only baseline (0.610 ± 0.068) is a matched 5×5 cross-validated Random Forest using recording-quality covariates (log-duration, SNR, bandwidth, loudness, peak, channel correlation). Exceeding this baseline does not establish that a model is free of recording-quality confounding.

The V_feat improvement from the original 0.500 (zero AUC) to 0.543 reflects correction of the modality masking bug, which previously caused the handcrafted-feature stream to receive all-zero inputs.

### Cross-validation structure

Each of the 25 outer folds contains 220 outer-training recordings (113 speakers) and 55 outer-test recordings (28–29 speakers). The inner split uses 176 inner-training and 44 inner-validation recordings. Across all 25 folds: outer-test speakers appear zero times in inner training, zero times in inner validation, and zero times in outer training. Preprocessing for epoch selection is fitted on inner-training speakers only; final preprocessing before outer-test evaluation is fitted on outer-training speakers only.

### Temporal consistency

Held-out temporal concordance (outer-test recordings only, V_joint progression scores ranked within each fold against the fold's outer-training distribution) yielded a mean concordance of **0.619** across 155 speaker×fold pairs. Within-person recording pairs sharing the same temporal rank bucket are excluded from both training and evaluation ordering. Per-fold eligible pair counts range from 33 to 40; same-bucket pairs excluded range from 5 to 12 per fold. The ranking term contributes gradients in all 25 folds.

Temporal concordance should not be compared directly to the figure reported in the original evaluation, which included training-partition recordings in the concordance calculation.

### Speech-feature group differences

Speaker-averaged comparisons across 80 dementia-associated and 61 comparison speakers (5,000 bootstrap resamples, 10,000 label permutations) identify pause duration (Hedges g ≈ 0.288, permutation p = 0.091) and pause proportion (Hedges g ≈ 0.289, permutation p = 0.082) as the most consistently directional differences among the prespecified timing and linguistic features. The strongest raw signal is in spectral features (MFCC-13 mean: g ≈ 0.333, p = 0.053). No feature survives Benjamini–Hochberg correction at the 37-feature family level after permutation testing. Covariate-adjusted models (sex, recording duration, SNR, bandwidth, loudness, peak, channel correlation) retain the direction of pause effects at nominal significance. All group-contrast findings are descriptive and hypothesis-generating.

### Within-person longitudinal change

Among 31 speakers with two or more distinct ordered recording epochs, the geometric mean word rate declined by **9.0%** between earliest and latest available recordings (bootstrap CI: 1.0%–17.4%; sign-flip p = 0.047). An exact log decomposition attributes approximately 8.2% to the pause-excluded delivery component and 0.9% to the non-pause share, distinguishing within-speaker delivery dynamics from accumulating silence. Recording duration declined by a geometric mean of **32.1%** (sign-flip p = 0.010). These are exploratory endpoint comparisons; causal attribution to dementia-related progression is not established.

### Latent speech states

Gaussian mixture modelling of progression scores on training folds selected k by BIC without a forced minimum. BIC selected k = 1 in **7 of 25** folds, k = 2 in **15 of 25**, and k = 3 in **3 of 25**. An operational two-state summary is used for description where relevant, but this partition does not reflect a model-identified optimal structure in all folds. No state is labelled with any clinical category; states are referred to as State-0 and State-1 only.

### Feature attribution

Random Forest SHAP analysis (5,000 speaker-resampled bootstraps) identifies timing features (speech rate, pause duration, pause proportion) and spectral features as the dominant contributors to attribution mass. Bootstrap uncertainty intervals are reported per feature in `data/results/shap_speaker_bootstrap_v1.csv`. Feature family shares are reported in `data/results/shap_family_uncertainty_v1.csv`.

### Fixed-specification logistic probes

Speaker-level logistic regression probes (5-fold CV, 5 seeds, speaker-averaged features) with pre-specified feature families yield AUCs of 0.573 for Spectral-23 features, 0.553 for quality-only features, 0.543 for all 37 handcrafted features, and 0.530 for the three timing features alone. No family substantially exceeds the quality-only probe, which is consistent with recording quality functioning as a partial confounder of observed group differences.

---

## Repository structure

```
├── src/mcd/
│   ├── models.py                   FusionNet architecture and joint loss (V1: apply_modality_mask)
│   ├── states.py                   State discovery utilities
│   ├── evaluate.py                 Evaluation helpers
│   └── data.py                     Data loading utilities
│
├── scripts/
│   ├── 01–05_*.py                  Data preparation, feature extraction, embeddings
│   ├── 06_train_baselines_v1.py    Fold-local matched baselines including quality-only RF
│   ├── 07_train_joint_v1.py        V1 training: fold-local screening, inner split, phase-2 refit
│   ├── 08_states_v1.py             GMM state discovery without forced k ≥ 2
│   ├── 09_*.py                     Original pipeline figures (V0)
│   ├── 10_figures_v1.py            V1 corrected result figures
│   ├── 11_group_analysis_v1.py     Group effects, longitudinal, model contrasts
│   ├── 12_model_probes_v1.py       Speaker-level logistic probes, score structure
│   ├── 13_domain_synthesis_v1.py   Fluency index, rate decomposition, SHAP uncertainty
│   ├── 14_robustness_v1.py         Relative rate robustness, timing identity
│   ├── 15_validate_v1.py           71-check validation suite
│   └── 16_downstream_v1.py         Temporal concordance, SHAP, orig-vs-V1 comparison
│
├── tests/
│   └── test_v1_corrections.py      13 regression tests (all pass)
│
├── data/
│   ├── processed/                  Manifest, features, embeddings index, ordering
│   └── results/                    All result tables (*_v1.csv = corrected; others = original V0)
│
└── figures/                        Figures (*_v1.png = corrected; others = original V0)
```

---

## Pipeline

### Stage 1 — Data preparation
Raw audio and metadata from DementiaNet are assembled into a speaker-grouped manifest with stratified splits. Recordings are resampled to 16 kHz mono and transcribed using faster-whisper with word-level timestamps. ASR provenance is unresolved: SHA-256 hashes of all 275 transcripts are recorded in `data/results/transcript_provenance_v1.csv`; the transcripts are treated as a fixed external asset.

### Stage 2 — Feature extraction
Forty-one handcrafted acoustic and linguistic features are extracted without librosa (pure NumPy/SciPy spectral processing). Frozen WavLM-base-plus and BERT-base-uncased embeddings (768-dim each) are extracted for all recordings.

### Stage 3 — Model training and evaluation (V1)
Feature screening by quality-covariate correlation and VIF is performed within each outer fold using outer-training speakers only, retaining 33–37 features per fold. FusionNet is trained with a speaker-disjoint inner validation split for early stopping. After epoch selection, preprocessing and a fresh model are refit on all outer-training speakers and evaluated once on outer-test speakers. Modality variants receive their intended feature streams independently: V_feat uses handcrafted features only; V_audio uses WavLM plus handcrafted features; V_text uses BERT plus handcrafted features; V_joint and V_cls use all three streams. Same-bucket temporal pairs are excluded from ranking supervision.

### Stage 4 — Post-hoc analysis
Scripts 11–14 perform bootstrap group contrasts, within-person longitudinal endpoint comparisons, rate decomposition, fixed-specification logistic probes, full-pipeline permutation tests, and attribution uncertainty. Training score distributions for percentile normalisation are taken from `states_assignments_v1.csv` (split = train), which contains actual outer-training scores on the same scale as test scores.

---

## Model

**FusionNet** (`src/mcd/models.py`)

Three input streams — handcrafted features (ℝ³⁷), WavLM embeddings (ℝ⁷⁶⁸), BERT embeddings (ℝ⁷⁶⁸) — are independently encoded to 64-dimensional representations and concatenated. A shared MLP produces two outputs: a group logit and a progression score. Stream inclusion is controlled by `apply_modality_mask()`, which zeros each stream independently based on the variant configuration.

**Joint loss:**  BCE(group_logit, y) + λ · mean softplus(−(s_later − s_earlier))

The ranking term applies a soft preference for increasing progression scores over defensibly ordered within-person pairs (λ = 0.5). Pairs sharing the same temporal rank bucket are excluded. Pairs are drawn exclusively from `data/processed/recording_order.csv`; no ordering is inferred from filename structure.

---

## Ethical Considerations

The progression score is a research measure and has not been validated against any clinical instrument. It must not be used to infer individual diagnosis or severity. No per-person decline verdicts are published for named individuals. Recording quality features are treated as covariates throughout. Intermediate latent states are labelled State-0 and State-1 only and carry no clinical interpretation. All analyses are post-hoc and hypothesis-generating; none constitutes preregistered confirmatory evidence.

---

## Setup

```bash
pip install -e .
pip install torch transformers faster-whisper soundfile praat-parselmouth shap scipy
```

Place DementiaNet files in `data/raw/` (not committed). Run scripts in order: 01 → 05 for data preparation; 06–08 for V1 training and states; 10–16 for figures and analysis. Scripts 05 and 07 require network access on first run for model downloads (HuggingFace Hub). A CUDA-capable GPU is recommended for script 07; all other scripts run on CPU.

To reproduce V1 results from scratch:
```bash
python scripts/06_train_baselines_v1.py
python scripts/07_train_joint_v1.py
python scripts/08_states_v1.py
python scripts/11_group_analysis_v1.py
python scripts/12_model_probes_v1.py
python scripts/13_domain_synthesis_v1.py
python scripts/14_robustness_v1.py
python scripts/15_validate_v1.py   # 71 checks, all pass
python scripts/16_downstream_v1.py
python scripts/10_figures_v1.py
```

---

## References

1. De Looze et al. (2021). Cognitive and Structural Correlates of Conversational Speech Timing in MCI and Alzheimer's Disease. *Front. Aging Neurosci.* doi:10.3389/fnagi.2021.637404
2. Mueller et al. (2018). Declines in Connected Language Are Associated with Very Early MCI. *Front. Aging Neurosci.* doi:10.3389/fnagi.2017.00437
3. Fraser, Meltzer & Rudzicz (2016). Linguistic Features Identify Alzheimer's Disease in Narrative Speech. *J. Alzheimer's Dis.* doi:10.3233/JAD-150520
4. Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE J-STSP.*
5. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL.*
6. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS.*
7. Gao et al. (2025). ADCeleb: A Longitudinal Speech Dataset from Public Figures for Early Detection of Alzheimer's Disease. *Interspeech.* doi:10.21437/Interspeech.2025-523
