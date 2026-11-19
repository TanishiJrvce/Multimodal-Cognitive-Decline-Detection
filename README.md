# Explainable Multimodal Learning of Cognitive Decline Trajectories from Longitudinal Spontaneous Speech

**Tanishi Jain** · V0 · DementiaNet (MIT licence)

---

## Overview

This repository presents a multimodal framework for learning dementia-associated speech trajectories from longitudinal spontaneous speech. Rather than framing the task as binary classification, the framework jointly optimises a group-discrimination objective and a weak pairwise temporal ranking loss over within-person recording pairs, producing both a group probability and an uncalibrated progression score per recording. All analyses use the publicly available DementiaNet corpus and are conducted without access to clinical severity ratings, MMSE scores, or approval-gated data.

The central question is whether acoustic and linguistic representations, trained with minimal supervision, can capture speech changes associated with dementia and reveal reproducible intermediate patterns of speech organisation — without enforcing hard ordinal labels or clinical staging assumptions.

---

## Results

Evaluation follows strict speaker-grouped cross-validation (5-fold × 5 seeds). No speaker appears in both training and validation partitions. The recording-quality shortcut baseline (AUC 0.625, based on loudness and SNR alone) defines the minimum bar for a model to be considered informative.

### Model performance

| Model | AUC mean ± SD | Balanced accuracy |
|---|---|---|
| Quality-only baseline | 0.625 | — |
| SVM, handcrafted features | 0.600 ± 0.080 | — |
| FusionNet — features only | 0.500 ± 0.000 | 0.500 |
| FusionNet — WavLM + features | 0.636 ± 0.098 | 0.607 |
| FusionNet — BERT + features | 0.609 ± 0.088 | 0.576 |
| FusionNet — all modalities, classification only | 0.656 ± 0.097 | 0.615 |
| **FusionNet — all modalities + ranking loss** | **0.665 ± 0.088** | **0.622** |

Averaging predictions across a speaker's recordings yields a speaker-level AUC of **0.688 ± 0.109**.

### Temporal consistency

Progression scores correctly ordered **34 of 40** defensible within-person recording pairs (pair accuracy 0.850) in the original evaluation. Extended analysis over all 25 saved model checkpoints yielded training-split concordance of **87.8%** and held-out concordance of **55.3%** (bootstrap CI: 43.9%–65.6%), with an ordering-permutation p-value of 0.124 on the held-out ensemble. These results establish that the progression coordinate generalises temporal ordering only partially beyond the training speakers and should not be interpreted as a validated longitudinal tracking instrument.

### Speech-feature group differences

Speaker-averaged comparisons across 80 dementia-associated and 61 comparison speakers (5,000 bootstrap resamples, 10,000 label permutations) identify pause duration (+88 ms, Hedges g ≈ 0.29) and pause proportion (+1.7 pp, Hedges g ≈ 0.29) as the most consistently directional differences. No feature survives Benjamini–Hochberg correction at the 37-feature family level after permutation testing. Covariate-adjusted models (sex, recording duration, SNR, bandwidth, loudness, peak, channel correlation) retain the direction of pause effects at nominal p ≈ 0.03 without achieving adjusted significance. All group-contrast findings are descriptive and hypothesis-generating.

### Within-person longitudinal change

Among 31 speakers with two or more distinct ordered recording epochs, the geometric mean word rate declined by **9.0%** between earliest and latest available recordings (bootstrap CI: 1.0%–17.4%; sign-flip p = 0.047, BH-adjusted q = 0.072 across four contrasts). An exact log decomposition attributes approximately 90% of the mean log rate change to the pause-excluded delivery component and 10% to the non-pause share, distinguishing within-speaker delivery dynamics from accumulating silence. This decomposition is exploratory; the pause-excluded measure is derived from recognised word timestamps and cannot isolate cognitive, motor, or transcription-related contributions.

### Latent speech states

Gaussian mixture modelling of progression scores on training folds selected k = 1 by BIC in 24 of 25 seed–fold runs. An operational two-state summary is used for description: State-0 (n = 176, 40% dementia-associated) and State-1 (n = 99, 57% dementia-associated). State assignments varied across held-out model fits for 219 of 275 recordings. The study does not establish a universal sequence of discrete clinical stages, and no state is labelled MCI or any clinical category.

### Feature attribution

Random Forest SHAP analysis (5,000 speaker-resampled bootstraps) identifies timing features (speech rate, pause duration, pause proportion) as having approximately **1.9× higher attribution density** per feature than spectral descriptors. Pause duration and pause proportion ranked in the top five features in 96.6% and 85.7% of bootstrap resamples, respectively. Attribution is computed for the RF baseline and does not constitute a causal partition of disease mechanisms.

---

## Repository Structure

```
├── scripts/
│   ├── 01_build_manifest.py        Stage 1: manifest construction, speaker-grouped splits
│   ├── 02_preprocess.py            Stage 1: audio preprocessing (16 kHz mono, VAD)
│   ├── 03_transcribe.py            Stage 1: ASR transcription with word timestamps
│   ├── 04_extract_features.py      Stage 2: handcrafted acoustic and linguistic features
│   ├── 05_embeddings.py            Stage 2: WavLM and BERT embedding extraction
│   ├── 05b_feature_screen.py       Stage 2: quality-correlation and collinearity screening
│   ├── 06_train_baselines.py       Stage 2: SVM / RF / LGBM baselines
│   ├── 07_train_joint.py           Stage 3: FusionNet training (5 experiments × 5 seeds × 5 folds)
│   ├── 08_states.py                Stage 3: GMM latent state discovery
│   ├── 09_evaluate.py              Stage 3: speaker-level evaluation, SHAP, trajectory plots
│   ├── 10_pipeline_figures.py      Supplementary figures for pipeline results
│   ├── 11_group_analysis.py        Post-hoc speaker-group contrasts, longitudinal endpoint analysis
│   ├── 12_model_probes.py          Fixed-spec logistic probes, pipeline permutation, OOF score structure
│   ├── 13_domain_synthesis.py      Rate decomposition, fluency index, SHAP speaker bootstrap
│   ├── 14_robustness.py            Sensitivity analyses (subtype, repetition, relative rate)
│   └── 15_validate.py              Numerical and structural integrity validation
│
├── src/mcd/
│   ├── models.py                   FusionNet architecture and joint loss
│   ├── states.py                   State discovery utilities
│   ├── evaluate.py                 Evaluation helpers
│   └── data.py                     Data loading utilities
│
├── data/
│   ├── processed/                  Committed CSVs (manifest, features, embeddings index, ordering)
│   └── results/                    All result tables (43 files)
│
└── figures/                        All figures (28 files: pipeline + extended analysis)
```

---

## Pipeline

### Stage 1 — Data preparation
Raw audio and metadata from DementiaNet are assembled into a speaker-grouped manifest with stratified splits. Recordings are resampled to 16 kHz mono and transcribed using faster-whisper with word-level timestamps.

### Stage 2 — Feature extraction
Forty-one handcrafted acoustic and linguistic features are extracted without librosa (pure NumPy/SciPy spectral processing). Features are screened by quality-covariate correlation (|r| > 0.3 excluded) and variance inflation factor (VIF > 10 excluded), retaining 37. Frozen WavLM-base-plus and BERT-base-uncased embeddings (768-dim each) are extracted for all recordings.

### Stage 3 — Model training and evaluation
FusionNet combines three input streams through per-modality encoders (LayerNorm → Dropout → Linear → GELU) and a shared fusion MLP, producing a group logit and a progression score. Training uses speaker-grouped cross-validation with Adam, early stopping, and modality dropout. SHAP values, speaker-level aggregated metrics, and within-person trajectory plots are computed from the best checkpoints.

### Extended analysis
Post-hoc scripts (11–15) perform bootstrap group contrasts, within-person longitudinal endpoint comparisons, rate decomposition, fixed-specification logistic probes, full-pipeline permutation tests, and integrity validation. These analyses extend the original pipeline findings without modifying any committed inputs.

---

## Model

**FusionNet** (`src/mcd/models.py`)

Three input streams — handcrafted features (ℝ³⁷), WavLM embeddings (ℝ⁷⁶⁸), BERT embeddings (ℝ⁷⁶⁸) — are independently encoded to 64-dimensional representations and concatenated. A shared MLP produces two outputs: a group logit and a progression score.

**Joint loss:**  BCE(group\_logit, y) + λ · mean softplus(−(s\_later − s\_earlier))

The ranking term applies a soft preference for increasing progression scores over defensibly ordered within-person pairs (λ = 0.5). Pairs are drawn exclusively from `data/processed/recording_order.csv`; no ordering is inferred from filename structure alone.

---

## Ethical Considerations

The progression score is a research measure and has not been validated against any clinical instrument. It must not be used to infer individual diagnosis or severity. No per-person decline verdicts are published for named individuals. Recording quality features are treated as covariates throughout. Intermediate latent states are labelled State-0 through State-k and carry no clinical interpretation. All analyses are post-hoc and hypothesis-generating; none constitutes preregistered confirmatory evidence.

---

## Setup

```bash
pip install -e .
pip install torch transformers faster-whisper soundfile praat-parselmouth shap scipy
```

Place DementiaNet files in `data/raw/` (not committed). Run scripts in order: 01 → 09 for the main pipeline; 10 → 15 for extended analysis. Scripts 05, 07, and 12 require network access on first run for model downloads (HuggingFace Hub). A CUDA-capable GPU is recommended for script 07; all other scripts run on CPU.

---

## References

1. De Looze et al. (2021). Cognitive and Structural Correlates of Conversational Speech Timing in MCI and Alzheimer's Disease. *Front. Aging Neurosci.* doi:10.3389/fnagi.2021.637404
2. Mueller et al. (2018). Declines in Connected Language Are Associated with Very Early MCI. *Front. Aging Neurosci.* doi:10.3389/fnagi.2017.00437
3. Fraser, Meltzer & Rudzicz (2016). Linguistic Features Identify Alzheimer's Disease in Narrative Speech. *J. Alzheimer's Dis.* doi:10.3233/JAD-150520
4. Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE J-STSP.*
5. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL.*
6. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS.*
7. Gao et al. (2025). ADCeleb: A Longitudinal Speech Dataset from Public Figures for Early Detection of Alzheimer's Disease. *Interspeech.* doi:10.21437/Interspeech.2025-523
