# Explainable Multimodal Learning of Cognitive Decline Trajectories from Longitudinal Spontaneous Speech
## Consolidated Research Report — V0

**Author:** Tanishi Jain  
**Source commit:** `a391a57e5fe127dd44e211ffe6bd19e0d5777ac0`  
**Dataset:** DementiaNet (MIT licence) — 275 recordings, 141 speakers  
**Repo:** https://github.com/TanishiJrvce/Multimodal-Cognitive-Decline-Detection/tree/V0

> This report consolidates three layers of work: (1) the original V0 pipeline and results, (2) the manuscript-ready results and discussion, and (3) a reproducible secondary analysis package (`analysis/`). All numerical findings are sourced from committed output files. Ethical constraints and claim boundaries are stated throughout.

---

## Abstract

We present a multimodal framework for learning dementia-associated speech trajectories from longitudinal spontaneous speech, trained without binary classifier framing and without approval-gated clinical datasets. Using DementiaNet (275 recordings, 141 speakers), we train a compact three-stream FusionNet jointly optimising group classification and weak pairwise temporal ranking. The model combines WavLM acoustic embeddings, BERT linguistic embeddings, and 37 screened handcrafted features. On speaker-grouped held-out folds the joint model achieves AUC 0.665 (recording-level) and 0.688 (speaker-level), exceeding both single-modality baselines and a recording-quality shortcut (AUC 0.625). The progression scores recover defensible temporal ordering in 87.8% of within-person training pairs and 55.3% of held-out pairs. A secondary analysis package (8 figures, 31 tables) provides speaker-level group contrasts, within-person endpoint effects, temporal generalisation, and attribution analyses. No feature survives 37-feature BH correction. No state is labelled MCI.

---

## 1. Research Question

Can acoustic and linguistic representations, trained with **weak temporal supervision** and within-person comparisons, capture dementia-associated speech trajectories and reveal reproducible intermediate states — without binary classifier framing or approval-gated clinical datasets?

---

## 2. Data

**DementiaNet** — public interviews, MIT licence. No MMSE, no MCI labels, no clinician ratings.

| Split | Recordings | Speakers |
|---|---|---|
| Train | 227 | 116 |
| Validation | 48 | 25 |
| **Total** | **275** | **141** |

- **Speaker-grouped splits** throughout — no speaker appears in both sets.
- **Ordered pairs:** 82 recordings across 36 speakers have defensible temporal ranks → 40 within-person ordered pairs for ranking supervision.
- **Quality shortcut:** quality-only RandomForest AUC = 0.625 (dementia-group interviews are systematically noisier). All models must exceed this.

---

## 3. Pipeline Summary

```
Stage 1 — Data preparation
  01_build_manifest.py    → manifest.csv, speaker-grouped splits
  02_preprocess.py        → 16 kHz mono WAV, VAD
  03_transcribe.py        → word-level transcripts (faster-whisper)

Stage 2 — Feature extraction
  04_extract_features.py  → 41 handcrafted acoustic + linguistic features (pure numpy/scipy)
  05_embeddings.py        → WavLM-base-plus (768-dim) + BERT-base-uncased (768-dim)
  05b_feature_screen.py   → quality-correlation + VIF screening → 37 retained features
  06_train_baselines.py   → SVM / RF / LGBM baselines

Stage 3 — Joint model + evaluation
  07_train_joint.py       → FusionNet: 5 experiments × 5 seeds × 5 folds
  08_states.py            → GMM latent state discovery (k=1–6, BIC + silhouette)
  09_evaluate.py          → Speaker-level AUC, SHAP, trajectory plots

Secondary analysis (analysis/)
  analysis.py             → Group contrasts, longitudinal endpoint effects, model contrasts
  model_probes.py         → Logistic probes, pipeline permutation, OOF score structure
  domain_synthesis.py     → Rate decomposition, fluency index
  robustness.py           → Sensitivity checks
  figures.py              → 8 figure pairs
  build_report.py         → PDF report generator
  validate.py             → SHA-256 integrity checks
```

---

## 4. Features

### 4.1 Handcrafted (37 screened from 41 raw)

**Acoustic (28):** F0 mean/std/range/% voiced/slope, jitter, shimmer, HNR, RMS mean/std, speech rate, pause mean/rate/ratio/silence fraction, 13 MFCCs × mean+std, spectral centroid, flatness.

**Linguistic (9):** Type-token ratio, filler rate, word rate, sentence length, Zipf word frequency, lexical density, pronoun rate, repetition rate, word count.

Screened by: quality-covariate correlation (|r| > 0.3 removed) + VIF collinearity (VIF > 10 removed). Scaler fit on training fold only.

### 4.2 Neural Embeddings (frozen)

| Embedding | Dim | How |
|---|---|---|
| WavLM-base-plus | 768 | Mean-pooled frame representations |
| BERT-base-uncased | 768 | [CLS] token from ASR transcript |

---

## 5. Model — FusionNet

```
feat ∈ ℝ³⁷,  wavlm ∈ ℝ⁷⁶⁸,  bert ∈ ℝ⁷⁶⁸

Per-modality encoder (each):
  LayerNorm → Dropout(0.3) → Linear(d→64) → GELU

Fusion MLP:
  concat(z_feat, z_wavlm, z_bert) ∈ ℝ¹⁹²
  → Linear(192→64) → GELU → Dropout(0.3)

Heads:
  group_logit  (1-dim) → dementia-group probability
  prog_score   (1-dim) → relative speech-change score (unitless, unbounded)
```

**Joint loss:**

```
L = BCE(logit, y) + λ · mean softplus(−(s_later − s_earlier))
```

λ = 0.5. Ranking pairs from `recording_order.csv` only — never inferred from filenames alone.

**Training:** 5-fold StratifiedGroupKFold (speaker-grouped) × 5 seeds. Adam lr=1e-4, weight decay=1e-4. Early stopping patience=10, max 100 epochs. Modality dropout p=0.2.

---

## 6. Results

### 6.1 Original Model Performance

| Model | AUC mean ± std | Balanced Acc |
|---|---|---|
| Quality-only baseline | 0.625 | — |
| SVM (handcrafted) | 0.600 ± 0.080 | — |
| **V_feat** (features only) | 0.500 ± 0.000 | 0.500 |
| **V_audio** (WavLM + features) | 0.636 ± 0.098 | 0.607 |
| **V_text** (BERT + features) | 0.609 ± 0.088 | 0.576 |
| **V_cls** (all modalities, cls-only) | 0.656 ± 0.097 | 0.615 |
| **V_joint** (all modalities + ranking) | **0.665 ± 0.088** | **0.622** |

**Speaker-level AUC** (mean predictions per speaker): **0.688 ± 0.109**

**Joint vs. cls-only:** V_joint exceeded V_cls by mean 0.0092 AUC. Positive differences in 15/25 matched runs; seed-level mean differences ranged −0.017 to +0.039 — heterogeneous, not decisive.

### 6.2 Speaker-Level Group Contrasts *(secondary analysis)*

Speaker averages compared across 80 dementia-associated and 61 comparison speakers (5,000 bootstrap resamples + 10,000 label permutations per feature):

| Feature | Dementia mean | Comparison mean | Difference | Hedges g | Perm. p | BH q (37 tests) |
|---|---|---|---|---|---|---|
| pause_mean_s | +88 ms longer | — | +0.088 s | ≈ +0.29 | nominal ≈ 0.03 | > 0.05 |
| pause_ratio | +1.72 pp higher | — | +0.0172 | ≈ +0.29 | nominal ≈ 0.03 | > 0.05 |
| mattr50 (lexical diversity) | ≈ equal | — | −0.0014 | ≈ 0 | — | — |

> **No feature survives 37-feature BH FDR correction.** Available-covariate adjustment (sex + 6 recording covariates, HC3 sandwich) retains the direction of pause differences with nominal q near 0.03 — not adjusted-significant. Conclusions are descriptive only.

### 6.3 Within-Person Endpoint Changes *(secondary analysis)*

31 speakers with ≥ 2 distinct ordered recording buckets; earliest vs. latest bucket compared:

- **Mean log word-rate change:** −0.0946 (geometric average decrease **9.03%**; pointwise speaker-bootstrap CI: 1.01%–17.35% decrease)
- **Sign-flip p = 0.0474** (exploratory); four-contrast BH q = 0.0721 — below adjusted significance
- **Decomposition:** ≈ 90.2% of mean log change assigned to pause-excluded word delivery; ≈ 9.8% to non-pause share change

> This decomposition is exploratory and uncertainty is scale-sensitive. The pause-excluded measure is word-based and cannot distinguish cognitive formulation demands from motor, respiratory, or transcription contributions.

### 6.4 Temporal Generalisation *(secondary analysis)*

Bucket-averaged progression scores, equal speaker weighting:

| Split | Temporal concordance | Bootstrap CI |
|---|---|---|
| Training speakers | **87.8%** | — |
| Held-out speakers | **55.3%** | 43.9% – 65.6% |
| Five-seed OOF ensemble (held-out) | 60.8% | — |

**Within-person ordering-permutation p = 0.124** (OOF consensus). Group discrimination by held-out progression percentile: **AUC ≈ 0.505** (speaker-level, CI: 0.410–0.599) — approximately chance after held-out normalisation.

> Training concordance (87.8%) substantially exceeds held-out (55.3%). The progression head does not generalise temporal ordering clearly to unseen speakers. Shared representation learning does not establish shared clinical meaning for both heads.

### 6.5 Latent Speech States *(original + secondary)*

BIC selected k = 1 in **24 of 25** runs (seeds × folds). Operational two-state assignments varied across held-out models for **219 of 275** recordings.

Seed-0 / fold-0 representative two-state summary:

| State | N recordings | Mean prog. score | % Dementia group |
|---|---|---|---|
| State-0 | 176 | −0.333 ± 0.175 | 40% |
| State-1 | 99 | +0.142 ± 0.161 | 57% |

> States provide a compact operational description of lower vs. higher score regions. The study **does not establish a universal sequence of discrete clinical stages**. No state is labelled MCI. Continuous variation is better supported than discrete staging.

### 6.6 Attribution Analysis *(secondary analysis)*

**RF SHAP (5,000 speaker-bootstrap resamples):**

- Timing features (speech_rate, pause_mean_s, pause_ratio) have **≈ 1.9× the mean absolute SHAP per feature** of spectral descriptors
- pause_mean_s ranked top-5 in **96.6%** of bootstrap samples
- pause_ratio ranked top-5 in **85.7%** of bootstrap samples

> SHAP per-feature-family total: spectral family accounts for more summed attribution (more features), but timing features have higher **attribution density** per feature. Attribution concerns the RF baseline, not the neural progression head.

### 6.7 Logistic Probe Families *(secondary analysis)*

Fixed-spec logistic probes (C=0.1, balanced, speaker-averaged features, 5-fold CV × 5 seeds):

| Probe family | Features | AUC |
|---|---|---|
| Timing3 | speech_rate, pause_mean_s, pause_ratio | see `analysis/tables/supplementary_probe_summary.csv` |
| Language5 | mattr50, filler_rate, rep_rate, bigram_rep, sent_len | — |
| All37 | All screened features | — |
| Quality6 | Recording quality covariates | — |

**Full-pipeline permutation (499 permutations, seed-0):**
- Timing3 and All37 permutation p-values in `analysis/tables/permutation_Timing3.csv` and `permutation_All37.csv`

### 6.8 Subgroup Analysis

Computed on RF baseline (fold-0, seed-0, n=55 validation recordings):

| Subgroup | N | AUC |
|---|---|---|
| Male | 50 | 0.623 |
| Female | 5 | 0.500 (too few) |
| Quality tertile: low | 19 | **0.690** |
| Quality tertile: mid | 21 | 0.564 |
| Quality tertile: high | 15 | 0.589 |

Higher AUC in the low-quality tertile confirms the recording-quality confound remains partially active.

---

## 7. Discussion

### 7.1 Classification vs. Temporal Ordering

The group classifier provides partial discrimination (AUC 0.665–0.688) across unseen speakers. The progression head generalises temporal ordering far less clearly — training concordance 87.8%, held-out 55.3%, OOF permutation p = 0.124. These are not interchangeable measures of impairment. Joint training improves classification marginally but does not establish a validated longitudinal tracking capability.

### 7.2 Speech Timing as the Most Interpretable Signal

Pausing and rate differences between groups (Hedges g ≈ 0.29) are descriptively consistent with the clinical literature (De Looze et al. 2021; Mueller et al. 2018), though no feature individually survives FDR correction. The within-person decomposition shows that most of the measured slowing is in pause-excluded word delivery, not accumulating silence — distinguishing within-speaker delivery dynamics from between-group pausing patterns. These distinctions matter for mechanistic interpretation.

### 7.3 Lexical Diversity Does Not Track Pausing

Near-zero MATTR-50 group difference (−0.0014) coexists with timing-related patterns, consistent with speech-production efficiency and vocabulary breadth being partially distinct. Comparable MATTR values are not equivalent to intact semantic content.

### 7.4 Attribution Requires the Right Denominator

Spectral features dominate summed RF SHAP by count, but timing features have ≈1.9× higher attribution density per feature. This normalisation distinction matters for ranking feature families fairly. Attribution concerns the RF baseline; correlated inputs prevent interpreting SHAP as a causal partition.

### 7.5 Progression Score Encodes Sampling Structure

Held-out progression percentiles associate with recording duration and spectral variability. The negative association with bigram repetition attenuates after word-count adjustment. The score organises a mixture of speech and sampling properties — characterising these gives it a more precise empirical meaning without assigning an exclusively cognitive cause.

### 7.6 Discrete States Are Not Established

BIC favours k = 1 in 24/25 runs; most recordings change state assignment across held-out fits. The two-state operational description is useful for compact summarisation but does not establish a universal clinical staging sequence. Continuous variation in the progression score is the better-supported finding.

### 7.7 Scope of Inference

The cohort comprises public interview recordings with heterogeneous dementia histories, sparse approximate timing, and unequal recording counts per person. Clinical severity, biomarkers, education, and exact recording age are unobserved. Analyses cannot causally separate healthy ageing from disease or infer individual diagnosis.

---

## 8. Limitations

1. **Small N:** 141 speakers, 275 recordings; cross-validation variance is high (AUC std ≈ 0.09).
2. **No clinical metadata:** No MMSE, CDR, or MCI labels. Subgroup analysis limited to gender and quality tertile.
3. **Sparse temporal ordering:** Only 82/275 recordings have defensible ranks; ranking supervision is sparse.
4. **Quality confound:** Recording-quality shortcut is real and partially active in all models (visible in low-quality tertile AUC).
5. **Frozen embeddings:** WavLM and BERT not fine-tuned; task-specific fine-tuning on larger datasets could improve substantially.
6. **Training vs. held-out gap in temporal concordance:** 87.8% → 55.3% — the progression head does not generalise clearly.
7. **No feature survives FDR:** All group-contrast findings are descriptive and hypothesis-generating only.
8. **Dataset demographics:** Predominantly Caucasian/White, English-speaking.

---

## 9. Ethical Statement

- No per-person "decline verdicts" published for named individuals.
- Progression score is a **research measure only** — not a clinical severity scale.
- Recording quality features are covariates, not predictors.
- Temporal ordering is a soft hypothesis, not enforced; only defensible orderings used.
- No intermediate cluster is labelled "MCI" — labels are State-0 … State-k.
- All analyses are post-hoc and hypothesis-generating; none is preregistered confirmatory evidence.
- Neither a high model score nor a discovered state is a confirmed MCI or dementia diagnosis.

---

## 10. References

1. De Looze et al. (2021). Cognitive and Structural Correlates of Conversational Speech Timing in MCI and Mild-to-Moderate Alzheimer's Disease. *Front. Aging Neurosci.* DOI: 10.3389/fnagi.2021.637404
2. Mueller et al. (2018). Declines in Connected Language Are Associated with Very Early MCI: Results from WRAP. *Front. Aging Neurosci.* DOI: 10.3389/fnagi.2017.00437
3. Themistocleous, Eckerstrom & Kokkinakis (2020). Voice quality and speech fluency distinguish individuals with MCI from Healthy Controls. *PLOS ONE.* DOI: 10.1371/journal.pone.0236009
4. Ahmed et al. (2013). Connected speech as a marker of disease progression in autopsy-proven Alzheimer's disease. *Brain.* DOI: 10.1093/brain/awt269
5. Heitz, Schneider & Langer (2024). The Influence of ASR on Linguistic Features and Automatic Alzheimer's Disease Detection from Spontaneous Speech. *LREC-COLING.*
6. Fraser, Meltzer & Rudzicz (2016). Linguistic Features Identify Alzheimer's Disease in Narrative Speech. *J. Alzheimer's Dis.* DOI: 10.3233/JAD-150520
7. Nadeau & Bengio (1999). Inference for the Generalization Error. *NeurIPS 12.*
8. Petti & Korhonen (2024). LoSST-AD: A Longitudinal Corpus for Tracking Alzheimer's Disease Related Changes in Spontaneous Speech. *LREC-COLING.*
9. Gao et al. (2025). ADCeleb: A Longitudinal Speech Dataset from Public Figures for Early Detection of Alzheimer's Disease. *Interspeech.* DOI: 10.21437/Interspeech.2025-523
10. Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. *IEEE JSTSP.*
11. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL.*
12. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS.*

---

## 11. Reproducibility

**Pipeline (Stage 1–3):** Run scripts 01 → 09 as described in `README.md`. Requires GPU (RTX 4050 or better) for script 07. Total runtime ≈ 60–90 minutes.

**Secondary analysis:** Run scripts in `analysis/` — no GPU required, runs in minutes. See `analysis/README.md`.

**Integrity:** `analysis/tables/source_fingerprints.csv` records SHA-256 of every committed input CSV. `analysis/tables/analysis_validation.csv` records validation checks.

**Hardware:** NVIDIA RTX 4050 Laptop (6 GB VRAM), Python 3.12, PyTorch 2.6.0+cu124, transformers 5.23.0.
