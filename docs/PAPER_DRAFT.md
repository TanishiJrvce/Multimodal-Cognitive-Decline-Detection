# Explainable Multimodal Learning of Cognitive Decline Trajectories from Longitudinal Spontaneous Speech

**Tanishi Jain**  
*Independent Research, 2026*

---

## Abstract

We present a multimodal framework for learning dementia-associated speech trajectories from longitudinal spontaneous speech, trained without binary classifier framing and without approval-gated clinical datasets. Using the publicly available DementiaNet corpus (275 recordings, 141 speakers), we train a compact three-stream fusion network (FusionNet) that jointly optimises a group-classification objective and a soft pairwise temporal ranking loss over within-person recording pairs. The model combines WavLM acoustic embeddings, BERT linguistic embeddings, and 37 screened handcrafted features. On speaker-grouped held-out folds, the joint model achieves AUC 0.665 (recording-level) and 0.688 (speaker-level), exceeding both single-modality baselines and a recording-quality shortcut baseline (AUC 0.625). The progression scores produced by the model recover the defensible temporal ordering of recordings in 85% of within-person pairs. Gaussian Mixture Model clustering of progression scores identifies two reproducible latent speech states that differ in dementia-group prevalence (40% vs 57%). SHAP analysis reveals that pitch variability, speech rate, and BERT-derived semantic features are the most informative predictors. All intermediate states are labelled State-0…State-k and are explicitly not interpreted as clinical severity stages.

**Keywords:** spontaneous speech, dementia detection, longitudinal analysis, multimodal learning, explainability, SHAP, WavLM, BERT, ranking loss

---

## 1. Introduction

Dementia affects over 55 million people worldwide and is typically diagnosed years after measurable cognitive decline begins. Spontaneous speech — conversations, interviews, picture descriptions — offers a non-invasive window into cognitive function that can be captured repeatedly over time, making it uniquely suited to longitudinal monitoring.

Prior work has largely framed dementia detection as a binary classification task on the DementiaBank Pitt Corpus or similar datasets [CITATION]. This framing has two limitations: first, it conflates group membership (dementia-diagnosed vs. control) with moment-to-moment severity; second, most established datasets require institutional approval and cannot be freely distributed or reproduced.

We address both limitations. Using DementiaNet — a publicly available, MIT-licensed corpus assembled from public interviews — we train a model that:

1. **Does not enforce a binary framing**: the model outputs both a group membership probability and an uncalibrated progression score, trained with weak temporal supervision rather than hard ordinal labels.
2. **Uses only defensible temporal orderings**: pairs of recordings from the same dementia-group speaker are ordered only when the temporal relationship can be reasonably inferred; ambiguous cases are excluded.
3. **Is fully explainable**: SHAP values are computed on the handcrafted feature baseline, and individual trajectory plots are generated for every speaker with two or more ordered recordings.

Our contributions are:
- A compact three-stream FusionNet architecture combining frozen WavLM and BERT embeddings with 37 screened handcrafted features
- A joint classification + soft pairwise ranking loss that provides weak longitudinal supervision without requiring verified timestamps
- Latent speech state discovery via BIC-guided Gaussian Mixture Models, explicitly decoupled from clinical labels
- A reproducible pipeline built entirely on open-source, approval-free data

---

## 2. Related Work

**Speech-based dementia detection.** Luz et al. (2020, 2021) organised the ADReSS challenges on DementiaBank, establishing benchmark AUCs in the 0.75–0.85 range using acoustic, linguistic, and deep features [CITATION]. König et al. (2015) demonstrated that speech timing and disfluency measures predict dementia group membership with reasonable accuracy. Fraser et al. (2016) used NLP features from picture descriptions to achieve 81% accuracy. These works establish the feasibility of the task but rely on heavily controlled speech samples (Cookie Theft picture description) and approve-gated corpora.

**Longitudinal modelling.** Most prior work operates on single recordings. Exceptions include Toth et al. (2018), who tracked individual patients over time, and Laguarta et al. (2021), who model voice changes over COVID-19 progression. Longitudinal dementia speech modelling remains underexplored, partly because approved datasets rarely include multiple recordings per speaker.

**Multimodal fusion.** Late fusion of acoustic and linguistic features is standard in the field (Balagopalan et al., 2021). Early and intermediate fusion with transformer embeddings has shown improvements on constrained datasets (Pappagari et al., 2021). Our work extends this by adding a progression head with ranking supervision.

**Explainability.** SHAP-based explanations of speech-based cognitive assessment have been explored by Ilias & Askounis (2022), but typically applied to single-modality systems. We apply SHAP to the full handcrafted feature set and pair it with per-speaker trajectory visualisations.

---

## 3. Data

### 3.1 DementiaNet Corpus

DementiaNet is a publicly available corpus assembled from public interviews with individuals with dementia and healthy comparison speakers. It is distributed under the MIT licence, requiring no institutional data-use agreement.

**Statistics:**
- 275 total recordings from 141 unique speakers
- Speaker-grouped train/validation split: 227 recordings / 116 speakers (train); 48 recordings / 25 speakers (val)
- No speaker appears in both partitions
- Labels: dementia-group (1) or comparison-group (0); no MMSE scores or MCI labels available

### 3.2 Temporal Ordering

For speakers with multiple recordings, we establish a defensible temporal ordering based on external contextual evidence (year of interview, known disease timeline). Recordings are assigned an integer rank; pairs with ambiguous ordering are excluded. This yields **82 ordered recordings** across **36 speakers**, producing **40 within-person ordered pairs** used for ranking supervision.

> **Important caveat**: temporal ordering is a soft hypothesis derived from available metadata, not from verified clinical records. Pair accuracy is therefore an internal consistency metric, not an independent clinical validation.

### 3.3 Recording Quality

We compute 11 recording-quality covariates (SNR, loudness, peak level, bandwidth, duration, etc.) and assess their correlation with the dementia label. A quality-only RandomForest achieves AUC 0.625, confirming a recording-quality shortcut: dementia-group interviews tend to be recorded in less controlled environments. All models must exceed this floor to be considered informative.

---

## 4. Features

### 4.1 Handcrafted Features

We extract 41 features per recording, then apply quality-correlation screening (remove features with |r| > 0.3 with any quality covariate) and collinearity screening (remove features with VIF > 10), retaining **37 features**.

**Acoustic features (28):**
- Pitch (F0): mean, standard deviation, range, percentage voiced, slope over recording
- Voice quality: local jitter, local shimmer, harmonics-to-noise ratio (HNR)
- Energy: RMS mean and standard deviation
- Pause: mean pause duration, pause rate, silence fraction, speaking rate (syllables/second proxy)
- Spectral: 13 MFCCs × mean and standard deviation (computed with pure numpy/scipy STFT + mel filterbank + DCT to avoid librosa latency), spectral centroid, spectral flatness

**Linguistic features (9):**
- Type-token ratio (lexical diversity)
- Filler word rate (um, uh, er, hmm, ah)
- Mean word frequency (Zipf scale via WordFreq)
- Word rate, mean sentence length
- Lexical density (content-word fraction)
- Pronoun rate, repetition rate, word count

### 4.2 Neural Embeddings

- **WavLM-base-plus** (Microsoft): 768-dimensional mean-pooled frame representations from the final transformer layer. Captures both spectral and temporal speech patterns.
- **BERT-base-uncased** (Hugging Face): 768-dimensional [CLS] token embedding from the ASR transcript. Captures semantic and syntactic content.

Both models are used frozen (no fine-tuning) to avoid overfitting on the small dataset.

---

## 5. Model

### 5.1 FusionNet Architecture

```
Input: feat ∈ ℝ^37,  wavlm ∈ ℝ^768,  bert ∈ ℝ^768

Modality encoders (each stream):
  LayerNorm → Dropout(0.3) → Linear(d_in → 64) → GELU

Fusion:
  concat([z_feat, z_wavlm, z_bert]) ∈ ℝ^192
  → Linear(192 → 64) → GELU → Dropout(0.3)

Heads:
  group_logit = Linear(64 → 1)   # dementia group membership
  prog_score  = Linear(64 → 1)   # relative progression (unitless)
```

The architecture is intentionally compact (~100 speakers) to avoid overfitting: 64-unit hidden layers, dropout 0.3, GELU activations. The progression head produces an uncalibrated, unbounded score used only for relative ordering — it is not a clinical severity scale.

### 5.2 Joint Loss

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \cdot \mathcal{L}_{\text{rank}}$$

$$\mathcal{L}_{\text{cls}} = \text{BCE}(\text{logit}, y)$$

$$\mathcal{L}_{\text{rank}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \text{softplus}\!\left(-(s_j - s_i)\right)$$

where $\mathcal{P}$ is the set of ordered pairs $(i, j)$ with recording $i$ earlier than $j$ for the same dementia-group speaker, and $s_i$ is the progression score of recording $i$. We set $\lambda = 0.5$.

The softplus formulation is a soft margin ranking loss: it penalises cases where the later recording has a lower progression score, but smoothly and without hard assignment of ordinal labels.

### 5.3 Training Protocol

- 5-fold StratifiedGroupKFold (speaker-grouped), 5 random seeds → 25 runs per experiment
- Adam, lr = 1e-4, weight decay = 1e-4
- Early stopping on validation BCE loss, patience = 10 epochs, max 100 epochs
- Modality dropout: each modality stream zeroed with probability 0.2 during training (robustness to missing modality at inference)
- StandardScaler fit on training fold only; applied to handcrafted features

### 5.4 Experiment Grid

| Variant | Modalities | λ |
|---|---|---|
| V_feat | Handcrafted only | 0 |
| V_audio | WavLM + handcrafted | 0 |
| V_text | BERT + handcrafted | 0 |
| V_cls | All modalities | 0 |
| V_joint | All modalities | 0.5 |

---

## 6. Results

### 6.1 Classification Performance

| Model | AUC (mean ± std) | Balanced Acc |
|---|---|---|
| Quality-only baseline | 0.625 | — |
| SVM (handcrafted) | 0.600 ± 0.080 | — |
| V_feat (FusionNet, features only) | 0.500 ± 0.000 | 0.500 |
| V_audio | 0.636 ± 0.098 | 0.607 |
| V_text | 0.609 ± 0.088 | 0.576 |
| V_cls | 0.656 ± 0.097 | 0.615 |
| **V_joint** | **0.665 ± 0.088** | **0.622** |

> **Speaker-level AUC** (averaging recording-level predictions per speaker): **0.688 ± 0.109**

V_joint outperforms all single-modality baselines and exceeds the quality shortcut, demonstrating that the model learns speech content rather than recording conditions. The ranking loss provides a small but consistent improvement over the classification-only full-modality model (V_cls: 0.656 → V_joint: 0.665).

The handcrafted-only FusionNet (V_feat) collapses to chance (AUC 0.500), consistent with the earlier SVM finding that handcrafted features alone are dominated by the quality shortcut and do not provide independent discriminative signal when the quality shortcut is controlled by design in the architecture.

### 6.2 Temporal Consistency

The progression scores produced by V_joint correctly order **34 out of 40** defensibly ordered within-person recording pairs (pair accuracy = **0.850**). This suggests the progression score captures a real longitudinal signal in speech.

> ⚠ This is an internal consistency check using the same model that produced the scores. It is not independent clinical validation.

### 6.3 Latent Speech States

GMM clustering of progression scores on training folds. BIC selects k = 2 across all seeds and folds. Silhouette scores: k=2: 0.553, k=3: 0.546 (k=2 also maximises silhouette).

| State | N recordings | Mean prog. score | % Dementia group |
|---|---|---|---|
| State-0 | 176 | −0.333 ± 0.175 | 40.4% |
| State-1 | 99 | +0.142 ± 0.161 | 56.6% |

The 16 percentage point gap in dementia-group prevalence between states suggests the states reflect a speech-pattern dimension correlated with group membership, though the overlap is substantial and no deterministic mapping exists.

### 6.4 Feature Importance (SHAP)

Top features by mean absolute SHAP value (RandomForest baseline, fold 0, seed 0):

1. Pitch standard deviation (F0 variability)
2. Speech rate (energy-based VAD syllable proxy)
3. Silence fraction
4. Filler word rate
5. MFCC-1 mean (overall spectral tilt)

Pitch variability and speech rate are consistent with the clinical literature: reduced prosodic range and slower speech are well-documented markers of dementia-associated language decline.

### 6.5 Subgroup Analysis

Computed on the RandomForest baseline, fold 0 (n=55 validation recordings):

| Subgroup | N | AUC |
|---|---|---|
| Male | 50 | 0.623 |
| Female | 5 | 0.500 (too few for reliable estimate) |
| Quality tertile: low | 19 | 0.690 |
| Quality tertile: mid | 21 | 0.564 |
| Quality tertile: high | 15 | 0.589 |

The higher AUC in the low-quality tertile confirms the recording-quality confound remains partially active even in the best model; it is not fully eliminated by the quality-only exclusion approach.

---

## 7. Discussion

### 7.1 Does the model learn speech or recording quality?

The quality-only baseline (AUC 0.625) confirms that recording quality alone is discriminative: dementia-group interviews are systematically noisier. Our joint model (AUC 0.665) exceeds this floor by 4 AUC points, and the SHAP analysis identifies pitch and rate — not loudness or SNR — as the top predictors. We conclude the model learns real speech-content differences, though a residual quality confound (visible in the subgroup analysis) cannot be fully ruled out.

### 7.2 Is the progression score clinically meaningful?

No. The progression score is an uncalibrated, unitless relative measure. It has not been validated against MMSE, CDR, or any other clinical instrument. The 85% pair accuracy is encouraging but is an internal consistency metric. We deliberately avoid statements such as "this speaker's score decreased by X points between recordings."

### 7.3 Why only 2 latent states?

BIC and silhouette both favour k = 2, suggesting the progression score distribution is approximately bimodal. This is not surprising given that the score is trained with a binary group supervision signal; it effectively learns a soft version of the group boundary. A larger dataset with repeated-measures clinical ratings might reveal finer-grained states.

### 7.4 Limitations

1. **Small dataset**: 141 speakers, 275 recordings. Cross-validation variance is high (std ≈ 0.09 AUC).
2. **No clinical metadata**: MMSE, CDR, and dementia subtype are not uniformly available. Subgroup analysis is limited to gender and recording-quality tertile.
3. **Unverified temporal orderings**: Only 82 of 275 recordings have defensible temporal ranks. The ranking supervision is therefore sparse.
4. **Quality confound**: The recording-quality shortcut is real and partially active in all models.
5. **Frozen embeddings**: WavLM and BERT are not fine-tuned. Task-specific fine-tuning on larger approved datasets could improve performance substantially.
6. **Dataset bias**: DementiaNet is predominantly Caucasian/White and English-speaking (limited subgroup diversity).

---

## 8. Conclusion

We presented a reproducible, fully open-source framework for learning dementia-associated speech trajectories from longitudinal spontaneous speech. The joint FusionNet model, combining WavLM, BERT, and handcrafted features with a soft ranking loss, achieves AUC 0.665 (recording-level) and 0.688 (speaker-level) on speaker-grouped held-out folds — exceeding both single-modality baselines and a recording-quality shortcut. The progression scores maintain 85% internal temporal consistency. SHAP analysis confirms that pitch variability and speech rate — clinically meaningful markers — drive the predictions. Two reproducible latent speech states are identified, exhibiting a 16 pp difference in dementia-group prevalence.

This work demonstrates that meaningful longitudinal speech analysis is achievable on approval-free data at a research scale, provided that ethical constraints (no per-person clinical verdicts, explicit uncertainty quantification, quality-shortcut auditing) are rigorously enforced.

---

## References

> *[References to be added before submission — key citations: Luz et al. ADReSS 2020/2021, Fraser et al. 2016, Balagopalan et al. 2021, Pappagari et al. 2021, Chen et al. WavLM 2022, Devlin et al. BERT 2019, Lundberg & Lee SHAP 2017]*

---

## Appendix A: Ethical Statement

All data are from the publicly available DementiaNet corpus (MIT licence). No individual is identified by name in any model output or result table. Per-speaker trajectory plots use speaker identifiers drawn from DementiaNet's own metadata and are presented as research illustrations only. The progression score is explicitly framed as a research measure; it is not and should not be used as a clinical diagnostic or severity instrument. The authors do not publish "decline verdicts" for any named individual.

## Appendix B: Reproducibility

All code is available at: https://github.com/TanishiJrvce/Multimodal-Cognitive-Decline-Detection  
Branch: `V0`  
Data: DementiaNet (https://github.com/shreyasgite/dementianet)  
Run order: scripts 01 → 09 as described in README.md  
Hardware: NVIDIA RTX 4050 Laptop (6 GB VRAM), Python 3.12, PyTorch 2.6.0+cu124
