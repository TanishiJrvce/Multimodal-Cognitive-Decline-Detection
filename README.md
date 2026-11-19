# Explainable Multimodal Learning of Cognitive Decline Trajectories from Longitudinal Spontaneous Speech

> **Research Project — V0**  
> Dataset: [DementiaNet](https://github.com/shreyasgite/dementianet) (MIT licence)  
> Author: Tanishi Jain

---

## Research Question

Can acoustic and linguistic representations, trained with **weak temporal supervision** and within-person comparisons, capture dementia-associated speech trajectories and reveal reproducible intermediate speech states — without relying on binary classifier framing or unapproved clinical datasets?

---

## Documentation

| Document | Description |
|---|---|
| [`docs/MASTER_REPORT.md`](docs/MASTER_REPORT.md) | **Full consolidated report** — original results + secondary analysis findings + discussion |
| [`docs/PAPER_DRAFT.md`](docs/PAPER_DRAFT.md) | Academic paper draft |
| [`analysis/README.md`](analysis/README.md) | Secondary analysis package guide |

---

## Results Summary

All numbers are on speaker-grouped held-out validation folds (no speaker appears in both train and val).

| Model | AUC (mean ± std) | Balanced Acc |
|---|---|---|
| Quality-only baseline (loudness/SNR) | 0.625 | — |
| Handcrafted features — SVM | 0.600 ± 0.080 | — |
| **V_feat** — FusionNet, features only | 0.500 | 0.500 |
| **V_audio** — WavLM + features | 0.636 ± 0.098 | 0.607 |
| **V_text** — BERT + features | 0.609 ± 0.088 | 0.576 |
| **V_cls** — all modalities, cls-only loss | 0.656 ± 0.097 | 0.615 |
| **V_joint** — all modalities + ranking loss | **0.665 ± 0.088** | **0.622** |

**Speaker-level AUC** (averaging predictions across a speaker's recordings): **0.688 ± 0.109**

**Temporal pair accuracy**: 85% (34 out of 40 defensibly ordered pairs have increasing progression scores) — this is an internal consistency check, not independent clinical validation.

### Latent Speech States (GMM)

BIC selected **k = 2** states across all seeds and folds:

| State | N recordings | Mean prog. score | % Dementia group |
|---|---|---|---|
| State-0 | 176 | −0.333 | 40% |
| State-1 | 99 | +0.142 | 57% |

> ⚠ State labels are **not** clinical diagnoses. State-0 ≠ "healthy", State-1 ≠ "MCI or severe dementia". States reflect patterns in speech that correlate with group membership — nothing more.

### Key Finding

The joint model (WavLM acoustic embeddings + BERT linguistic embeddings + 37 screened handcrafted features, trained with classification + soft temporal ranking loss) achieves **AUC 0.665**, meaningfully exceeding the recording-quality shortcut (AUC 0.625). This demonstrates that the model is learning speech content and not merely responding to loudness or recording conditions.

---

## Dataset

**DementiaNet** — 275 recordings from 141 speakers (dementia group and healthy comparison), sourced from public interviews, distributed under MIT licence.

- Train: 227 recordings / 116 speakers
- Val: 48 recordings / 25 speakers
- Speaker-grouped splits: **no speaker appears in both sets**
- No MMSE scores, no MCI labels, no clinician ratings available

> ⚠ Manifest counts are estimates derived from filename parsing; not a clinically verified inventory.

---

## Pipeline

```
data/raw/                  ← DementiaNet files (not committed; see Setup)
  ↓
scripts/01_build_manifest.py    Stage 1: manifest, speaker-grouped splits, ordering
scripts/02_preprocess.py        Stage 1: 16 kHz mono, VAD
scripts/03_transcribe.py        Stage 1: ASR with faster-whisper (word timestamps)
scripts/04_extract_features.py  Stage 2: 41 handcrafted features (acoustic + linguistic)
scripts/05_embeddings.py        Stage 2: WavLM-base-plus + BERT-base-uncased embeddings
scripts/05b_feature_screen.py   Stage 2: quality-correlation and collinearity screening → 37 features
scripts/06_train_baselines.py   Stage 2: SVM/RF/LGBM baselines (AUC 0.60)
scripts/07_train_joint.py       Stage 3: FusionNet — 5 experiments × 5 seeds × 5 folds
scripts/08_states.py            Stage 3: GMM latent state discovery (BIC, silhouette)
scripts/09_evaluate.py          Stage 3: speaker metrics, SHAP, trajectory plots
```

### Feature Groups

**Acoustic (41 raw → 37 screened)**
- Pitch (F0): mean, std, range, % voiced, slope
- Voice quality: jitter, shimmer, HNR
- Energy: RMS mean/std, speech rate from energy-based VAD
- Spectral: MFCC 1–13 (mean + std), spectral centroid, flatness
- Pause: mean pause duration, pause rate, silence fraction

**Linguistic (11 features)**
- Type-token ratio, filler word rate, word rate, sentence length
- Mean word frequency (Zipf), lexical density, pronoun rate, repetition rate

**Embeddings**
- WavLM-base-plus: 768-dim mean-pooled frame representations
- BERT-base-uncased: 768-dim [CLS] token from transcript

---

## Model Architecture

```
FusionNet
├── Encoder streams (each: LayerNorm → Dropout(0.3) → Linear → GELU)
│   ├── Handcrafted features  → 64-dim
│   ├── WavLM embeddings      → 64-dim
│   └── BERT embeddings       → 64-dim
├── Fusion MLP: Linear(192→64) → GELU → Dropout(0.3)
└── Two heads:
    ├── group_logit  (1-dim) → dementia-associated group probability
    └── prog_score   (1-dim) → relative speech-change score (unitless)
```

**Joint loss**: `BCE(group_logit, y) + λ · mean softplus(−(prog[later] − prog[earlier]))`  
λ = 0.5. Ranking pairs from `data/processed/recording_order.csv` only — never inferred from filenames alone.

**Training**: 5-fold StratifiedGroupKFold × 5 seeds, Adam lr=1e-4, early stopping patience=10, max 100 epochs, modality dropout p=0.2.

---

## Interpretability

SHAP values computed via `TreeExplainer` on the best RandomForest baseline (fold 0, seed 0).  
See `figures/shap_beeswarm.png` and `data/results/shap_values.csv`.

Trajectory plots for 6 speakers with ≥ 2 defensibly ordered recordings are in `figures/trajectory_*.png`.

---

## Ethical Constraints

- No per-person "decline verdicts" are published for named individuals
- Progression score is a **research measure**, not a clinical severity score
- Recording quality features (loudness, peak, bandwidth) are covariates only, not predictors
- Temporal ordering is a **soft hypothesis**, not enforced; only defensible orderings are used
- No intermediate cluster is labelled "MCI" — labels are State-0 … State-k

---

## Setup

```bash
pip install -e .
pip install torch transformers faster-whisper soundfile praat-parselmouth shap
```

Place DementiaNet files in `data/raw/` (not committed — large audio files).  
Run scripts in order: 01 → 02 → 03 → 04 → 05 → 05b → 06 → 07 → 08 → 09.

GPU recommended (RTX 4050 or better). Script 07 takes ~30 min on a laptop GPU.

---

## Repository Structure

```
├── data/
│   ├── processed/           ← CSVs committed; WAVs/NPYs/transcripts in .gitignore
│   │   ├── manifest.csv
│   │   ├── quality.csv
│   │   ├── recording_order.csv
│   │   ├── features_acoustic.csv
│   │   ├── features_linguistic.csv
│   │   ├── features_screened.csv
│   │   ├── feature_list.json
│   │   └── features_delta.csv
│   └── results/             ← All result CSVs committed
├── figures/                 ← SHAP plot + 6 trajectory PNGs committed
├── scripts/                 ← All pipeline scripts
├── src/mcd/                 ← Python package (models, states, evaluate, data)
├── configs/
├── docs/
└── tests/
```

---

## Citation

If you use this code or findings, please cite:

```
Jain, T. (2026). Explainable Multimodal Learning of Cognitive Decline Trajectories 
from Longitudinal Spontaneous Speech. GitHub: TanishiJrvce/Multimodal-Cognitive-Decline-Detection.
```
