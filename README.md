# Explainable Multimodal Learning of Cognitive Decline Trajectories from Longitudinal Spontaneous Speech

**Research question.** Can acoustic and linguistic representations, trained with weak temporal
supervision and within-person comparisons, capture dementia-associated speech trajectories and
reveal reproducible intermediate speech states?

Dataset: [DementiaNet](https://github.com/shreyasgite/dementianet) (MIT). Manifest counts only
(train: 227 recordings / 116 speaker folders; val: 48 / 25) – not a verified full inventory.
No clinical MCI labels or MMSE targets exist. Model-derived states are **never** called "MCI".

## Design

| Component | Role |
|---|---|
| Group supervision | dementia-associated vs comparison speakers (BCE) |
| Weak temporal supervision | soft pairwise ranking of defensibly ordered recordings from the same dementia speaker |
| Within-person comparison | delta features / embeddings relative to the speaker's earliest recording |

**Score definitions** (`src/mcd/models.py`)
- `group_logit` (per segment) → speaker-level prediction = mean logit over all of a speaker's segments. Group ≠ proof of impairment in each recording.
- `progression` (per segment) → recording score = mean over segments; used only for *relative* ordering. Not calibrated, not monotonic-by-assumption.
- Ranking loss: `softplus(-(s_later - s_earlier))` – a soft preference, applied only to pairs in `data/processed/recording_order.csv` with a stated `order_basis`. Ambiguous filename suffixes are never used.

## Pipeline

1. `scripts/01_build_manifest.py` – merge metadata + split files; speaker-grouped CV folds; curated ordering.
2. `scripts/02_preprocess.py` – 16 kHz mono, VAD, ASR (faster-whisper, word timestamps, disfluencies kept), diarisation / target-speaker selection.
3. `scripts/03_extract_features.py` – handcrafted acoustic + linguistic; frozen WavLM and BERT embeddings (segment level).
4. `scripts/04_train.py` – baselines (audio / text / fused; classification-only) then joint model.
5. `scripts/05_states.py` – state discovery (k = 1..6, BIC + stability), train-only fit.
6. `scripts/06_evaluate.py` – speaker-level metrics, held-out ordering, stability, within-person changes, SHAP, sensitivity.

## Leakage rules
All segments/recordings of a speaker stay in one partition (`GroupKFold` on speaker id). Scalers, feature
selection, fusion and clustering are fit on train folds only. Temporal-order recovery is evidence of task
performance, not independent clinical validation; features used to define states cannot validate them.

## Setup
```
pip install -e .
```
Put DementiaNet files in `data/raw/` (not committed). See `docs/ROADMAP.md`.
