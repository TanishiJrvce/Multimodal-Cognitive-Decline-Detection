# Feature design decisions

Informed by the Edge AI course decks (Class 19 *Feature engineering*, Class 22 *Combining features*;
https://adithyamd.com/courses/edge-ai/). Principles borrowed, and how we apply them:

| Principle (source) | Decision here |
|---|---|
| A feature is a small number computed over a **window** (frames) | Fixed framing: 25 ms frames / 10 ms hop for spectral/prosodic features; 30 s segments for embeddings and segment-level stats. Same window definition for every recording. |
| Four tests: **separates, robust, cheap, not redundant** | Per-feature screening on **training folds only**: (1) group separation, (2) robustness to recording gain/noise/channel (feature must not track SNR or loudness), (3) cost, (4) drop one of any pair with \|Spearman\| > 0.9. |
| **Volume-robust** features (ZCR, crest factor survive gain change; loudness/peak do not) | YouTube audio has arbitrary gain. Absolute loudness/peak are *excluded* as predictors; audio is peak/loudness-normalised first; loudness is kept only as a **quality covariate** for sensitivity analysis. |
| ZCR means nothing in silence (noise crosses zero constantly) | VAD gates on **RMS first, then ZCR**: pauses are defined only on frames below the energy threshold, so noisy silence is not mistaken for speech. Pause thresholds come from ASR word gaps (>= 250 ms) cross-checked with VAD. |
| Fixed DSP front end + small learned model is the sweet spot | Handcrafted DSP features + frozen WavLM/BERT embeddings feeding a compact fusion head (already in `models.py`). |
| Fusion ladder: early / **feature** / **late** (Class 22). Late fusion is cheap and debuggable | Baselines: audio-only, text-only, **late fusion** (average of per-modality scores), then **feature fusion** (concat -> shared MLP). Reports whether fusion beats late fusion. |
| Features from different streams must be on the **same time window**, then scaled | Acoustic and text features are computed over the same 30 s segments (ASR word timestamps align text to audio); scalers fit on training data only. |
| Extra signals cost something; add only if one cannot do the job | Start with the small interpretable set; add MFCC/mel-statistics only if they improve held-out speaker-level metrics. |

## Feature list (v1)
- **Acoustic (per segment, then recording mean/SD):** speech rate, articulation rate, pause count/duration/ratio (energy-gated), F0 mean/SD/range, jitter, shimmer, HNR, spectral centroid/flatness, MFCC 1-13 mean/SD (optional tier).
- **Linguistic:** MATTR-50, filler rate, immediate and n-gram repetition, mean sentence length, syntactic depth (spaCy), adjacent-sentence semantic coherence.
- **Quality covariates (not predictors):** SNR estimate, mean loudness, clipping fraction, duration, interviewer-speech share.
