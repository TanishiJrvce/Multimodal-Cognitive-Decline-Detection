# Feature & pipeline design decisions

Derived from a slide-by-slide reading of the Edge AI course (https://adithyamd.com/courses/edge-ai/):
Classes 6, 8, 11, 19, 20, 21, 22, 26, 27, 31. Each decision below names the point it comes from and
what we do about it. The course targets microcontrollers; we borrow the *reasoning*, not the hardware limits.

## 1. Deep points and their consequences for this project

**A. "It was never the same": training-serving skew and drift (Class 6).**
Recorded in a quiet lab, heard on a noisy street. In DementiaNet the skew is *inside the dataset*: the
"15 years before symptoms" clips come from older broadcasts/archives than "after symptoms" clips, and comparison
speakers were sampled by age bucket. Recording era, codec, bandwidth and noise are therefore **correlated with
the temporal label**. A ranking head can score well by detecting audio era, not cognition.
-> Compute quality covariates for every recording (section 3). Add **shortcut baselines** (below). Report
temporal-ordering performance with and without quality features regressed out. Treat any gain as unproven until
the quality-only baseline is beaten.

**B. "No amount of ML fixes a sensor that never recorded the signal" (Class 11).**
Check bandwidth before trusting a feature. Many YouTube clips are band-limited or heavily compressed.
-> Measure **effective bandwidth at the original 44.1 kHz *before* downsampling** (afterwards the evidence is gone).
Voice-quality measures (jitter, shimmer, HNR) are fragile on compressed/noisy audio: computed, but only
interpreted on recordings passing a quality gate, and always reported with the quality covariate.

**C. The four tests of a feature: separates, robust, cheap, not redundant (Class 19).**
"Robust" = ignores what should not matter (volume, distance, device). Loudness and peak failed the
redundancy test; ZCR/crest factor survive gain changes.
-> Feature screening on **training folds only**: (1) separates, (2) robust: |corr| with SNR, bandwidth, loudness,
duration below threshold, (3) cheap, (4) drop one of any pair with |Spearman| > 0.9 (e.g. speech rate vs
articulation rate, pause count vs pause ratio). "Domain knowledge finds candidates; plots check them."
Absolute loudness/peak are not predictors (only covariates).

**D. Silence has high ZCR; a fan looks like voice (Class 19 slide notes + lab).**
ZCR is meaningless until energy says something is there; a loud low-ZCR hum fools a loudness/ZCR detector;
the fix is *frame-to-frame loudness variation* (syllable rhythm) or spectral shape.
-> Pause detection is **energy-first**. Speech presence in YouTube audio (music beds, applause, interviewer) is not
decided by loudness alone: use ASR word timestamps + a neural VAD, and a syllabic-modulation check as a sanity test.

**E. Thresholds differ per room; real systems learn or adapt them (Class 19 lab notes, Class 24 pointer).**
-> No fixed dBFS thresholds. Pause threshold is **relative to each recording's noise floor** (e.g. 10th-percentile
frame energy + margin). The minimum pause duration (default 250 ms) is a *parameter we vary* (150/250/500/1000 ms)
and report sensitivity for.

**F. Hang-over / smoothing trade flicker for delay (Class 20).**
Raw frame decisions flicker; hold-on bridges gaps between syllables but delays offsets.
-> Frame-level speech/non-speech decisions are smoothed (majority vote + hangover) before pauses are counted;
pause statistics are reported for several smoothing settings, since "pause" is partly a smoothing artefact.

**G. Window length: >= the longest event; too short = noisy features; too long = blurred (Class 20).**
Overlapping windows cut at random are the most common leakage in stream projects (Class 31).
-> Features have their own minimum windows: lexical diversity (MATTR) needs hundreds of words, coherence needs
>= several sentences, pause rate needs tens of seconds. Clips are short (about 30-60 s, to be confirmed in
`quality.csv`), so most features are **recording-level**. Window length is a sensitivity analysis
(15/30/60 s). **Split by person first, then cut windows.** Overlapping windows never cross the split.
Segments from one person are not independent samples (more segments != more participants).

**H. Standardise with training statistics only; same scaling everywhere (Classes 6, 19, 22).**
"The classic bug": scaling differs between training and use. -> One `preprocess` module for every file; scalers
fit per fold on training speakers, stored, and applied unchanged to held-out speakers.

**I. Anti-alias before decimating; DC removal; pre-emphasis; Hann window (Classes 8, 21).**
Skipping the low-pass folds high frequencies down. -> 44.1 kHz stereo -> mono -> polyphase low-pass
resample to 16 kHz (`resample_poly`), remove DC, pre-emphasis 0.97 only for spectral features. Speech
survives 8 kHz, so 16 kHz is ample; F0 by autocorrelation (noise-robust, averages over the window).
Frames: 25-30 ms window, 10 ms hop, Hann, 40 mel bands / 13 MFCC (course front end: 30 ms/20 ms/40 mel).

**J. Fusion ladder and when to climb it (Class 22).**
Reasons to fuse: complementary, redundant, coverage. Fusion costs synchronisation and calibration; add a second
modality only if the first, tuned, cannot do the job. Early/feature/late fusion trade cross-modal learning against
data need; late fusion degrades gracefully; **the cost of each error type decides the combiner**.
-> Modalities are aligned on the same time window using ASR word timestamps (with tolerance, analogous to the
"coincidence window": ASR timestamps are noisy). Baselines in order: audio-only, text-only, late fusion,
feature fusion. Train with **modality dropout** so the fused model survives a missing/failed modality (ASR failure,
no usable speech). Report "fused only helps if it beats late fusion".

**K. Climb only as far as you have to; stopping early is a success (Classes 8, 27).**
-> Baseline ladder: majority class -> 1-2 feature rule (e.g. pause ratio + speech rate) -> classical ML on
handcrafted features -> frozen embeddings -> fusion -> temporal model. If a lower rung already matches a higher
one on held-out speakers, we report that and keep the simpler model.

**L. Accuracy lies; measure the right thing, per group (Classes 20, 26, 31).**
"99% accurate" can be unusable; the overall average weights groups by their count; report the **worst group**.
-> Speaker-level AUC / balanced accuracy / sensitivity at fixed specificity. Per-group tables: gender, ethnicity,
language/accent, dementia subtype, age-at-recording bucket, recording quality tertile. Group sizes are tiny
(~100 speakers): every per-group number carries a bootstrap-over-speakers interval and is flagged when n is small.

**M. Three ways to fool yourself (Class 31).**
(1) overlapping windows split at random; (2) same speaker in train and test; (3) tuning on the test set.
-> Speaker-grouped splits only; thresholds, k, hyperparameters chosen by inner CV on training speakers;
the official validation split and any final held-out test are touched once. Five layers of evaluation planned
(model quality, system behaviour, resources/runtime, interpretability/user trust, responsible AI), not just AUC.

**N. Responsible AI at every stage; moral feasibility first (Classes 26, 27).**
Who is recorded, who is missing, labels, evaluation, deployment, monitoring. Dataset feasibility "often kills a
project": coverage matters more than size; <~20 speakers per condition is risky.
-> Document the cohort honestly: public figures, famous, likely mostly English-speaking and Caucasian; subtype mix;
approximate dates. No clinical use or individual diagnosis claims; the progression score is a research
measure; group membership != impairment in each recording; threshold choice is an ethical decision, so we report
operating-point trade-offs instead of a single "best" threshold. Dual-use note: speech-based cognitive screening
of identifiable people is sensitive; results are reported at group/aggregate level; we do not publish per-person
"decline" verdicts for named individuals.

**O. Keep results, not raw; but keep the unsure samples (Class 6).**
-> Cache features/embeddings per recording; inspect low-confidence and mis-ranked recordings qualitatively
(they are the most informative for error analysis).

## 2. Baselines that guard against shortcuts
1. **Majority/chance.**
2. **Quality-only model** (bandwidth, SNR, clipping, duration, loudness, channel layout): predicts group? predicts temporal order?
3. **Metadata-only model** (gender, language) for group classification.
4. **Rule on 1-2 features** (pause ratio, speech rate).
If 2 or 3 approaches the full model, the headline claim is weakened and says so.

## 3. Quality covariates (not predictors; used for sensitivity analysis and confound checks)
effective bandwidth (99% spectral rolloff at original rate), SNR estimate (speech-level vs noise-floor percentiles),
clipping fraction, raw loudness (dBFS), DC offset, duration, original sample rate/channels, channel correlation
(true stereo vs dual mono), ASR mean word confidence, interviewer-speech share.

## 4. Feature list (v1)
- **Acoustic:** speech rate, articulation rate, pause count/duration/ratio (adaptive, smoothed), F0 mean/SD/range, jitter, shimmer, HNR (quality-gated), spectral centroid/flatness, MFCC 1-13 mean/SD.
- **Linguistic:** MATTR-50, filler rate, immediate and n-gram repetition, mean sentence length, syntactic depth, adjacent-sentence coherence.
- **Within-person:** change relative to the speaker's earliest ranked recording (only where order is defensible).
