# Manuscript-ready Results and Discussion

Source: final V0 commit a391a57e5fe127dd44e211ffe6bd19e0d5777ac0. Original results and newly executed secondary analyses are distinguished. This is editable research text, not a claim of clinical validation.

## Results

### Original model performance

The joint multimodal model achieved mean recording-level AUC 0.665 and balanced accuracy 0.622 across speaker-grouped evaluations. Averaging group probabilities within speakers yielded mean AUC 0.688 and balanced accuracy 0.651. Relative to classification-only training, joint training improved mean AUC by 0.0092, with positive differences in 15 of 25 matched runs. Seed-level mean differences ranged from -0.0169 to +0.0387, indicating heterogeneous contributions of the weak ranking objective.

### Speaker-level speech differences

After averaging recordings within speakers, mean qualifying pause duration was 88 ms longer and pause proportion 1.72 percentage points greater in the dementia-associated group; both standardised differences were approximately 0.29 pooled SD. MATTR-50 showed a difference of -0.0014. None of the 37 feature-wise speaker-label permutation tests survived BH correction. Available-covariate adjustment retained the direction of pause differences, with nominal p-values near 0.03, although their adjusted q-values remained above 0.05.

### Within-person rate decomposition

Among 31 speakers with distinct ordered recording buckets, the mean endpoint log word-rate change was -0.0946, corresponding to a geometric average decrease of 9.03% (pointwise speaker-bootstrap interval: 1.01% to 17.35% decrease). An exact decomposition assigned approximately 90.2% of that mean log change to pause-excluded word delivery and 9.8% to non-pause share. The overall log-rate sign-flip p-value was 0.0474 and exploratory four-contrast q-value 0.0721; absolute-rate and rank-based sensitivity analyses were less decisive.

### Temporal-score generalisation

Using bucket-averaged scores and equal speaker weighting, temporal pair concordance was 87.8% on training speakers and 55.3% on held-out speakers. The held-out bootstrap interval was 43.9%-65.6%. A five-seed held-out percentile ensemble achieved concordance 60.8%, with within-person ordering-permutation p = 0.124. Group discrimination by the held-out progression percentile was approximately chance at speaker level (AUC 0.505, interval 0.410-0.599).

### State and explanation analyses

One mixture component was favoured by BIC in 24 of 25 runs, while operational two-state assignments varied across held-out models for 219 of 275 recordings. In the RF explanation analysis, timing features had approximately 1.9 times the mean absolute SHAP per feature of spectral descriptors. Pause duration and pause proportion ranked in the top five in 96.6% and 85.7% of speaker-bootstrap explanation samples, respectively.

## Discussion

### Different outputs answer different scientific questions

The completed framework integrates dementia-associated group supervision with weak within-person ordering, but the outputs are not interchangeable measures of impairment. The group classifier provides partial discrimination across unseen speakers, whereas the progression head generalises temporal ordering less clearly and has little group discrimination after held-out normalisation. Shared representation learning therefore does not establish a shared clinical meaning for both heads. The contribution is an explicit investigation of this distinction, rather than a claim that classification performance automatically validates severity tracking.

### Fluency is not one measurement

Descriptive pausing differences and robust RF attribution directions identify temporal organisation as an interpretable component of the representation. However, the longitudinal decomposition shows that measured proportional slowing is mostly assigned to pause-excluded word delivery, not increasing silent-time share. This distinguishes within-speaker delivery dynamics from between-group pausing differences. Because the pause-excluded measure is word-based and derived from recognised timestamps, it cannot distinguish cognitive formulation demands from motor, respiratory, or transcription-related contributions. The estimated decomposition is exploratory and its uncertainty is scale-sensitive.

### Vocabulary breadth and cognitive efficiency need not coincide

The near-zero group difference in local lexical diversity coexists with timing-related patterns. This is compatible with speech-production efficiency and sampled vocabulary breadth being partially distinct properties. Primary longitudinal work has also found different sensitivities across fluency, semantic content, and lexical diversity [2]. Nevertheless, comparable MATTR values are not equivalent to intact semantic content, and slightly higher diversity in a score region does not establish compensation. Interview familiarity, sample length, and transcription behaviour remain competing explanations.

### Interpretability requires the right denominator

Spectral descriptors account for most summed RF attribution, but that family is also much larger. Normalisation by feature count reveals comparatively high attribution density in the three timing measures. Their stable importance under resampling of explanation-sample speakers supports their role as understandable model descriptors. These analyses concern the RF baseline, not direct attribution of the neural progression head, and correlated inputs prevent interpreting attribution shares as a causal partition of disease mechanisms.

### The progression coordinate includes sampling structure

Held-out progression percentiles relate to recording duration and multiple spectral variability measures. The negative association with bigram repetition attenuates after accounting for word count. These results indicate that the coordinate organises a mixture of speech and sampling properties. Shorter later-bucket clips describe corpus selection and cannot be equated with reduced patient speech productivity. Characterising these associations gives the score a more precise empirical meaning without assigning an exclusively cognitive cause.

### Continuous variation is better established than discrete stages

The state results provide a compact operational description of lower and higher score regions. Model selection more frequently supports one score population, and most recordings change low/high assignment across held-out fits. Consequently the study does not establish a universal sequence of discrete clinical stages or an MCI category. This result does not exclude a biological disease continuum; it identifies the level of structure resolved by this representation and dataset.

### Uncertainty is scientifically informative

Speaker-level feature effects, endpoint changes, and supplementary linear probes have broad uncertainty. A richer multivariate representation may carry useful group information even when no single handcrafted measurement is established after multiplicity correction. Conversely, the modest original joint-training gain cannot be described as decisive superiority from the available repeated-fold results. The analysis separates descriptive patterns, exploratory statistical evidence, and clinical interpretation instead of treating them as equivalent.

### Scope of inference

The cohort comprises public interview recordings with heterogeneous dementia-associated histories, sparse approximate timing, and unequal numbers of recordings per person. Clinical severity, biomarker status, education, exact recording age, and standardised elicitation demands are not observed in the committed tables. Analyses cannot causally separate healthy ageing from disease or infer individual diagnosis. Within those bounds, the study contributes a multimodal weak-supervision framework, evidence about functional speech timing, and a detailed account of the difference between learned temporal alignment and its generalisation.

## Primary references

- [[1] De Looze et al. (2021). Cognitive and Structural Correlates of Conversational Speech Timing in MCI and Mild-to-Moderate Alzheimer's Disease. DOI 10.3389/fnagi.2021.637404.](https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2021.637404/full)
- [[2] Mueller et al. (2018). Declines in Connected Language Are Associated with Very Early MCI: Results from WRAP. DOI 10.3389/fnagi.2017.00437.](https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2017.00437/full)
- [[3] Themistocleous, Eckerstrom and Kokkinakis (2020). Voice quality and speech fluency distinguish individuals with MCI from Healthy Controls. DOI 10.1371/journal.pone.0236009.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0236009)
- [[4] Ahmed et al. (2013). Connected speech as a marker of disease progression in autopsy-proven Alzheimer's disease. DOI 10.1093/brain/awt269.](https://pubmed.ncbi.nlm.nih.gov/24142144/)
- [[5] Heitz, Schneider and Langer (2024). The Influence of ASR on Linguistic Features and Automatic Alzheimer's Disease Detection from Spontaneous Speech. LREC-COLING.](https://aclanthology.org/2024.lrec-main.1386/)
- [[6] Fraser, Meltzer and Rudzicz (2016). Linguistic Features Identify Alzheimer's Disease in Narrative Speech. DOI 10.3233/JAD-150520.](https://journals.sagepub.com/doi/10.3233/JAD-150520)
- [[7] Nadeau and Bengio (1999). Inference for the Generalization Error. NeurIPS 12. Used as context for dependence-aware repeated-CV interpretation.](https://proceedings.neurips.cc/paper/1999/hash/7d12b66d3df6af8d429c1a357d8b9e1a-Abstract.html)
- [[8] Petti and Korhonen (2024). LoSST-AD: A Longitudinal Corpus for Tracking Alzheimer's Disease Related Changes in Spontaneous Speech. LREC-COLING.](https://aclanthology.org/2024.lrec-main.944/)
- [[9] Gao et al. (2025). ADCeleb: A Longitudinal Speech Dataset from Public Figures for Early Detection of Alzheimer's Disease. DOI 10.21437/Interspeech.2025-523.](https://www.isca-archive.org/interspeech_2025/gao25b_interspeech.html)