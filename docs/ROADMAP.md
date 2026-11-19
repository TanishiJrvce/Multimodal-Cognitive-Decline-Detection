# Roadmap

1. **Data (lightweight)** – download manifests/metadata from DementiaNet; build `manifest.csv` (speaker, recording, label, split, gender, age proxy) and `recording_order.csv` (speaker, recording, rank, order_basis). Only rank recordings with an explicit, documented basis (e.g. stated year); log every excluded ambiguity.
2. **Features + baselines** – ASR, target-speaker separation, handcrafted + frozen embeddings; LR/SVM/GBM per modality (audio, text, fused), speaker-grouped CV, speaker-level AUC.
3. **Joint model** – `FusionNet` with `lambda_rank` ∈ {0, >0}; multiple seeds; paired comparison across folds. Ablate within-person delta inputs.
4. **States** – on train-fold progression-aware embeddings: compare k=1..6 (BIC, silhouette, bootstrap ARI, cross-seed/split ARI); test ordering of states vs. time within person; evaluate on features *not* used to build states.
5. **Explainability** – SHAP on handcrafted-feature models and on fused-model heads (kernel SHAP on feature groups); within-person feature change plots.
6. **Robustness** – recording quality (SNR), topic, diagnostic-disclosure keyword masking, comparison-group trajectories vs age/elapsed time; pre-symptom recordings analysed descriptively.
