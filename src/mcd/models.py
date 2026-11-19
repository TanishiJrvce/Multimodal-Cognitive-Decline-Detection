"""Compact fusion model with a group head and a progression head, plus joint loss.

FusionNet takes three separate input tensors:
  feat   – handcrafted features (B, feat_dim)
  wavlm  – WavLM embeddings    (B, 768), zeros if audio modality dropped
  bert   – BERT embeddings     (B, 768), zeros if text modality dropped

Outputs:
  group_logit  (B,) – log-odds of dementia group membership
  prog_score   (B,) – relative progression / speech-change score (unitless, unbounded)

V1 FIX: modality masking is now explicit and independent per stream.
  Each stream is zeroed only when its own key is absent from mod_names.
  "audio" absent → zero WavLM only.
  "text"  absent → zero BERT only.
  "feat"  absent → zero handcrafted only.
  No cross-stream contamination.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):
    """Three-stream fusion: handcrafted + WavLM + BERT → shared MLP → two heads.

    Small by design (~100 independent speakers):
      - 64-unit hidden layers
      - dropout 0.3
      - GELU activations
    """

    def __init__(self, feat_dim: int, wavlm_dim: int = 768, bert_dim: int = 768,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        # Per-modality encoders: LayerNorm → Dropout → Linear → GELU
        self.enc_feat  = nn.Sequential(
            nn.LayerNorm(feat_dim),  nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden), nn.GELU())
        self.enc_wavlm = nn.Sequential(
            nn.LayerNorm(wavlm_dim), nn.Dropout(dropout),
            nn.Linear(wavlm_dim, hidden), nn.GELU())
        self.enc_bert  = nn.Sequential(
            nn.LayerNorm(bert_dim),  nn.Dropout(dropout),
            nn.Linear(bert_dim, hidden),  nn.GELU())

        # Shared fusion MLP
        self.shared = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(dropout))

        # Task heads
        self.group_head = nn.Linear(hidden, 1)   # dementia vs comparison (logit)
        self.prog_head  = nn.Linear(hidden, 1)   # progression score (unitless)

    def forward(self, feat: torch.Tensor, wavlm: torch.Tensor,
                bert: torch.Tensor):
        z_feat  = self.enc_feat(feat)
        z_wavlm = self.enc_wavlm(wavlm)
        z_bert  = self.enc_bert(bert)
        z = self.shared(torch.cat([z_feat, z_wavlm, z_bert], dim=-1))
        return self.group_head(z).squeeze(-1), self.prog_head(z).squeeze(-1)


def apply_modality_mask(X_feat, X_wavlm, X_bert, mod_names: set):
    """Zero-out modality streams that are excluded.

    V1 FIX: Each stream is controlled INDEPENDENTLY by its own key.
      "feat"  absent → zero handcrafted features.
      "audio" absent → zero WavLM.
      "text"  absent → zero BERT.
    This replaces the original bug where "audio" absent also zeroed handcrafted features.
    """
    import numpy as np
    X_f = X_feat.copy()  if hasattr(X_feat,  'copy') else X_feat.clone()
    X_w = X_wavlm.copy() if hasattr(X_wavlm, 'copy') else X_wavlm.clone()
    X_b = X_bert.copy()  if hasattr(X_bert,  'copy') else X_bert.clone()

    if "feat" not in mod_names:
        if hasattr(X_f, 'fill_'):
            X_f = X_f.fill_(0.0)
        else:
            X_f = np.zeros_like(X_f)
    if "audio" not in mod_names:
        if hasattr(X_w, 'fill_'):
            X_w = X_w.fill_(0.0)
        else:
            X_w = np.zeros_like(X_w)
    if "text" not in mod_names:
        if hasattr(X_b, 'fill_'):
            X_b = X_b.fill_(0.0)
        else:
            X_b = np.zeros_like(X_b)
    return X_f, X_w, X_b


def joint_loss(logit: torch.Tensor, prog: torch.Tensor,
               y: torch.Tensor, pairs: list,
               batch_indices: list, lambda_rank: float = 0.5,
               margin: float = 0.0) -> torch.Tensor:
    """Compute joint classification + pairwise ranking loss.

    Args:
        logit:         (B,) group logits
        prog:          (B,) progression scores
        y:             (B,) binary labels (float)
        pairs:         list of (global_i, global_j) index pairs where i is
                       earlier than j (same speaker, defensible order,
                       strictly different rank bucket)
        batch_indices: list of global indices for the current batch
        lambda_rank:   weight on ranking loss (0 = classification only)
        margin:        minimum required prog[j] - prog[i]
    """
    # Classification loss (BCE)
    cls_loss = F.binary_cross_entropy_with_logits(logit, y.float())

    # Pairwise ranking loss (only if lambda_rank > 0 and pairs exist)
    if lambda_rank == 0 or not pairs:
        return cls_loss

    batch_set = {g: l for l, g in enumerate(batch_indices)}
    rank_losses = []
    for gi, gj in pairs:
        if gi in batch_set and gj in batch_set:
            li, lj = batch_set[gi], batch_set[gj]
            # prog[j] should be > prog[i] (later = higher progression)
            rank_losses.append(F.softplus(-(prog[lj] - prog[li] - margin)))
    if not rank_losses:
        return cls_loss

    rank_loss = torch.stack(rank_losses).mean()
    return cls_loss + lambda_rank * rank_loss


def speaker_aggregate(values, speakers):
    """Mean over all segments/recordings per speaker."""
    import pandas as pd
    return pd.Series(values).groupby(pd.Series(speakers).values).mean()
