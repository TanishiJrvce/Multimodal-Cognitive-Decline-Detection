"""Compact fusion model with a group head and a progression head, plus joint loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):
    """Inputs: dict modality -> (B, d_m). Modalities may be any subset (audio-only, text-only, fused).
    Each modality: LayerNorm -> Linear(hidden). Fusion = concat -> shared MLP. Small on purpose
    (~100 independent speakers)."""

    def __init__(self, dims: dict, hidden=64, dropout=0.3):
        super().__init__()
        self.enc = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU())
                                  for k, d in dims.items()})
        self.shared = nn.Sequential(nn.Linear(hidden * len(dims), hidden), nn.GELU(), nn.Dropout(dropout))
        self.group_head = nn.Linear(hidden, 1)   # dementia-associated vs comparison (logit)
        self.prog_head = nn.Linear(hidden, 1)    # relative speech-change score (unitless)

    def forward(self, x: dict):
        z = self.shared(torch.cat([self.enc[k](x[k]) for k in self.enc], dim=-1))
        return self.group_head(z).squeeze(-1), self.prog_head(z).squeeze(-1), z


def joint_loss(g_logit, y, s_early=None, s_late=None, lambda_rank=0.5, margin=0.0, pair_w=None):
    """BCE for group + soft pairwise ranking on ordered pairs (dementia speakers only).
    lambda_rank=0 gives the classification-only baseline."""
    loss = F.binary_cross_entropy_with_logits(g_logit, y)
    if lambda_rank > 0 and s_early is not None and len(s_early):
        r = F.softplus(-(s_late - s_early - margin))
        if pair_w is not None:
            r = r * pair_w
        loss = loss + lambda_rank * r.mean()
    return loss


def speaker_aggregate(values, speakers):
    """Mean over all segments/recordings per speaker."""
    import pandas as pd
    return pd.Series(values).groupby(pd.Series(speakers).values).mean()
