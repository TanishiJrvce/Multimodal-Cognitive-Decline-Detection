"""Late-fusion MLP: one small encoder per modality, concatenate, shared heads (class + MMSE regression)."""
import torch, torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, dims, n_classes=3, hidden=64, dropout=0.3):
        super().__init__()
        self.enc = nn.ModuleDict({k: nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout))
                                  for k, d in dims.items()})
        self.body = nn.Sequential(nn.Linear(hidden * len(dims), hidden), nn.ReLU(), nn.Dropout(dropout))
        self.cls = nn.Linear(hidden, n_classes)
        self.reg = nn.Linear(hidden, 1)

    def forward(self, xs):
        h = torch.cat([self.enc[k](xs[k]) for k in self.enc], dim=1)
        z = self.body(h)
        return self.cls(z), self.reg(z).squeeze(1)
