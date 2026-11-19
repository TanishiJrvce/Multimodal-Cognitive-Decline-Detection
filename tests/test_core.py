import pandas as pd, torch
from mcd.data import order_pairs, speaker_folds
from mcd.models import FusionNet, joint_loss


def test_pairs_need_basis():
    o = pd.DataFrame(dict(speaker=["a"] * 3, recording=["r1", "r2", "r3"], rank=[1, 2, None],
                          order_basis=["year", "year", None]))
    assert len(order_pairs(o)) == 1


def test_speaker_disjoint_folds():
    m = pd.DataFrame(dict(speaker=[f"s{i // 3}" for i in range(60)], label=[(i // 3) % 2 for i in range(60)]))
    for tr, te in speaker_folds(m, 4):
        assert not set(m.speaker[tr]) & set(m.speaker[te])


def test_model_loss():
    net = FusionNet(feat_dim=8, wavlm_dim=4, bert_dim=5)
    g, p = net(torch.randn(4, 8), torch.randn(4, 4), torch.randn(4, 5))
    loss = joint_loss(g, p, torch.ones(4), [(0, 1)], [0, 1, 2, 3])
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert net.prog_head.weight.grad.norm() > 0
