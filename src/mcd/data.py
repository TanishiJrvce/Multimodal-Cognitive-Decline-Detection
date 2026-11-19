"""Manifests, speaker-grouped splits, and defensible recording-order pairs."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def speaker_folds(manifest: pd.DataFrame, n_splits=5, seed=0):
    """Yield (train_idx, test_idx) with all recordings of a speaker in one partition.
    manifest needs columns: speaker, label (1 = dementia-associated, 0 = comparison)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from sgkf.split(manifest, manifest["label"], groups=manifest["speaker"])


def order_pairs(order: pd.DataFrame) -> pd.DataFrame:
    """order columns: speaker, recording, rank (int, lower=earlier), order_basis (str).
    Only rows with a non-empty order_basis and distinct ranks form pairs; within dementia speakers only.
    Returns columns: speaker, early, late, gap_rank."""
    order = order.dropna(subset=["rank", "order_basis"])
    out = []
    for spk, g in order.groupby("speaker"):
        g = g.sort_values("rank")
        rows = g.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[j]["rank"] > rows[i]["rank"]:
                    out.append((spk, rows[i]["recording"], rows[j]["recording"],
                                rows[j]["rank"] - rows[i]["rank"]))
    return pd.DataFrame(out, columns=["speaker", "early", "late", "gap_rank"])


def within_person_delta(feats: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
    """Subtract each speaker's earliest-recording features (train-time reference)."""
    first = order.sort_values("rank").groupby("speaker").first()["recording"]
    ref = feats.loc[first.values].set_index(first.index)
    spk = order.set_index("recording")["speaker"].reindex(feats.index)
    return feats - ref.reindex(spk.values).set_axis(feats.index)
