"""Speaker-level metrics and held-out temporal ordering."""
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


def speaker_metrics(spk_scores, spk_labels):
    s, y = np.asarray(spk_scores), np.asarray(spk_labels)
    return dict(auc=roc_auc_score(y, s), bal_acc=balanced_accuracy_score(y, s > 0), n_speakers=len(y))


def pair_accuracy(rec_scores: dict, pairs):
    """Fraction of held-out pairs with score(late) > score(early); plus per-speaker Kendall tau via pairs.
    This is temporal-task performance, NOT independent clinical validation."""
    hits = [rec_scores[r.late] > rec_scores[r.early] for r in pairs.itertuples()]
    return dict(pair_acc=float(np.mean(hits)) if hits else np.nan, n_pairs=len(hits),
                n_speakers=pairs["speaker"].nunique())
