"""Handcrafted acoustic and linguistic features (per recording / segment)."""
import numpy as np


def acoustic_features(wav: np.ndarray, sr: int, words: list[dict] | None = None) -> dict:
    """wav mono float. words: [{'word','start','end'}] from ASR (for rate/pauses)."""
    import librosa, parselmouth
    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    f0 = snd.to_pitch().selected_array["frequency"]
    f0 = f0[f0 > 0]
    pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    hnr = parselmouth.praat.call(snd.to_harmonicity(), "Get mean", 0, 0)
    feats = dict(f0_mean=f0.mean() if len(f0) else np.nan, f0_std=f0.std() if len(f0) else np.nan,
                 jitter=jitter, shimmer=shimmer, hnr=hnr,
                 rms_mean=float(librosa.feature.rms(y=wav).mean()))
    if words:
        dur = words[-1]["end"] - words[0]["start"]
        gaps = np.array([b["start"] - a["end"] for a, b in zip(words, words[1:])])
        long = gaps[gaps > 0.25]
        feats.update(speech_rate=len(words) / max(dur, 1e-6), pause_count_per_min=len(long) / max(dur / 60, 1e-6),
                     pause_mean=long.mean() if len(long) else 0.0, pause_ratio=long.sum() / max(dur, 1e-6))
    return feats


FILLERS = {"um", "uh", "er", "erm", "hmm", "ah"}


def linguistic_features(words: list[str], sentences: list[str], sent_embs: np.ndarray | None = None) -> dict:
    """words: lowercase tokens incl. fillers/repetitions. sent_embs: (n_sent, d) for coherence."""
    n = max(len(words), 1)
    types = set(words)
    ttr_win = np.mean([len(set(words[i:i + 50])) / 50 for i in range(0, max(n - 50, 1), 25)]) if n >= 50 else len(types) / n
    bigrams = list(zip(words, words[1:]))
    rep_bigram = 1 - len(set(bigrams)) / max(len(bigrams), 1)
    immediate_rep = sum(a == b for a, b in zip(words, words[1:])) / n
    feats = dict(mattr50=ttr_win, filler_rate=sum(w in FILLERS for w in words) / n,
                 bigram_repetition=rep_bigram, immediate_repetition=immediate_rep,
                 mean_sent_len=np.mean([len(s.split()) for s in sentences]) if sentences else 0.0)
    if sent_embs is not None and len(sent_embs) > 1:
        e = sent_embs / np.linalg.norm(sent_embs, axis=1, keepdims=True)
        feats["coherence_adjacent"] = float((e[:-1] * e[1:]).sum(1).mean())
    return feats  # add spaCy dependency depth / clause counts in a later step
