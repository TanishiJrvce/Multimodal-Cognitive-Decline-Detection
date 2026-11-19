"""Dataset loading. Manifest CSV cols: speaker,audio_path,transcript,label,mmse,age,gender.
label: 0=healthy, 1=MCI, 2=dementia. Mock mode generates synthetic features directly."""
import numpy as np, pandas as pd

def load_manifest(path):
    return pd.read_csv(path)

def make_mock(n=120, seed=0, n_classes=3):
    """Synthetic data with class signal spread unevenly across 3 modalities."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    meta = pd.DataFrame({
        "speaker": [f"S{i:03d}" for i in range(n)],
        "label": y,
        "mmse": 30 - 4 * y + rng.normal(0, 2, n),
        "age": rng.integers(55, 90, n),
        "gender": rng.choice(["M", "F"], n),
    })
    def block(d, strength):
        X = rng.normal(0, 1, (n, d))
        X[:, :4] += strength * y[:, None]   # first 4 dims informative
        return X.astype(np.float32)
    feats = {
        "acoustic": block(30, 0.5),
        "linguistic": block(20, 0.6),
        "wavlm": block(64, 0.25),
        "bert": block(64, 0.3),
    }
    names = {
        "acoustic": ["pause_rate", "speech_rate", "pitch_mean", "jitter"] + [f"ac_{i}" for i in range(26)],
        "linguistic": ["ttr", "mean_sent_len", "pronoun_ratio", "repetitions"] + [f"li_{i}" for i in range(16)],
        "wavlm": [f"wavlm_{i}" for i in range(64)],
        "bert": [f"bert_{i}" for i in range(64)],
    }
    return meta, feats, names
