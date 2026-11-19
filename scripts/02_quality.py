"""Per-recording quality covariates, computed on the ORIGINAL audio (before any downsampling).
Not predictors: used for confound checks and sensitivity analysis (see docs/FEATURE_DESIGN.md)."""
import sys
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch
from tqdm import tqdm

m = pd.read_csv("data/processed/manifest.csv")
rows = []
for r in tqdm(m.itertuples(), total=len(m)):
    x, sr = sf.read(r.audio_path, always_2d=True, dtype="float32")
    ch = x.shape[1]
    ch_corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if ch > 1 and x[:, 0].std() > 0 and x[:, 1].std() > 0 else np.nan
    mono = x.mean(1)
    dur = len(mono) / sr
    dc = float(mono.mean())
    mono = mono - dc
    # frame energies (25 ms / 10 ms) for noise floor + speech level
    n, h = int(0.025 * sr), int(0.010 * sr)
    idx = np.arange(0, len(mono) - n, h)
    e = np.array([np.sqrt(np.mean(mono[i:i + n] ** 2)) for i in idx]) + 1e-9
    db = 20 * np.log10(e)
    noise, speech = np.percentile(db, 10), np.percentile(db, 90)
    # effective bandwidth: frequency below which 99% of (long-term) energy lies
    f, p = welch(mono, fs=sr, nperseg=4096)
    cum = np.cumsum(p) / p.sum()
    bw99 = float(f[np.searchsorted(cum, 0.99)])
    # energy above 8 kHz relative to total (is there anything beyond what 16 kHz keeps?)
    hf_ratio = float(p[f > 8000].sum() / p.sum()) if sr > 16000 else np.nan
    rows.append(dict(file=r.file, orig_sr=sr, channels=ch, ch_corr=ch_corr, duration_s=dur, dc_offset=dc,
                     loudness_dbfs=float(20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-9)),
                     peak=float(np.abs(mono).max()), clip_frac=float(np.mean(np.abs(x) > 0.999)),
                     snr_db=float(speech - noise), bw99_hz=bw99, hf_ratio_above8k=hf_ratio))
q = pd.DataFrame(rows)
q.to_csv("data/processed/quality.csv", index=False)
print(q.describe().T[["mean", "std", "min", "50%", "max"]].round(3))
