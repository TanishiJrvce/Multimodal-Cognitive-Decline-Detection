"""Extract handcrafted acoustic + linguistic features per recording.

Outputs: data/processed/features_acoustic.csv
         data/processed/features_linguistic.csv

Both are recording-level (one row per file). Within-person deltas are computed
in a later step, only for recordings with defensible temporal order.

Design rationale: docs/FEATURE_DESIGN.md

NOTE: Uses pure numpy/scipy for spectral features (no librosa/numba) to avoid
the 30-second numba JIT cold-start penalty on Windows.
"""
import json, re, math
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from scipy.signal import get_window
from scipy.fftpack import dct

FILLER = {"um", "uh", "er", "erm", "hmm", "ah", "mm", "hm"}
MIN_PAUSE = 0.25        # seconds – sensitivity check: 0.15, 0.25, 0.50 in later runs
ENERGY_PERCENTILE = 15  # adaptive noise floor: 15th percentile frame energy
PRE_EMPH = 0.97

# ---------- FAST NUMPY SPECTRAL ----------
def _stft_power(y: np.ndarray, n_fft: int = 512, hop: int = 160,
                win_len: int = 400) -> np.ndarray:
    """Return power spectrogram (n_fft//2+1, T) using numpy — no librosa."""
    window = get_window("hann", win_len, fftbins=True)
    # pad window to n_fft
    pad = (n_fft - win_len) // 2
    window = np.pad(window, (pad, n_fft - win_len - pad))
    frames = np.array([
        y[i:i+n_fft] * window
        for i in range(0, max(len(y) - n_fft, 1), hop)
        if i + n_fft <= len(y)
    ])                              # (T, n_fft)
    if len(frames) == 0:
        return np.zeros((n_fft // 2 + 1, 1))
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2  # (T, n_fft//2+1)
    return spec.T                   # (n_fft//2+1, T)

def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40,
                    fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """Triangular mel filterbank (n_mels, n_fft//2+1)."""
    if fmax is None:
        fmax = sr / 2.0
    n_bins = n_fft // 2 + 1
    # mel scale
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_bins))
    for m in range(n_mels):
        lo, cen, hi = bin_pts[m], bin_pts[m+1], bin_pts[m+2]
        for k in range(lo, cen):
            if cen > lo:
                fb[m, k] = (k - lo) / (cen - lo)
        for k in range(cen, hi):
            if hi > cen:
                fb[m, k] = (hi - k) / (hi - cen)
    return fb

_MEL_FB = None  # cached filterbank (computed once)

def spectral_feats(wav: np.ndarray, sr: int = 16000, n_mfcc: int = 13) -> dict:
    """Compute spectral_centroid, spectral_flatness, MFCCs using numpy only."""
    global _MEL_FB
    n_fft, hop, win_len = 512, 160, 400
    pe = np.append(wav[0], wav[1:] - PRE_EMPH * wav[:-1])
    S = _stft_power(pe, n_fft=n_fft, hop=hop, win_len=win_len)  # (257, T)
    freqs = np.linspace(0, sr / 2, S.shape[0])

    # spectral centroid
    S_sum = S.sum(axis=0) + 1e-10
    centroid = (freqs[:, None] * S).sum(axis=0) / S_sum    # (T,)

    # spectral flatness (geometric mean / arithmetic mean of power per frame)
    log_S = np.log(S + 1e-10)
    flatness = np.exp(log_S.mean(axis=0)) / (S.mean(axis=0) + 1e-10)

    # MFCCs: mel-filterbank → log → DCT
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(sr, n_fft, n_mels=40)
    mel_S = _MEL_FB @ S                                    # (40, T)
    log_mel = np.log(mel_S + 1e-10)
    # DCT-II (n_mfcc coefficients)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc, :]  # (13, T)

    out = dict(
        spectral_centroid_mean=float(centroid.mean()),
        spectral_flatness_mean=float(flatness.mean()),
    )
    for i in range(n_mfcc):
        out[f"mfcc{i+1:02d}_mean"] = float(mfcc[i].mean())
        out[f"mfcc{i+1:02d}_std"]  = float(mfcc[i].std())
    return out

def clean(word):
    return re.sub(r"[^a-z']", "", word.lower())

# ---------- ACOUSTIC ----------
def acoustic_feats(wav16k: np.ndarray, words: list[dict]) -> dict:
    """wav16k: float32 mono at 16 kHz, DC-removed, normalised."""
    sr = 16000
    # adaptive pause threshold: 15th-percentile frame energy
    n_fr, hop = int(0.025 * sr), int(0.010 * sr)
    idx = np.arange(0, max(len(wav16k) - n_fr, 1), hop)
    rms_frames = np.array([np.sqrt(np.mean(wav16k[i:i+n_fr]**2)) for i in idx]) + 1e-9
    thr = np.percentile(rms_frames, ENERGY_PERCENTILE)

    # speech-rate / pause (from ASR word timestamps, energy-gated)
    dur = words[-1]["end"] - words[0]["start"] if words else 0
    gaps = []
    if words:
        for a, b in zip(words, words[1:]):
            g = b["start"] - a["end"]
            if g >= MIN_PAUSE:
                # check if the gap frames are actually quiet (gate against music/noise)
                s_i, e_i = int(a["end"] * sr / hop), int(b["start"] * sr / hop)
                seg = rms_frames[max(s_i, 0):max(e_i, 1)]
                if len(seg) and seg.mean() < thr * 2.5:
                    gaps.append(g)
    n_words = len(words)
    speech_rate = n_words / max(dur, 1)
    # articulation rate: words per second of speech (excluding pause time)
    pause_total = sum(gaps)
    art_rate = n_words / max(dur - pause_total, 1)

    # F0 / voice quality via Praat (on 16 kHz)
    snd = parselmouth.Sound(wav16k, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    f0 = pitch.selected_array["frequency"]; f0 = f0[f0 > 0]
    pp = call(snd, "To PointProcess (periodic, cc)", 60, 400)
    try:
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = call(snd.to_harmonicity(), "Get mean", 0, 0)
    except Exception:
        jitter = shimmer = hnr = np.nan

    # Spectral features using fast numpy/scipy (no librosa/numba cold-start)
    spec_out = spectral_feats(wav16k, sr=sr, n_mfcc=13)

    out = dict(
        speech_rate=speech_rate, articulation_rate=art_rate,
        pause_count=len(gaps), pause_total_s=pause_total,
        pause_mean_s=float(np.mean(gaps)) if gaps else 0.0,
        pause_ratio=pause_total / max(dur, 1),
        f0_mean=float(f0.mean()) if len(f0) else np.nan,
        f0_std=float(f0.std()) if len(f0) else np.nan,
        f0_range=float(f0.max() - f0.min()) if len(f0) > 1 else np.nan,
        jitter=jitter, shimmer=shimmer, hnr=hnr,
    )
    out.update(spec_out)
    return out

# ---------- LINGUISTIC ----------
def linguistic_feats(words: list[str], dur: float) -> dict:
    """words: list of raw word strings from ASR (keep fillers, repetitions)."""
    low = [clean(w) for w in words]
    low = [w for w in low if w]  # drop punctuation-only tokens
    n = max(len(low), 1)
    # MATTR-50
    wins = [low[i:i+50] for i in range(0, max(n - 50, 1), 25)]
    mattr = float(np.mean([len(set(w)) / 50 for w in wins])) if len(low) >= 50 else len(set(low)) / n
    # filler rate
    fill = sum(w in FILLER for w in low)
    # immediate repetition
    imm_rep = sum(a == b for a, b in zip(low, low[1:]))
    # bigram repetition
    bigrams = list(zip(low, low[1:]))
    bi_rep = 1 - len(set(bigrams)) / max(len(bigrams), 1)
    # mean sentence length (heuristic: split on sentence-ending punctuation)
    text = " ".join(words)
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    mean_sent = float(np.mean([len(s.split()) for s in sents])) if sents else 0.0
    # type-token ratio (whole transcript, unwindowed, as complement to MATTR)
    ttr = len(set(low)) / n
    return dict(
        mattr50=mattr, ttr=ttr, filler_rate=fill / n,
        filler_count=fill, immediate_rep_rate=imm_rep / n,
        bigram_rep=bi_rep, n_words=n, words_per_min=60 * n / max(dur, 1),
        mean_sent_len=mean_sent, n_sents=len(sents),
    )

# ---------- MAIN ----------
import sys, os
os.environ.setdefault("TQDM_DISABLE", "1")   # disable tqdm progress bar in subprocess

m = pd.read_csv("data/processed/manifest.csv")
aud_rows, ling_rows = [], []

for i, r in enumerate(m.itertuples()):
    tj = Path("data/processed/transcripts") / f"{r.file}.json"
    wav_path = Path("data/processed/audio16k") / f"{r.file}.wav"
    if not tj.exists() or not wav_path.exists():
        print(f"MISSING {r.file}", flush=True); continue
    d = json.load(open(tj))
    wav, _ = sf.read(str(wav_path), dtype="float32")

    words_raw = [w["w"] for w in d["words"]]
    af = {"file": r.file}; af.update(acoustic_feats(wav, d["words"])); aud_rows.append(af)
    lf = {"file": r.file}; lf.update(linguistic_feats(words_raw, d["duration"])); ling_rows.append(lf)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(m)} done ({aud_rows[-1]['file']})", flush=True)

pd.DataFrame(aud_rows).to_csv("data/processed/features_acoustic.csv", index=False)
pd.DataFrame(ling_rows).to_csv("data/processed/features_linguistic.csv", index=False)
print(f"done {len(aud_rows)}", flush=True)
print(pd.DataFrame(aud_rows).describe().T[["mean","std","min","50%","max"]].round(3))
