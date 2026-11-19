"""Preprocess + transcribe (faster-whisper, word timestamps, disfluency-preserving prompt).

Preprocessing (identical for every file; see docs/FEATURE_DESIGN.md):
  stereo -> mono, DC removal, anti-aliased polyphase resample to 16 kHz, peak-normalise to -1 dBFS.
Outputs per recording: data/processed/transcripts/<file>.json  (words with start/end/probability)
                       data/processed/audio16k/<file>.wav
Usage: python scripts/03_transcribe.py [--limit N] [--model medium.en] [--device cuda]
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--model", default="medium.en")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

# make CUDA libs shipped with torch visible to ctranslate2 (Windows)
import torch  # noqa
if sys.platform == "win32":
    os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
from faster_whisper import WhisperModel

PROMPT = "Um, so, uh, I was, I was thinking, you know, like, hmm... well, I mean, yeah."  # keeps fillers/repeats
out_a, out_t = Path("data/processed/audio16k"), Path("data/processed/transcripts")
out_a.mkdir(parents=True, exist_ok=True); out_t.mkdir(parents=True, exist_ok=True)

m = pd.read_csv("data/processed/manifest.csv")
if a.limit:  # balanced small test: alternate groups
    m = pd.concat([m[m.label == 1].head(a.limit // 2), m[m.label == 0].head(a.limit - a.limit // 2)])

model = WhisperModel(a.model, device=a.device, compute_type="float16" if a.device == "cuda" else "int8")

def load16k(path):
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    x = x.mean(1); x = x - x.mean()
    g = math.gcd(16000, sr)
    x = resample_poly(x, 16000 // g, sr // g).astype("float32")
    return x / (np.abs(x).max() + 1e-9) * 0.89  # -1 dBFS peak

for r in tqdm(m.itertuples(), total=len(m)):
    tj = out_t / f"{r.file}.json"
    if tj.exists():
        continue
    x = load16k(r.audio_path)
    sf.write(out_a / f"{r.file}.wav", x, 16000)
    segs, info = model.transcribe(x, language="en", word_timestamps=True, initial_prompt=PROMPT,
                                  condition_on_previous_text=False, vad_filter=False, beam_size=5)
    words = []
    for s in segs:
        for w in s.words:
            words.append(dict(w=w.word.strip(), start=round(w.start, 3), end=round(w.end, 3), p=round(w.probability, 3)))
    json.dump(dict(file=r.file, duration=len(x) / 16000, n_words=len(words), words=words), open(tj, "w"))
print("done", len(m))
