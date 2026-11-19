"""Extract frozen pretrained embeddings per recording.

Outputs:
  data/processed/embeddings_wavlm.npy   shape (N, 768)  – WavLM-base-plus mean-pooled
  data/processed/embeddings_bert.npy    shape (N, 768)  – BERT-base-uncased [CLS]
  data/processed/embeddings_index.csv   file  <-> row-index mapping for both arrays

Both models are used FROZEN (no fine-tuning). Rationale:
  - WavLM captures paralinguistic/prosodic patterns unseen by handcrafted features.
  - BERT [CLS] captures higher-level syntactic/semantic coherence.
  - Freezing prevents overfitting on our small N=275 dataset.
  - If fine-tuning is later explored, it must only use training speakers.

Design refs: docs/FEATURE_DESIGN.md §Representations, Edge AI class 19/22.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import json
import torch
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── WavLM ──────────────────────────────────────────────────────────────────────
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

WAVLM_ID = "microsoft/wavlm-base-plus"
print(f"Loading WavLM ({WAVLM_ID}) …")
wav_processor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_ID)
wav_model = WavLMModel.from_pretrained(WAVLM_ID).to(DEVICE).eval()

# ── BERT ───────────────────────────────────────────────────────────────────────
from transformers import BertTokenizerFast, BertModel

BERT_ID = "bert-base-uncased"
print(f"Loading BERT ({BERT_ID}) …")
bert_tok = BertTokenizerFast.from_pretrained(BERT_ID)
bert_model = BertModel.from_pretrained(BERT_ID).to(DEVICE).eval()

# ── Manifest ───────────────────────────────────────────────────────────────────
m = pd.read_csv("data/processed/manifest.csv")

wavlm_rows, bert_rows, index_rows = [], [], []

for idx, r in enumerate(tqdm(m.itertuples(), total=len(m), desc="Embeddings")):
    wav_path = Path("data/processed/audio16k") / f"{r.file}.wav"
    tj_path  = Path("data/processed/transcripts") / f"{r.file}.json"

    missing = []
    if not wav_path.exists(): missing.append("audio16k")
    if not tj_path.exists():  missing.append("transcript")
    if missing:
        print(f"MISSING {r.file}: {missing}")
        continue

    # ── WavLM: mean-pool last hidden layer ───────────────────────────────────
    wav, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == 16000, f"Expected 16 kHz, got {sr} for {r.file}"

    # WavLM max context ~30 s at 16 kHz → chunk-and-average for longer files
    chunk_sec = 30
    chunk_len = chunk_sec * sr
    chunks = [wav[i:i+chunk_len] for i in range(0, len(wav), chunk_len)]
    chunk_embs = []
    with torch.no_grad():
        for ch in chunks:
            inp = wav_processor(ch, sampling_rate=16000, return_tensors="pt",
                                padding=True).to(DEVICE)
            out = wav_model(**inp)
            # mean-pool across time (last hidden state)
            chunk_embs.append(out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy())
    wavlm_emb = np.mean(chunk_embs, axis=0)   # (768,)

    # ── BERT: [CLS] token of full transcript ─────────────────────────────────
    d = json.load(open(tj_path))
    text = d.get("text", "").strip()
    if not text:
        words = [w["w"] for w in d.get("words", [])]
        text = " ".join(words)
    # Truncate to BERT max 512 tokens (keep first 510 word-piece tokens + [CLS]/[SEP])
    enc = bert_tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = bert_model(**enc)
    bert_emb = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()   # [CLS] (768,)

    wavlm_rows.append(wavlm_emb)
    bert_rows.append(bert_emb)
    index_rows.append({"file": r.file, "row_index": len(index_rows)})

# ── Save ───────────────────────────────────────────────────────────────────────
out_dir = Path("data/processed")
np.save(out_dir / "embeddings_wavlm.npy", np.stack(wavlm_rows))
np.save(out_dir / "embeddings_bert.npy",  np.stack(bert_rows))
pd.DataFrame(index_rows).to_csv(out_dir / "embeddings_index.csv", index=False)

print(f"\nSaved {len(index_rows)} recordings")
print(f"  WavLM: {np.stack(wavlm_rows).shape}")
print(f"  BERT:  {np.stack(bert_rows).shape}")
print("embeddings_index.csv rows:", len(index_rows))
