"""Frozen WavLM / BERT embeddings (mean pooled)."""
import numpy as np
import torch


@torch.no_grad()
def wavlm_embed(wav16k: np.ndarray, name="microsoft/wavlm-base-plus", layer=-1, device="cpu"):
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained(name)
    m = WavLMModel.from_pretrained(name).to(device).eval()
    x = fe(wav16k, sampling_rate=16000, return_tensors="pt").to(device)
    h = m(**x, output_hidden_states=True).hidden_states[layer]
    return h.mean(1).squeeze(0).cpu().numpy()


@torch.no_grad()
def bert_embed(text: str, name="bert-base-uncased", device="cpu"):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModel.from_pretrained(name).to(device).eval()
    x = tok(text, truncation=True, max_length=512, return_tensors="pt").to(device)
    return m(**x).last_hidden_state[:, 0].squeeze(0).cpu().numpy()
