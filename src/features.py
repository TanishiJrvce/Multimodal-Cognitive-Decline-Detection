"""Feature extraction for real data. Each fn returns an (n, d) float32 array; results cached to data/features/*.npy."""
import re, os, numpy as np

FILLERS = {"um", "uh", "er", "erm", "hmm"}
PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their", "this", "that"}

def linguistic(transcripts):
    """Lexical richness, sentence length, pronoun ratio, fillers, repetitions. Pure python (no spaCy needed)."""
    out = []
    for t in transcripts:
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        w = re.findall(r"[a-z']+", t.lower())
        n = max(len(w), 1)
        reps = sum(1 for a, b in zip(w, w[1:]) if a == b)
        out.append([
            len(set(w)) / n,                        # type-token ratio
            n / max(len(sents), 1),                 # mean sentence length
            sum(x in PRONOUNS for x in w) / n,      # pronoun ratio
            sum(x in FILLERS for x in w) / n,       # filler ratio
            reps / n,                               # immediate repetitions
            n,                                      # total words
        ])
    names = ["ttr", "mean_sent_len", "pronoun_ratio", "filler_ratio", "repetition_ratio", "n_words"]
    return np.array(out, dtype=np.float32), names

def acoustic(audio_paths):
    """openSMILE eGeMAPS functionals + librosa pause/speech-rate proxies."""
    import opensmile, librosa
    smile = opensmile.Smile(opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.Functionals)
    rows = []
    for p in audio_paths:
        f = smile.process_file(p).iloc[0].values
        y, sr = librosa.load(p, sr=16000)
        iv = librosa.effects.split(y, top_db=30)
        speech = sum(e - s for s, e in iv) / sr
        total = len(y) / sr
        pause = [(iv[i + 1][0] - iv[i][1]) / sr for i in range(len(iv) - 1)]
        extra = [1 - speech / total, len(pause) / total, float(np.mean(pause)) if pause else 0.0]
        rows.append(np.concatenate([f, extra]))
    names = list(smile.feature_names) + ["pause_ratio", "pause_rate", "mean_pause_len"]
    return np.array(rows, dtype=np.float32), names

def wavlm(audio_paths, model="microsoft/wavlm-base-plus"):
    import torch, librosa
    from transformers import AutoFeatureExtractor, WavLMModel
    fe, m = AutoFeatureExtractor.from_pretrained(model), WavLMModel.from_pretrained(model).eval()
    out = []
    for p in audio_paths:
        y, _ = librosa.load(p, sr=16000)
        with torch.no_grad():
            h = m(**fe(y, sampling_rate=16000, return_tensors="pt")).last_hidden_state
        out.append(h.mean(1).squeeze(0).numpy())
    return np.array(out, dtype=np.float32), [f"wavlm_{i}" for i in range(out[0].shape[0])]

def bert(transcripts, model="bert-base-uncased"):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok, m = AutoTokenizer.from_pretrained(model), AutoModel.from_pretrained(model).eval()
    out = []
    for t in transcripts:
        with torch.no_grad():
            h = m(**tok(t, return_tensors="pt", truncation=True, max_length=512)).last_hidden_state
        out.append(h[:, 0].squeeze(0).numpy())
    return np.array(out, dtype=np.float32), [f"bert_{i}" for i in range(out[0].shape[0])]

def extract_all(meta, feature_dir):
    """Extract once, cache to .npy; returns feats, names dicts."""
    os.makedirs(feature_dir, exist_ok=True)
    jobs = {"acoustic": (acoustic, "audio_path"), "linguistic": (linguistic, "transcript"),
            "wavlm": (wavlm, "audio_path"), "bert": (bert, "transcript")}
    feats, names = {}, {}
    for k, (fn, col) in jobs.items():
        fp = f"{feature_dir}/{k}.npy"
        if os.path.exists(fp):
            feats[k] = np.load(fp); names[k] = list(np.load(f"{feature_dir}/{k}_names.npy", allow_pickle=True))
        else:
            feats[k], names[k] = fn(meta[col].tolist())
            np.save(fp, feats[k]); np.save(f"{feature_dir}/{k}_names.npy", np.array(names[k], dtype=object))
    return feats, names
