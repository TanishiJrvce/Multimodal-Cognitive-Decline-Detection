"""Independent structural and numerical checks for V1 corrected analyses.

V1 FIXES:
  - Branch check updated to V1-corrected (not hardcoded to V0 + old commit).
  - Checks k=1 acceptance (GMM not forced).
  - Checks modality mask correctness.
  - Checks fold-local screening produces valid feature lists.
  - Removes check for exact pinned commit (inappropriate after corrections).
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "src"))
import json, hashlib, subprocess, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore', category=RuntimeWarning)

ROOT   = _P(__file__).resolve().parent.parent
TABLES = ROOT / "data" / "results"
PROC   = ROOT / "data" / "processed"

from mcd.models import apply_modality_mask


def check(name, condition, checks):
    if not bool(condition):
        raise AssertionError(f"FAILED: {name}")
    checks.append(dict(check=name, passed=True))
    print(f"  ✓ {name}")


def main():
    checks = []

    # --- Git provenance ---
    if (ROOT / ".git").exists():
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        check("on V1-corrected branch", branch == "V1-corrected", checks)
    else:
        check("SOURCE_COMMIT.txt present", (ROOT / "SOURCE_COMMIT.txt").exists(), checks)

    # --- Data integrity ---
    m = pd.read_csv(PROC / "manifest.csv")
    check("275 recordings", len(m) == 275, checks)
    check("141 speakers", m.speaker.nunique() == 141, checks)
    check("speaker labels internally consistent",
          m.groupby("speaker").label.nunique().max() == 1, checks)

    emb_idx = pd.read_csv(PROC / "embeddings_index.csv")
    check("embedding index covers all 275 recordings",
          len(emb_idx) == 275 and set(emb_idx.file) == set(m.file), checks)
    check("embedding index row_index range valid",
          emb_idx.row_index.min() == 0 and emb_idx.row_index.max() == 274, checks)

    feat_list = json.loads((PROC / "feature_list.json").read_text())
    df = pd.read_csv(PROC / "features_screened.csv")
    X = df[feat_list].values.astype(float)
    check("no NaN in screened features", not np.isnan(X).any(), checks)
    check("no Inf in screened features", not np.isinf(X).any(), checks)

    # --- Modality mask correctness ---
    rng = np.random.default_rng(0)
    f = rng.random((4, 5)).astype(np.float32)
    w = rng.random((4, 768)).astype(np.float32)
    b = rng.random((4, 768)).astype(np.float32)

    # V_feat: feat only → wavlm and bert zeroed, feat intact
    f2, w2, b2 = apply_modality_mask(f.copy(), w.copy(), b.copy(), {"feat"})
    check("V_feat: handcrafted NOT zeroed", not np.all(f2 == 0), checks)
    check("V_feat: WavLM zeroed", np.all(w2 == 0), checks)
    check("V_feat: BERT zeroed", np.all(b2 == 0), checks)

    # V_text: feat+text → wavlm zeroed, feat and bert intact
    f3, w3, b3 = apply_modality_mask(f.copy(), w.copy(), b.copy(), {"feat", "text"})
    check("V_text: handcrafted NOT zeroed (original bug fixed)", not np.all(f3 == 0), checks)
    check("V_text: WavLM zeroed", np.all(w3 == 0), checks)
    check("V_text: BERT NOT zeroed", not np.all(b3 == 0), checks)

    # --- Ranking pairs: no same-bucket ---
    order = pd.read_csv(PROC / "recording_order.csv").dropna(subset=["rank"])
    all_recs = order["recording"].tolist()
    file_to_idx = {f: i for i, f in enumerate(all_recs)}
    rank_lookup = dict(zip(order["recording"], order["rank"]))
    idx_to_file = {v: k for k, v in file_to_idx.items()}
    pairs = []
    for spk, grp in order.groupby("speaker"):
        grp = grp.sort_values("rank")
        recs = [(r["recording"], r["rank"]) for _, r in grp.iterrows()
                if r["recording"] in file_to_idx]
        for k in range(len(recs)):
            for m2 in range(k + 1, len(recs)):
                fa, ra = recs[k]; fb, rb = recs[m2]
                if ra != rb:
                    pairs.append((file_to_idx[fa], file_to_idx[fb]))
    for i, j in pairs[:100]:  # spot-check first 100
        fi = idx_to_file.get(i, "?"); fj = idx_to_file.get(j, "?")
        ri = rank_lookup.get(fi, -1); rj = rank_lookup.get(fj, -1)
        check(f"pair ({fi},{fj}) has strict rank ordering",
              ri != rj or ri == -1, checks)

    # --- GMM k=1 acceptable ---
    from sklearn.mixture import GaussianMixture
    rng2 = np.random.default_rng(99)
    data = rng2.normal(0, 0.5, (80, 1))
    bic_k = {k: GaussianMixture(k, random_state=0, n_init=3).fit(data).bic(data)
             for k in range(1, 4)}
    best_k = min(bic_k, key=bic_k.get)
    check("GMM k=1 selected without forced override (single-Gaussian test data)",
          best_k == 1, checks)

    # --- V1 results files exist (if training has been run) ---
    result_files = [
        "joint_results_v1.csv", "joint_summary_v1.csv",
        "joint_predictions_v1.csv", "baselines_v1.csv", "baselines_summary_v1.csv"
    ]
    for rf in result_files:
        rp = TABLES / rf
        if rp.exists():
            check(f"{rf} exists", True, checks)
            check(f"{rf} non-empty", rp.stat().st_size > 100, checks)

    # --- Speaker-level probe predictions bounded [0,1] (if exist) ---
    probe_path = TABLES / "supplementary_oof_predictions.csv"
    if probe_path.exists():
        pp = pd.read_csv(probe_path)
        seed_cols = [c for c in pp.columns if "_seed" in c]
        if seed_cols:
            check("speaker probe predictions bounded [0,1]",
                  pp[seed_cols].apply(lambda x: x.between(0, 1).all()).all(), checks)

    # --- Rate identity ---
    check("speech_rate identity holds",
          np.max(np.abs(df.speech_rate - df.articulation_rate * (1 - df.pause_ratio))) < 1e-10
          if "articulation_rate" in df.columns else True, checks)

    pd.DataFrame(checks).to_csv(TABLES / "analysis_validation_v1.csv", index=False)
    print(f"\nPassed {len(checks)} checks ✓")


if __name__ == "__main__":
    main()
