"""Speaker-independent CV: baselines, ablation over modality subsets, SHAP, robustness slices.
Usage: python -m src.run_cv [--config configs/default.yaml]"""
import argparse, itertools, json, yaml, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .data import make_mock, load_manifest
from .features import extract_all
from .model import FusionMLP

def train_fusion(Xtr, ytr, mtr, cfg, seed):
    torch.manual_seed(seed)
    dims = {k: v.shape[1] for k, v in Xtr.items()}
    net = FusionMLP(dims, int(max(ytr)) + 1, cfg["hidden"], cfg["dropout"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    xs = {k: torch.tensor(v) for k, v in Xtr.items()}
    y, m = torch.tensor(ytr), torch.tensor(((mtr - 15) / 10).astype(np.float32))
    for _ in range(cfg["epochs"]):
        net.train(); opt.zero_grad()
        lo, mm = net(xs)
        (nn.functional.cross_entropy(lo, y) + 0.5 * nn.functional.mse_loss(mm, m)).backward(); opt.step()
    return net.eval()

def predict(net, X):
    with torch.no_grad():
        lo, mm = net({k: torch.tensor(v) for k, v in X.items()})
    return lo.argmax(1).numpy(), mm.numpy() * 10 + 15

def cv_eval(meta, feats, mods, cfg, kind="fusion"):
    y, mmse, groups = meta.label.values, meta.mmse.values.astype(np.float32), meta.speaker.values
    skf = StratifiedGroupKFold(cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    pred, mpred = np.zeros(len(y), int), np.zeros(len(y))
    for tr, te in skf.split(np.zeros(len(y)), y, groups):
        sc = {k: StandardScaler().fit(feats[k][tr]) for k in mods}   # fit scaler on train fold only
        Xtr = {k: sc[k].transform(feats[k][tr]).astype(np.float32) for k in mods}
        Xte = {k: sc[k].transform(feats[k][te]).astype(np.float32) for k in mods}
        if kind == "fusion":
            net = train_fusion(Xtr, y[tr], mmse[tr], cfg, cfg["seed"])
            pred[te], mpred[te] = predict(net, Xte)
        else:
            clf = LogisticRegression(max_iter=2000) if kind == "logreg" else RandomForestClassifier(200, random_state=0)
            clf.fit(np.hstack(list(Xtr.values())), y[tr]); pred[te] = clf.predict(np.hstack(list(Xte.values())))
    r = {"acc": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}
    if kind == "fusion": r["mmse_pearson_r"] = pearsonr(mmse, mpred)[0]
    return r, pred

def shap_report(meta, feats, names, mods, cfg, out):
    import shap
    y, mmse = meta.label.values, meta.mmse.values.astype(np.float32)
    sc = {k: StandardScaler().fit(feats[k]) for k in mods}
    X = {k: sc[k].transform(feats[k]).astype(np.float32) for k in mods}
    net = train_fusion(X, y, mmse, cfg, cfg["seed"])
    cols = [n for k in mods for n in names[k]]; sizes = [X[k].shape[1] for k in mods]
    flat = np.hstack([X[k] for k in mods])
    def f(a):
        parts = np.split(a.astype(np.float32), np.cumsum(sizes)[:-1], axis=1)
        with torch.no_grad():
            return torch.softmax(net({k: torch.tensor(p) for k, p in zip(mods, parts)})[0], 1).numpy()
    ex = shap.KernelExplainer(f, shap.sample(flat, 20, random_state=0))
    sv = np.array(ex.shap_values(flat[:30], nsamples=100))      # (classes, n, feats) or (n, feats, classes)
    imp = np.abs(sv).mean(axis=(0, 1)) if sv.shape[0] == f(flat[:1]).shape[1] else np.abs(sv).mean(axis=(0, 2))
    top = pd.Series(imp, index=cols).sort_values(ascending=False).head(15)
    top.to_csv(f"{out}/shap_top_features.csv"); return top

def robustness(meta, pred):
    d = meta.assign(correct=(pred == meta.label.values), age_band=pd.cut(meta.age, [0, 65, 75, 120], labels=["<65", "65-75", ">75"]))
    return {"by_gender": d.groupby("gender").correct.mean().to_dict(),
            "by_age": d.groupby("age_band", observed=True).correct.mean().to_dict()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    if cfg["mock"]:
        meta, feats, names = make_mock()
    else:
        meta = load_manifest(cfg["manifest"]); feats, names = extract_all(meta, cfg["feature_dir"])
    all_mods = list(feats); res = {}
    for kind in ("logreg", "rf"):                                   # baselines on concatenated features
        res[f"baseline_{kind}(all)"] = cv_eval(meta, feats, all_mods, cfg, kind)[0]
    for r in range(1, len(all_mods) + 1):                           # ablation: every modality subset
        for mods in itertools.combinations(all_mods, r):
            res["fusion[" + "+".join(mods) + "]"] = cv_eval(meta, feats, list(mods), cfg)[0]
    _, pred = cv_eval(meta, feats, all_mods, cfg)
    res["robustness_full_model"] = robustness(meta, pred)
    df = pd.DataFrame({k: v for k, v in res.items() if "acc" in v}).T.round(3)
    print(df.to_string()); print(json.dumps(res["robustness_full_model"], indent=1, default=str))
    df.to_csv("reports/results.csv")
    print("\nTop SHAP features:\n", shap_report(meta, feats, names, all_mods, cfg, "reports"))

if __name__ == "__main__":
    main()
