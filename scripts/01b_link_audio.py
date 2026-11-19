import pathlib
import pandas as pd

n = lambda s: "".join(c for c in str(s).lower() if c.isalnum())
m = pd.read_csv("data/processed/manifest.csv")
rows = []
for grp, folder in [(1, "data/raw/audio"), (0, "data/raw/audio/nodementia")]:
    for p in pathlib.Path(folder).rglob("*.wav"):
        if grp == 1 and "nodementia" in p.parts:
            continue
        rows.append((grp, n(p.stem), str(p).replace("\\", "/")))
f = pd.DataFrame(rows, columns=["label", "key", "audio_path"]).drop_duplicates(["label", "key"])
m["key"] = m.file.map(n)
m = m.merge(f, on=["label", "key"], how="left").drop(columns="key")
m.to_csv("data/processed/manifest.csv", index=False)
print("files on disk:", f.groupby("label").size().to_dict())
print("manifest recs with audio:", m.groupby("label").audio_path.apply(lambda s: f"{s.notna().sum()}/{len(s)}").to_dict())
print("missing:", list(m.file[m.audio_path.isna()])[:10])
print("on disk but not in manifest:", f[~f.key.isin(m.file.map(n))].shape[0])
