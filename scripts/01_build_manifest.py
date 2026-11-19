"""Build manifest.csv and recording_order.csv from DementiaNet metadata + split files.

Ordering policy (conservative): only dementia-group recordings whose filename suffix is one of the
documented time buckets (0 = after symptoms, 5 = 0-5y before, 10 = 5-10y before, 15 = 10-15y before)
get a rank. A trailing second number (e.g. _5_2) is a clip index in the same bucket (equal rank, forms no pair).
Other suffixes and all comparison-group suffixes are left UNRANKED. Ranks are marked order_basis='suffix_bucket_unverified' until the
suffix->bucket mapping is verified against the metadata sheet.
"""
import re
from pathlib import Path
import pandas as pd

raw, out = Path("data/raw"), Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(raw / "train_dm.csv", sep="\t").assign(split="train")
va = pd.read_csv(raw / "valid_dm.csv", sep="\t").assign(split="valid")
m = pd.concat([tr, va], ignore_index=True)
m["speaker"] = m.path.str.split("/").str[-2]
bc = m.file.str.extract(r"_(\d+)(?:_(\d+))?$")
m["suffix"] = bc[0].astype(float)  # bucket (dementia: 0/5/10/15)
m["clip"] = bc[1].astype(float)    # optional clip index within a bucket (e.g. _5_2); equal rank
m["label"] = (m.label == "dementia").astype(int)  # 1 = dementia-associated, 0 = comparison
assert not set(m[m.split == "train"].speaker) & set(m[m.split == "valid"].speaker)

meta = pd.read_excel(raw / "metadata.xlsx", sheet_name="dementia")
meta2 = pd.read_excel(raw / "metadata.xlsx", sheet_name="no dementia").rename(
    columns={"birthdate": "birth", "deathdate": "death"})
meta = pd.concat([meta.assign(group="dementia"), meta2.assign(group="comparison")], ignore_index=True)
norm = lambda s: re.sub(r"[^a-z]", "", str(s).lower())
meta["key"] = meta["name"].map(norm)
m["key"] = m.speaker.map(norm)
keep = ["key", "dementia type", "birth", "death", "first symptoms", "gender", "ethnicity", "language"]
m = m.merge(meta[keep].drop_duplicates("key"), on="key", how="left")
print("metadata match rate:", m["gender"].notna().mean().round(3))
print("unmatched speakers:", sorted(m[m.gender.isna()].speaker.unique())[:20])
m.drop(columns="key").to_csv(out / "manifest.csv", index=False)

rank = {15: 1, 10: 2, 5: 3, 0: 4}
o = m[(m.label == 1) & m.suffix.isin(rank)].copy()
o["rank"] = o.suffix.map(rank)
o["order_basis"] = "suffix_bucket_unverified"
o = o.rename(columns={"file": "recording"})[["speaker", "recording", "rank", "order_basis", "split"]]
multi = o.groupby("speaker").size()
o = o[o.speaker.isin(multi[multi >= 2].index)]
o.to_csv(out / "recording_order.csv", index=False)
print("manifest rows:", len(m), "| speakers:", m.speaker.nunique())
print("ordered recordings:", len(o), "| speakers with >=2 ordered:", o.speaker.nunique(),
      "| by split:", o.groupby("split").speaker.nunique().to_dict())
