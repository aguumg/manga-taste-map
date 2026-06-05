"""Inspect Eastern model artifacts: feature columns, npz contents, PCA shape."""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"

meta = json.loads((DATA / "model_artifacts.json").read_text())
fc = meta["feat_cols"]
print("n_features:", meta["n_features"], "len feat_cols:", len(fc))
tags = [c for c in fc if not c.startswith("demo::") and not c.startswith("genre::")]
demo = [c for c in fc if c.startswith("demo::")]
genre = [c for c in fc if c.startswith("genre::")]
print("plain tags:", len(tags), "| demo:", len(demo), "| genre:", len(genre))
print("first 12 tags:", tags[:12])
print("demo:", demo)
print("genre:", genre)
print("pc1_top:", meta["pc1_top"][:6])
print("pc2_top:", meta["pc2_top"][:6])

npz = np.load(DATA / "model_artifacts.npz")
print("npz keys:", list(npz.keys()))
for k in npz.keys():
    print("  ", k, npz[k].shape, npz[k].dtype)
print("berserk_id:", meta["berserk_id"], "median_score:", meta["median_score"])
print("modes keys:", list(meta["modes"].keys()))
if "2" in meta["modes"]:
    for lbl, members in meta["modes"]["2"]["members"].items():
        print(f"  mode2[{lbl}] ({len(members)}):", members[:12])
