#!/usr/bin/env python3
"""Layouts for the app. Emits:
  data/layout.json         unified raw-PCA coords  { "e<id>":[x,y], "w<title>":[x,y] }
  data/layout_idf.json     unified IDF-weighted PCA coords (same keys)
  data/idf_weights.json    { tagName: idf } for client-side anchor-cosine reweighting
  data/layout_western.json Western-only PCA coords { "w<title>":[x,y] } (own axes -> spreads)
  data/eastern_drop.json   romance-no-action eastern ids removed from the map
"""
import json, sys, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from model import build_feature_matrix, BERSERK_ID

ROMANCE = {"Romance", "Mahou Shoujo"}
ACTIONISH = {"Action", "Adventure", "Horror", "Thriller", "Mystery",
             "Supernatural", "Psychological", "Sci-Fi", "Fantasy"}


def main():
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    Xdf = build_feature_matrix(corpus)
    feat = list(Xdf.columns)
    Xe = Xdf.values.astype(np.float64)
    ids = list(Xdf.index)
    id2t = {int(r["id"]): (r["english"] or r["romaji"]) for _, r in corpus.iterrows()}
    genres = {int(r["id"]): set(r["genres"]) for _, r in corpus.iterrows()}

    drop = [int(i) for i in ids if (genres.get(int(i), set()) & ROMANCE)
            and not (genres.get(int(i), set()) & ACTIONISH)]
    drop_set = set(drop)
    keep = [k for k, i in enumerate(ids) if int(i) not in drop_set]
    print(f"eastern {len(ids)} -> kept {len(keep)} (dropped {len(drop)} romance)")

    wz = np.load(DATA / "western_vectors.npz", allow_pickle=True)
    Xw = wz["X"].astype(np.float64); wt = [str(t) for t in wz["titles"]]
    print(f"western {len(wt)}")

    Xe_keep = Xe[keep]
    combined = np.vstack([Xe_keep, Xw])

    from sklearn.decomposition import PCA

    def fit_layout(M):
        p = PCA(2, random_state=0).fit(M)
        c = p.transform(M)
        return c[:len(keep)], c[len(keep):]

    # 1. raw unified
    ce, cw = fit_layout(combined)
    # 2. IDF unified
    df = (combined > 0).sum(0); N = combined.shape[0]
    idf = np.log((N + 1) / (df + 1)) + 1.0
    ce_i, cw_i = fit_layout(combined * idf)
    # 3. western-only (own axes)
    pw = PCA(2, random_state=0).fit(Xw)
    cwo = pw.transform(Xw)

    def write(fname, ce_arr, cw_arr, western_only=False):
        d = {}
        if not western_only:
            for k, gi in enumerate(keep):
                d[f"e{int(ids[gi])}"] = [round(float(ce_arr[k, 0]), 4), round(float(ce_arr[k, 1]), 4)]
        for k, t in enumerate(cw_arr):
            d[f"w{wt[k]}"] = [round(float(cw_arr[k, 0]), 4), round(float(cw_arr[k, 1]), 4)]
        (DATA / fname).write_text(json.dumps(d, ensure_ascii=False))

    write("layout.json", ce, cw)
    write("layout_idf.json", ce_i, cw_i)
    write("layout_western.json", None, cwo, western_only=True)
    (DATA / "idf_weights.json").write_text(json.dumps({feat[i]: round(float(idf[i]), 4)
                                                       for i in range(len(feat))}, ensure_ascii=False))
    (DATA / "eastern_drop.json").write_text(json.dumps(sorted(drop_set)))

    # diagnostics
    def ratio(ce_a, cw_a): return cw_a.std(0).mean() / ce_a.std(0).mean()
    print(f"\nWestern spread on unified:  raw={ratio(ce,cw):.2f}  IDF={ratio(ce_i,cw_i):.2f}")
    print(f"Western-only spread (std):  ({cwo.std(0)[0]:.2f},{cwo.std(0)[1]:.2f}) — full-scale by construction")
    # does western-only show genre structure? show extremes of each axis
    for ax, lbl in [(0, "PC1"), (1, "PC2")]:
        o = np.argsort(cwo[:, ax])
        lo = [wt[i] for i in o[:4]]; hi = [wt[i] for i in o[-4:]]
        print(f"  western-only {lbl}:  low={lo}  ...  high={hi}")
    print("\nsaved 4 layout files + idf_weights + eastern_drop")


if __name__ == "__main__":
    main()
