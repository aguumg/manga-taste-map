#!/usr/bin/env python3
"""Project Western titles into the Eastern AniList tag-space via description embeddings.

Gate-1 pivot: Comicvine concepts are junk, so each Western title is represented by
embedding its plot description and scoring it against every AniList tag's definition.
Top-K tags -> a sparse 254-dim vector in the SAME basis as the Eastern feature matrix,
so Western titles share the Eastern PCA map + cosine model.

Outputs:
  data/western_vectors.npz          ids, X (n,254), coords (n,2)
  data/western_projected_tags.json  per-title: name, top tags, nearest Eastern, coords
"""
import json, sys, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from model import build_feature_matrix, BERSERK_ID  # same matrix builder as build_html

TOPK = 18
SCALE_LO, SCALE_HI = 0.30, 0.95
MODE0_IDS = {30002: "Berserk", 30642: "Vinland Saga", 30013: "Claymore",
             30021: "Blade of the Immortal", 53390: "Attack on Titan",
             105398: "Hell's Paradise"}  # JP seinen-dark cluster (verify ids exist)

def main():
    # --- Eastern matrix in the canonical basis ---
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    Xdf = build_feature_matrix(corpus)
    feat_cols = list(Xdf.columns)
    Xe = Xdf.values.astype(np.float64)
    east_ids = list(Xdf.index)
    id2title = {}
    for _, r in corpus.iterrows():
        id2title[int(r["id"])] = (r["english"] or r["romaji"])

    tag_cols = [(i, c) for i, c in enumerate(feat_cols)
                if not (str(c).startswith("demo::") or str(c).startswith("genre::"))]
    tag_idx = [i for i, _ in tag_cols]
    tag_names = [c for _, c in tag_cols]
    print(f"Eastern: {Xe.shape[0]} titles x {Xe.shape[1]} feats ({len(tag_names)} tags)")

    # --- tag definition texts ---
    tagdesc = {}
    p = DATA / "tag_descriptions.json"
    if p.exists():
        raw = json.loads(p.read_text())
        tagdesc = raw if isinstance(raw, dict) else {d["name"]: d.get("description", "") for d in raw}
    tag_texts = [f"{n}. {tagdesc.get(n, '')}".strip() for n in tag_names]

    # --- western corpus ---
    west = json.loads((DATA / "western_corpus.json").read_text())
    west = {k: v for k, v in west.items() if len(v.get("description", "")) >= 40}
    w_ids = list(west.keys())
    w_desc = [west[k]["description"] for k in w_ids]
    print(f"Western: {len(w_ids)} titles with usable descriptions")

    # --- embed ---
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("embedding tags + western descriptions...")
    tag_emb = model.encode(tag_texts, normalize_embeddings=True, show_progress_bar=False)
    w_emb = model.encode(w_desc, normalize_embeddings=True, show_progress_bar=False)
    sim = w_emb @ tag_emb.T  # (n_west, n_tags) cosine

    # --- project: top-K tags -> sparse 254 vector ---
    Xw = np.zeros((len(w_ids), len(feat_cols)), dtype=np.float64)
    proj_tags = {}
    for r in range(len(w_ids)):
        s = sim[r]
        top = np.argsort(-s)[:TOPK]
        lo, hi = s[top].min(), s[top].max()
        rng = (hi - lo) or 1.0
        kept = []
        for j in top:
            val = SCALE_LO + SCALE_HI * (s[j] - lo) / rng  # high tag ~0.95
            Xw[r, tag_idx[j]] = val
            kept.append((tag_names[j], round(float(val), 3), round(float(s[j]), 3)))
        proj_tags[w_ids[r]] = kept

    # --- same PCA as build_html (deterministic) ---
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0).fit(Xe)
    w_xy = pca.transform(Xw)

    # --- nearest Eastern neighbors (cosine on 254) ---
    def norm(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    Xen, Xwn = norm(Xe), norm(Xw)
    cos = Xwn @ Xen.T  # (n_west, n_east)
    berserk_row = east_ids.index(BERSERK_ID) if BERSERK_ID in east_ids else None
    mode0_rows = [east_ids.index(i) for i in MODE0_IDS if i in east_ids]

    out = {}
    for r, wid in enumerate(w_ids):
        order = np.argsort(-cos[r])[:8]
        nn = [(id2title.get(int(east_ids[k]), str(east_ids[k])), round(float(cos[r, k]), 3)) for k in order]
        out[wid] = {
            "id": int(west[wid]["id"]), "name": west[wid]["name"],
            "year": west[wid].get("year"), "publisher": west[wid].get("publisher"),
            "description": west[wid]["description"][:600],
            "top_tags": proj_tags[wid],
            "nearest_eastern": nn,
            "coords": [round(float(w_xy[r, 0]), 4), round(float(w_xy[r, 1]), 4)],
            "berserk_cos": round(float(cos[r, berserk_row]), 3) if berserk_row is not None else None,
            "mode0_cos": round(float(np.mean([cos[r, m] for m in mode0_rows])), 3) if mode0_rows else None,
        }

    np.savez(DATA / "western_vectors.npz",
             ids=np.array([int(west[k]["id"]) for k in w_ids]),
             X=Xw, coords=w_xy)
    (DATA / "western_projected_tags.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

    # --- VALIDATION REPORT (Gate 2) ---
    print("\n================ GATE 2 VALIDATION ================")
    overall_mode0 = np.mean([cos[:, m] for m in mode0_rows]) if mode0_rows else 0
    print(f"corpus-wide mean cos to Mode-0 cluster: {overall_mode0:.3f}")
    def show(name_sub):
        for wid, v in out.items():
            if name_sub.lower() in v["name"].lower():
                print(f"\n--- {v['name']} ({v['publisher']}, {v['year']}) ---")
                print("  top projected tags:", [t[0] for t in v["top_tags"][:10]])
                print(f"  cos to Berserk: {v['berserk_cos']} | mean cos to Mode-0: {v['mode0_cos']}")
                print("  nearest Eastern:", [f"{t}({c})" for t, c in v["nearest_eastern"][:6]])
                return
        print(f"  (not found: {name_sub})")
    for t in ["Crossed", "Sara", "Y: The Last", "Saga", "Locke", "Crecy"]:
        show(t)
    print("\nsaved -> data/western_vectors.npz + western_projected_tags.json")

if __name__ == "__main__":
    main()
