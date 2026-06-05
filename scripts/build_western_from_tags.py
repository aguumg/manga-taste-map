#!/usr/bin/env python3
"""Build Western 254-dim vectors from LLM-assigned AniList tags (the working approach).

Replaces the failed description-embedding path. Each Western title's tags come from
data/western_llm_tags.json ({title:{tags:[[name,w],...], est_verdict, note}}), mapped
DIRECTLY onto the Eastern feature basis -> shares the same PCA map + cosine model.

Outputs western_vectors.npz + western_projected_tags.json (+ a Gate-2 validation print).
"""
import json, sys, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from model import build_feature_matrix, BERSERK_ID

MODE0_IDS = {30002: "Berserk", 30642: "Vinland Saga", 30013: "Claymore",
             30021: "Blade of the Immortal", 53390: "Attack on Titan",
             105398: "Hell's Paradise"}

def main():
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    Xdf = build_feature_matrix(corpus)
    feat_cols = list(Xdf.columns)
    col_idx = {c: i for i, c in enumerate(feat_cols)}
    Xe = Xdf.values.astype(np.float64)
    east_ids = list(Xdf.index)
    id2title = {int(r["id"]): (r["english"] or r["romaji"]) for _, r in corpus.iterrows()}

    llm = json.loads((DATA / "western_llm_tags.json").read_text())
    vocab = set(json.loads((DATA / "anilist_tag_vocab.json").read_text()).keys())
    # optional descriptions for display
    desc_by_name = {}
    p = DATA / "western_corpus.json"
    if p.exists():
        for v in json.loads(p.read_text()).values():
            desc_by_name[v["name"].lower()] = v.get("description", "")

    titles, Xw_rows, meta = [], [], []
    dropped_tags = set()
    for title, rec in llm.items():
        vec = np.zeros(len(feat_cols))
        used = []
        for entry in rec.get("tags", []):
            name, w = (entry[0], float(entry[1])) if isinstance(entry, list) else (entry, 0.7)
            if name in vocab and name in col_idx:
                vec[col_idx[name]] = max(vec[col_idx[name]], min(1.0, w))
                used.append((name, round(min(1.0, w), 2)))
            else:
                dropped_tags.add(name)
        if not used:
            continue
        titles.append(title)
        Xw_rows.append(vec)
        # match a description for display
        d = ""
        for k, dv in desc_by_name.items():
            if title.lower().split(":")[0] in k or k.split(":")[0] in title.lower():
                d = dv; break
        meta.append({"title": title, "known": rec.get("known", True),
                     "est_verdict": rec.get("est_verdict"), "tags": used,
                     "note": rec.get("note", ""), "description": (d or rec.get("note", ""))[:600]})
    Xw = np.array(Xw_rows)
    print(f"Built {len(titles)} Western vectors. Dropped {len(dropped_tags)} invalid tag names"
          + (f": {sorted(dropped_tags)[:8]}..." if dropped_tags else ""))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0).fit(Xe)
    w_xy = pca.transform(Xw)

    def norm(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    cos = norm(Xw) @ norm(Xe).T
    brow = east_ids.index(BERSERK_ID) if BERSERK_ID in east_ids else None
    m0 = [east_ids.index(i) for i in MODE0_IDS if i in east_ids]

    out = {}
    for r, t in enumerate(titles):
        order = np.argsort(-cos[r])[:8]
        nn = [(id2title.get(int(east_ids[k]), str(east_ids[k])), round(float(cos[r, k]), 3)) for k in order]
        out[t] = {**meta[r],
                  "coords": [round(float(w_xy[r, 0]), 4), round(float(w_xy[r, 1]), 4)],
                  "nearest_eastern": nn,
                  "berserk_cos": round(float(cos[r, brow]), 3) if brow is not None else None,
                  "mode0_cos": round(float(np.mean([cos[r, x] for x in m0])), 3) if m0 else None}

    np.savez(DATA / "western_vectors.npz", titles=np.array(titles, dtype=object),
             X=Xw, coords=w_xy)
    (DATA / "western_projected_tags.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n========= GATE 2 (LLM-tag) VALIDATION =========")
    corpus_m0 = float(np.mean([cos[:, x] for x in m0])) if m0 else 0
    print(f"corpus-wide mean cos to Mode-0: {corpus_m0:.3f}  (Western titles should beat this)")
    for sub in ["Crossed", "Sara", "Y: The Last", "Saga", "Locke", "Crécy", "Scalped", "Punisher"]:
        for t, v in out.items():
            if sub.lower() in t.lower():
                flag = "✓" if (v["mode0_cos"] or 0) > corpus_m0 else "✗ below-avg"
                print(f"\n--- {t} [{flag}] verdict={v['est_verdict']} ---")
                print("  tags:", [x[0] for x in v["tags"][:10]])
                print(f"  Berserk cos={v['berserk_cos']}  Mode-0 cos={v['mode0_cos']}")
                print("  nearest Eastern:", [f"{n}({c})" for n, c in v["nearest_eastern"][:6]])
                break
    # aggregate: do loved/liked Western beat meh on Mode-0 proximity?
    byv = {}
    for v in out.values():
        if v["est_verdict"]:
            byv.setdefault(v["est_verdict"], []).append(v["mode0_cos"] or 0)
    print("\nMean Mode-0 cos by user verdict:", {k: round(np.mean(x), 3) for k, x in byv.items()})
    print("saved -> data/western_vectors.npz + western_projected_tags.json")

if __name__ == "__main__":
    main()
