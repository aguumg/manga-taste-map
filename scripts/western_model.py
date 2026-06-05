"""
western_model.py — Western feature matrix + similarity model (NO heavy classifier).

Decides representation HONESTLY based on concept richness:
  - If thematic concepts are rich enough (median >= 3 per volume), build a binary
    concept feature matrix.
  - Otherwise FALL BACK to a TF-IDF vector over the volume descriptions
    (deck + desc), and report that we did so.

With only ~6 user labels we do NOT fit a supervised classifier. Primary signals:
  - cosine similarity of every volume to a chosen anchor (default = Crossed)
  - cosine similarity to the loved-Western centroid

Also runs a 2D PCA on the chosen feature matrix for the Western scatter.

Inputs:
  data/western_corpus.json
Outputs:
  data/western_artifacts.json   (features sparse, pca coords, richness report,
                                  representation used, anchor/centroid cosines,
                                  user seed)

Usage: python western_model.py
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = DATA / "western_corpus.json"
OUT = DATA / "western_artifacts.json"

# The reader's real Western verdicts -> overall. Resolved to ids by name+publisher.
USER_VERDICTS = {
    "loved": [("Crossed", "Avatar", 2008), ("Sara", "TKO", None)],
    "liked": [("Y The Last Man", "DC", 2002), ("Saga", "Image", 2012)],
    "meh":   [("Crecy", "Avatar", None), ("Locke & Key", "IDW", 2008)],
}

CONCEPT_MEDIAN_THRESHOLD = 3   # below this -> use description fallback


def richness_report(corpus):
    counts = [len(v.get("concepts_thematic", [])) for v in corpus]
    counts = np.array(counts)
    all_concepts = {}
    for v in corpus:
        for c in v.get("concepts_thematic", []):
            all_concepts[c] = all_concepts.get(c, 0) + 1
    top30 = sorted(all_concepts.items(), key=lambda kv: -kv[1])[:30]
    rep = {
        "n_volumes": len(corpus),
        "avg_concepts": float(counts.mean()) if len(counts) else 0.0,
        "median_concepts": float(np.median(counts)) if len(counts) else 0.0,
        "pct_ge3": float((counts >= 3).mean()) if len(counts) else 0.0,
        "pct_ge1": float((counts >= 1).mean()) if len(counts) else 0.0,
        "distinct_concepts": len(all_concepts),
        "top30": top30,
        # also raw (unfiltered) concept stats for comparison
        "avg_concepts_raw": float(np.mean([len(v.get("concepts", [])) for v in corpus])) if corpus else 0.0,
    }
    return rep


def build_concept_matrix(corpus):
    vocab = sorted({c for v in corpus for c in v.get("concepts_thematic", [])})
    idx = {c: i for i, c in enumerate(vocab)}
    X = np.zeros((len(corpus), len(vocab)), dtype=float)
    for r, v in enumerate(corpus):
        for c in v.get("concepts_thematic", []):
            X[r, idx[c]] = 1.0
    return X, vocab


def build_desc_matrix(corpus):
    texts = [((v.get("name") or "") + ". " + (v.get("deck") or "") + " "
              + (v.get("desc") or "")) for v in corpus]
    vec = TfidfVectorizer(max_features=600, stop_words="english",
                          ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    M = vec.fit_transform(texts)
    return M.toarray(), list(vec.get_feature_names_out())


def resolve_seed(corpus):
    """Map USER_VERDICTS to corpus volume ids by name+publisher+year."""
    seed = {}
    report = []
    for verdict, items in USER_VERDICTS.items():
        for name, pub, year in items:
            cand = None
            nlow = name.lower().replace("the ", "").strip()
            for v in corpus:
                vn = (v["name"] or "").lower().replace("the ", "").strip()
                if nlow in vn or vn in nlow:
                    if pub and pub.lower() not in (v["publisher"] or "").lower():
                        continue
                    if year and v.get("start_year") and abs(int(v["start_year"]) - year) > 2:
                        continue
                    cand = v
                    break
            if cand:
                seed[str(cand["id"])] = {"read": True, "overall": verdict}
                report.append((name, cand["id"], cand["name"], cand["publisher"], verdict))
            else:
                report.append((name, None, None, None, verdict))
    return seed, report


def cosine(A, b):
    bn = b / (np.linalg.norm(b) + 1e-12)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return An @ bn


def main():
    corpus = json.loads(CORPUS.read_text())
    print(f"Western corpus: {len(corpus)} volumes")

    rep = richness_report(corpus)
    print("\n=== RICHNESS REPORT (thematic concepts, cover/variant noise removed) ===")
    print(f"  avg concepts/vol:    {rep['avg_concepts']:.2f}")
    print(f"  median concepts/vol: {rep['median_concepts']:.1f}")
    print(f"  % vols with >=3:     {rep['pct_ge3']*100:.1f}%")
    print(f"  % vols with >=1:     {rep['pct_ge1']*100:.1f}%")
    print(f"  distinct concepts:   {rep['distinct_concepts']}")
    print(f"  (raw incl. noise avg/vol: {rep['avg_concepts_raw']:.2f})")
    print("  top concepts:", [f"{n}({c})" for n, c in rep["top30"][:15]])

    # choose representation
    use_concepts = rep["median_concepts"] >= CONCEPT_MEDIAN_THRESHOLD
    if use_concepts:
        X, vocab = build_concept_matrix(corpus)
        representation = "concepts (binary)"
    else:
        X, vocab = build_desc_matrix(corpus)
        representation = "description TF-IDF (concepts too sparse)"
    print(f"\nRepresentation used: {representation}  (features={X.shape[1]})")

    ids = [v["id"] for v in corpus]
    id_to_row = {v["id"]: r for r, v in enumerate(corpus)}

    seed, seed_report = resolve_seed(corpus)
    print("\nUser Western verdicts seeded:")
    for name, vid, vname, pub, verdict in seed_report:
        tag = f"id {vid} '{vname}' ({pub})" if vid else "NOT IN CORPUS"
        print(f"  [{verdict:6s}] {name:22s} -> {tag}")

    # anchor = Crossed if present, else first loved
    loved_ids = [int(k) for k, v in seed.items() if v["overall"] == "loved"]
    crossed = next((v["id"] for v in corpus
                    if "crossed" in (v["name"] or "").lower()
                    and "avatar" in (v["publisher"] or "").lower()), None)
    anchor_id = crossed or (loved_ids[0] if loved_ids else ids[0])

    anchor_cos = cosine(X, X[id_to_row[anchor_id]]).tolist()
    if loved_ids:
        loved_rows = [id_to_row[i] for i in loved_ids if i in id_to_row]
        centroid = X[loved_rows].mean(0)
        centroid_cos = cosine(X, centroid).tolist()
    else:
        centroid_cos = anchor_cos

    # PCA 2D for scatter
    n_comp = 2 if X.shape[1] >= 2 and X.shape[0] >= 3 else 1
    pca = PCA(n_components=n_comp, random_state=0).fit(X)
    coords = pca.transform(X)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])

    # sparse features for embedding
    sparse = []
    for r in range(X.shape[0]):
        nz = np.nonzero(X[r])[0]
        sparse.append([[int(j), round(float(X[r, j]), 4)] for j in nz])

    artifacts = {
        "representation": representation,
        "used_concepts": use_concepts,
        "features": vocab,
        "nFeat": len(vocab),
        "richness": rep,
        "anchorId": anchor_id,
        "seed": seed,
        "seed_report": [list(x) for x in seed_report],
        "volumes": [{
            "id": v["id"], "name": v["name"], "publisher": v["publisher"],
            "year": v.get("start_year"),
            "desc": v.get("desc") or v.get("deck") or "",
            "concepts": v.get("concepts_thematic", []),
            "issues": v.get("count_of_issues"),
            "x": round(float(coords[r, 0]), 4), "y": round(float(coords[r, 1]), 4),
            "anchorCos": round(anchor_cos[r], 4),
            "centroidCos": round(centroid_cos[r], 4),
            "vec": sparse[r],
        } for r, v in enumerate(corpus)],
    }
    OUT.write_text(json.dumps(artifacts, ensure_ascii=False))
    print(f"\nSaved -> {OUT}")
    print(f"  representation={representation}, anchor=id {anchor_id}, "
          f"labels seeded={len(seed)}")


if __name__ == "__main__":
    main()
