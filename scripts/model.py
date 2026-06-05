"""
model.py — Fit a taste model over AniList tag-rank features with honest p>>n CV.

Pipeline:
  1. Build feature matrix X (rows=titles, cols=tags). Value = rank/100 in [0,1],
     0 if the tag is absent. Drop tags present in <1% of corpus. Optionally add
     demographic + genre binary features. popularity/averageScore are EXCLUDED
     from the fitted features (display only) so the model can't just learn
     "famous" — the reader likes both niche and popular.
  2. Labels (n~=23): L2 logistic regression, C chosen by LEAVE-ONE-OUT CV to
     maximize LOO AUC. Report LOO-AUC and LOO accuracy (NOT training numbers).
  3. Robustness: nearest-shrunken-centroid / diagonal-LDA tag weighting
     w_tag = (mean_pos - mean_neg)/pooled_sd; score = X.w. Report its LOO-AUC.
  4. Baselines: cosine similarity to the Berserk vector and to the positive
     centroid (unsupervised).
  5. Bimodality: KMeans (k=2,k=3) on STRONG-POSITIVE feature vectors; report
     whether they split into interpretable modes; emit per-mode nearest titles.
  6. Score the whole corpus with the chosen model.

Outputs (consumed by build_outputs.py):
  data/model_artifacts.npz / .json   feature matrix, scores, metrics, PCA, modes
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MIN_FREQ = 0.01           # drop tags appearing in <1% of corpus
ADD_DEMOGRAPHIC = True    # Seinen/Shounen/... as binary features
ADD_GENRES = True         # AniList genres as binary features
BERSERK_ID = 30002

DEMOGRAPHIC_TAGS = {"Seinen", "Shounen", "Shoujo", "Josei", "Kids"}


def build_feature_matrix(corpus):
    """Return X (DataFrame, index=id) of tag-rank/100 + optional binaries."""
    n = len(corpus)
    # count tag frequency
    tag_count = {}
    for tags in corpus["tags"]:
        for t in tags:
            tag_count[t["name"]] = tag_count.get(t["name"], 0) + 1
    keep_tags = sorted(k for k, v in tag_count.items() if v / n >= MIN_FREQ)
    tag_idx = {t: i for i, t in enumerate(keep_tags)}

    X = np.zeros((n, len(keep_tags)), dtype=float)
    for r, tags in enumerate(corpus["tags"]):
        for t in tags:
            j = tag_idx.get(t["name"])
            if j is not None and t["rank"] is not None:
                X[r, j] = t["rank"] / 100.0
    cols = list(keep_tags)
    Xdf = pd.DataFrame(X, columns=cols, index=corpus["id"].values)

    # demographic binaries (some demographics are tags in AniList)
    if ADD_DEMOGRAPHIC:
        for dem in sorted(DEMOGRAPHIC_TAGS):
            col = f"demo::{dem}"
            present = corpus["tags"].apply(
                lambda ts: 1.0 if any(t["name"] == dem for t in ts) else 0.0)
            Xdf[col] = present.values

    # genre binaries
    if ADD_GENRES:
        all_genres = sorted({g for gs in corpus["genres"] for g in gs})
        for g in all_genres:
            col = f"genre::{g}"
            Xdf[col] = corpus["genres"].apply(lambda gs: 1.0 if g in gs else 0.0).values

    return Xdf


def loo_auc_logistic(Xl, yl, C, wl=None):
    """Leave-one-out predicted probabilities -> AUC & accuracy for a given C.
    Optional per-sample weights wl (e.g. loved=1.0, liked=0.5)."""
    loo = LeaveOneOut()
    preds = np.zeros(len(yl))
    for tr, te in loo.split(Xl):
        sc = StandardScaler().fit(Xl[tr])
        clf = LogisticRegression(C=C, max_iter=5000,  # default penalty is L2
                                 class_weight="balanced")
        if wl is not None:
            clf.fit(sc.transform(Xl[tr]), yl[tr], sample_weight=wl[tr])
        else:
            clf.fit(sc.transform(Xl[tr]), yl[tr])
        preds[te] = clf.predict_proba(sc.transform(Xl[te]))[:, 1]
    auc = roc_auc_score(yl, preds)
    acc = accuracy_score(yl, (preds >= 0.5).astype(int))
    return auc, acc, preds


def loo_auc_nsc(Xl, yl, wl=None):
    """Diagonal-LDA / nearest-shrunken-centroid LOO score.
    Per-tag weight w = (mean_pos - mean_neg)/pooled_sd; title score = X.w.
    Optional per-sample weights wl produce weighted class means (loved>liked)."""
    if wl is None:
        wl = np.ones(len(yl))
    loo = LeaveOneOut()
    preds = np.zeros(len(yl))
    for tr, te in loo.split(Xl):
        Xtr, ytr, wtr = Xl[tr], yl[tr], wl[tr]
        pos, neg = Xtr[ytr == 1], Xtr[ytr == 0]
        wp, wn = wtr[ytr == 1], wtr[ytr == 0]
        mp = np.average(pos, axis=0, weights=wp)
        mn = np.average(neg, axis=0, weights=wn)
        vp, vn = pos.var(0, ddof=1), neg.var(0, ddof=1)
        npos, nneg = len(pos), len(neg)
        pooled = np.sqrt(((npos - 1) * vp + (nneg - 1) * vn) /
                         max(npos + nneg - 2, 1)) + 1e-6
        w = (mp - mn) / pooled
        preds[te] = Xl[te] @ w
    auc = roc_auc_score(yl, preds)
    acc = accuracy_score(yl, (preds >= np.median(preds)).astype(int))
    return auc, acc, w


def cosine(A, b):
    bn = b / (np.linalg.norm(b) + 1e-12)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return An @ bn


def main():
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    res = pd.read_parquet(DATA / "labels_resolved.parquet")

    Xdf = build_feature_matrix(corpus)
    feat_cols = list(Xdf.columns)
    X = Xdf.values
    ids = Xdf.index.values
    id_to_row = {int(i): r for r, i in enumerate(ids)}
    print(f"Feature space: {X.shape[1]} features over {X.shape[0]} titles "
          f"(tags kept >= {MIN_FREQ*100:.0f}% freq).")

    # --- assemble fit labels ---
    # Base: original label table (STRONG_POS=1, NEG=0), weight 1.0.
    # Then MERGE the reader's real 43 exported ratings (data/user_library_seed.json):
    #   loved   -> positive, weight 1.0
    #   liked   -> positive, weight 0.5
    #   disliked-> negative, weight 1.0
    #   meh / unrated -> EXCLUDED
    # User ratings OVERRIDE the original table where ids overlap.
    lab = {}   # id -> (y, weight)
    base = res[res["group"].isin(["STRONG_POS", "NEG"])].dropna(subset=["id"])
    for _, r in base.iterrows():
        tid = int(r["id"])
        if tid in id_to_row:
            lab[tid] = (int(r["label"]), 1.0)

    user_seed = ROOT / "data" / "user_library_seed.json"
    n_user_added = 0
    if user_seed.exists():
        entries = json.loads(user_seed.read_text()).get("entries", {})
        wmap = {"loved": (1, 1.0), "liked": (1, 0.5), "disliked": (0, 1.0)}
        for k, v in entries.items():
            tid = int(k)
            rating = v.get("rating")
            if rating in wmap and tid in id_to_row:
                if tid not in lab:
                    n_user_added += 1
                lab[tid] = wmap[rating]      # user overrides
        print(f"Merged user export: {len(entries)} entries -> "
              f"{n_user_added} new fit labels added (meh/unrated excluded).")

    fit_ids = sorted(lab.keys())
    fit_rows = [id_to_row[i] for i in fit_ids]
    y = np.array([lab[i][0] for i in fit_ids])
    wts = np.array([lab[i][1] for i in fit_ids], dtype=float)
    Xl = X[fit_rows]
    print(f"Fit labels: {len(y)}  (pos={int((y==1).sum())}, neg={int((y==0).sum())}, "
          f"weighted-pos={wts[y==1].sum():.1f})")

    # --- primary: L2 logistic, choose C by LOO-AUC (weighted) ---
    Cgrid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best = None
    for C in Cgrid:
        auc, acc, _ = loo_auc_logistic(Xl, y, C, wl=wts)
        print(f"  logistic C={C:<5}  LOO-AUC={auc:.3f}  LOO-acc={acc:.3f}")
        if best is None or auc > best[1]:
            best = (C, auc, acc)
    bestC, log_auc, log_acc = best
    print(f"  -> chosen C={bestC} (LOO-AUC={log_auc:.3f}, LOO-acc={log_acc:.3f})")

    # robustness model (weighted class means)
    nsc_auc, nsc_acc, _ = loo_auc_nsc(Xl, y, wl=wts)
    print(f"  NSC / diagonal-LDA  LOO-AUC={nsc_auc:.3f}  LOO-acc={nsc_acc:.3f}")

    # --- fit final logistic on all labels, score full corpus ---
    scaler = StandardScaler().fit(Xl)
    clf = LogisticRegression(C=bestC, max_iter=5000,
                             class_weight="balanced").fit(
        scaler.transform(Xl), y, sample_weight=wts)
    model_score = clf.predict_proba(scaler.transform(X))[:, 1]

    # --- baselines: cosine to Berserk vector and to positive centroid ---
    # Positive set for cosine + bimodality = ALL fitted positives (original
    # STRONG_POS plus the reader's loved+liked from the export).
    berserk_vec = X[id_to_row[BERSERK_ID]]
    pos_ids_all = [i for i in fit_ids if lab[i][0] == 1]
    pos_rows = [id_to_row[i] for i in pos_ids_all]
    pos_centroid = X[pos_rows].mean(0)
    berserk_cos = cosine(X, berserk_vec)
    centroid_cos = cosine(X, pos_centroid)

    # --- bimodality on the full positive set ---
    Xpos = X[pos_rows]
    pos_ids = [int(ids[r]) for r in pos_rows]
    pos_titles = {int(c["id"]): (c["english"] or c["romaji"])
                  for _, c in corpus.iterrows()}
    modes = {}
    for k in (2, 3):
        if len(Xpos) < k:
            continue
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xpos)
        members = {}
        for pid, lab in zip(pos_ids, km.labels_):
            members.setdefault(int(lab), []).append(pos_titles[pid])
        # silhouette-ish: within vs between via inertia per k (report inertia)
        modes[k] = {"labels": km.labels_.tolist(),
                    "members": members,
                    "centroids": km.cluster_centers_.tolist(),
                    "inertia": float(km.inertia_),
                    "ids": pos_ids}

    # per-mode nearest unseen titles for k=2 (the expected bimodal split)
    mode_nn = {}
    if 2 in modes:
        for lab, cen in enumerate(modes[2]["centroids"]):
            cen = np.array(cen)
            sims = cosine(X, cen)
            order = np.argsort(-sims)
            picks = []
            for r in order:
                tid = int(ids[r])
                if tid in pos_ids:        # skip the strong positives themselves
                    continue
                picks.append({"id": tid,
                              "title": (corpus.iloc[r]["english"]
                                        or corpus.iloc[r]["romaji"]),
                              "country": corpus.iloc[r]["country"],
                              "sim": float(sims[r])})
                if len(picks) >= 15:
                    break
            mode_nn[lab] = picks

    # --- PCA for the taste map (2D) + top loadings ---
    pca = PCA(n_components=2, random_state=0).fit(X)
    coords = pca.transform(X)
    loadings = pca.components_  # (2, p)
    def top_loadings(pc, n=10):
        idx = np.argsort(-np.abs(pc))[:n]
        return [(feat_cols[i], float(pc[i])) for i in idx]
    pc1_top = top_loadings(loadings[0])
    pc2_top = top_loadings(loadings[1])
    print("\nPC1 top loadings:", [f"{n}:{w:+.2f}" for n, w in pc1_top[:6]])
    print("PC2 top loadings:", [f"{n}:{w:+.2f}" for n, w in pc2_top[:6]])

    # --- held-out MOD_POS sanity check ---
    mod = res[res["group"] == "MOD_POS"].dropna(subset=["id"])
    median_score = float(np.median(model_score))
    mod_check = []
    for _, r in mod.iterrows():
        tid = int(r["id"])
        if tid not in id_to_row:
            continue
        s = float(model_score[id_to_row[tid]])
        mod_check.append({"title": r["matched_english"] or r["matched_romaji"],
                          "id": tid, "score": s, "above_median": s > median_score})
    n_above = sum(m["above_median"] for m in mod_check)
    print(f"\nHeld-out MOD_POS above corpus median: {n_above}/{len(mod_check)} "
          f"(median={median_score:.3f})")

    # --- persist everything ---
    np.savez(DATA / "model_artifacts.npz",
             ids=ids, coords=coords, model_score=model_score,
             berserk_cos=berserk_cos, centroid_cos=centroid_cos)
    meta = {
        "feat_cols": feat_cols,
        "n_features": X.shape[1],
        "n_titles": X.shape[0],
        "min_freq": MIN_FREQ,
        "fit_labels": {"n": int(len(y)), "pos": int(y.sum()), "neg": int((y == 0).sum())},
        "logistic": {"C": bestC, "loo_auc": log_auc, "loo_acc": log_acc,
                     "C_grid": Cgrid},
        "nsc": {"loo_auc": nsc_auc, "loo_acc": nsc_acc},
        "pc1_top": pc1_top, "pc2_top": pc2_top,
        "modes": {str(k): {"members": v["members"], "inertia": v["inertia"],
                           "ids": v["ids"], "labels": v["labels"]}
                  for k, v in modes.items()},
        "mode_nn": {str(k): v for k, v in mode_nn.items()},
        "mod_pos_check": mod_check,
        "mod_pos_n_above": n_above,
        "median_score": median_score,
        "berserk_id": BERSERK_ID,
    }
    (DATA / "model_artifacts.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved artifacts -> {DATA/'model_artifacts.npz'} + .json")


if __name__ == "__main__":
    main()
