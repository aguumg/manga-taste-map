"""
build_outputs.py — Turn model artifacts into the deliverables.

Reads data/model_artifacts.{npz,json} + data/corpus.parquet + label tables and
produces:
  outputs/scored_titles.csv          every title: score, cosines, country, tags
  outputs/recommendations_top50.csv  top-50 by score, EXCLUDING already-read
  outputs/recommendations_by_mode.csv per-mode top-15 (if bimodal)
  outputs/taste_map.html             single-file interactive Plotly scatter
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def top_tags(tags, n=5):
    ts = sorted(tags, key=lambda t: -(t["rank"] or 0))
    return [t["name"] for t in ts[:n]]


def main():
    corpus = pd.read_parquet(DATA / "corpus.parquet").drop_duplicates("id")
    art = np.load(DATA / "model_artifacts.npz", allow_pickle=True)
    meta = json.loads((DATA / "model_artifacts.json").read_text())
    res = pd.read_parquet(DATA / "labels_resolved.parquet")

    ids = art["ids"]
    score = art["model_score"]
    bcos = art["berserk_cos"]
    ccos = art["centroid_cos"]
    coords = art["coords"]

    df = pd.DataFrame({"id": ids, "score": score,
                       "berserk_cos": bcos, "centroid_cos": ccos,
                       "pc1": coords[:, 0], "pc2": coords[:, 1]})
    df = df.merge(corpus, on="id", how="left")
    df["title"] = df["english"].fillna(df["romaji"])
    df["top_tags"] = df["tags"].apply(lambda ts: ", ".join(top_tags(list(ts))))

    # --- label membership ---
    def ids_for(group):
        return set(int(i) for i in
                   res[res["group"] == group].dropna(subset=["id"])["id"])
    strong = ids_for("STRONG_POS")
    neg = ids_for("NEG")
    modpos = ids_for("MOD_POS")
    exclude_ids = strong | neg | ids_for("EXCLUDE")  # never-recommend set

    df["label_group"] = df["id"].apply(
        lambda i: "STRONG_POS" if i in strong else
                  "NEG" if i in neg else
                  "MOD_POS" if i in modpos else "")

    # --- scored_titles.csv ---
    scored = df[["id", "title", "country", "year", "averageScore", "popularity",
                 "score", "berserk_cos", "centroid_cos", "top_tags",
                 "label_group"]].sort_values("score", ascending=False)
    scored.to_csv(OUT / "scored_titles.csv", index=False)

    # --- recommendations_top50.csv (exclude already-read) ---
    recs = df[~df["id"].isin(exclude_ids)].sort_values("score", ascending=False)
    recs50 = recs.head(50)[["id", "title", "country", "year", "score",
                            "berserk_cos", "centroid_cos", "top_tags"]]
    recs50.to_csv(OUT / "recommendations_top50.csv", index=False)

    # --- per-mode top-15 (if bimodal) ---
    # mode_nn is keyed by k=2 cluster label ("0","1") -> list of nearest titles
    mode_nn = meta.get("mode_nn", {})
    if mode_nn:
        rows = []
        title_by_id = dict(zip(df["id"], df["title"]))
        tags_by_id = dict(zip(df["id"], df["top_tags"]))
        for lab, picks in mode_nn.items():
            for rank, p in enumerate(picks, 1):
                if p["id"] in exclude_ids:
                    continue
                rows.append({"mode": lab, "rank": rank, "id": p["id"],
                             "title": title_by_id.get(p["id"], p["title"]),
                             "country": p["country"], "sim": round(p["sim"], 3),
                             "top_tags": tags_by_id.get(p["id"], "")})
        pd.DataFrame(rows).to_csv(OUT / "recommendations_by_mode.csv", index=False)

    # --- taste_map.html ---
    build_map(df, strong, neg, modpos, meta)

    print("Wrote scored_titles.csv, recommendations_top50.csv, "
          "recommendations_by_mode.csv, taste_map.html")


def build_map(df, strong, neg, modpos, meta):
    log = meta["logistic"]
    pc1 = ", ".join(f"{n}" for n, w in meta["pc1_top"][:4])
    pc2 = ", ".join(f"{n}" for n, w in meta["pc2_top"][:4])

    def hover(row):
        return (f"<b>{row['title']}</b><br>{row['country']} · {row.get('year','')}"
                f"<br>score={row['score']:.2f} · Berserk-cos={row['berserk_cos']:.2f}"
                f"<br>{row['top_tags']}")

    base = df[df["label_group"] == ""]
    sp = df[df["id"].isin(strong)]
    ng = df[df["id"].isin(neg)]
    mp = df[df["id"].isin(modpos)]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=base["pc1"], y=base["pc2"], mode="markers",
        marker=dict(size=5, color=base["score"], colorscale="Turbo",
                    cmin=0, cmax=1, opacity=0.55, showscale=True,
                    colorbar=dict(title="pred<br>score")),
        text=base.apply(hover, axis=1), hoverinfo="text", name="corpus (2000)"))

    fig.add_trace(go.Scattergl(
        x=sp["pc1"], y=sp["pc2"], mode="markers+text",
        marker=dict(symbol="star", size=16, color="gold",
                    line=dict(width=1, color="black")),
        text=sp["title"], textposition="top center",
        textfont=dict(size=9, color="black"),
        hovertext=sp.apply(hover, axis=1), hoverinfo="text",
        name="STRONG positives"))

    fig.add_trace(go.Scattergl(
        x=ng["pc1"], y=ng["pc2"], mode="markers+text",
        marker=dict(symbol="x", size=11, color="crimson",
                    line=dict(width=1)),
        text=ng["title"], textposition="bottom center",
        textfont=dict(size=8, color="crimson"),
        hovertext=ng.apply(hover, axis=1), hoverinfo="text",
        name="negatives"))

    fig.add_trace(go.Scattergl(
        x=mp["pc1"], y=mp["pc2"], mode="markers+text",
        marker=dict(symbol="diamond", size=11, color="silver",
                    line=dict(width=1, color="black")),
        text=mp["title"], textposition="top center",
        textfont=dict(size=8, color="dimgray"),
        hovertext=mp.apply(hover, axis=1), hoverinfo="text",
        name="moderate (held-out)"))

    subtitle = (f"L2-logistic on AniList tag-ranks (C={log['C']}), "
                f"LOO-CV AUC={log['loo_auc']:.2f} · "
                f"NSC robustness AUC={meta['nsc']['loo_auc']:.2f} · "
                f"{meta['n_features']} tag/genre features, n={meta['fit_labels']['n']} labels")
    fig.update_layout(
        title=dict(text=("Manga/Manhwa Taste Map<br>"
                         f"<span style='font-size:12px'>{subtitle}</span>"),
                   x=0.5),
        xaxis_title=f"PC1  (loads on: {pc1})",
        yaxis_title=f"PC2  (loads on: {pc2})",
        template="plotly_white", width=1300, height=850,
        legend=dict(orientation="h", y=-0.12))

    fig.write_html(OUT / "taste_map.html", include_plotlyjs=True, full_html=True)


if __name__ == "__main__":
    main()
