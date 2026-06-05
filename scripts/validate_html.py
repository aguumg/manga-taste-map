"""Validate the embedded JSON + replicate JS cosine & taste-fit to prove the
browser will compute the same thing. Run after build_html.py."""
import json
import math
import re

import numpy as np

HTML = "outputs/taste_map_app.html"


def load_data():
    h = open(HTML, encoding="utf-8").read()
    blob = re.search(r"const DATA = (\{.*?\});\nconst LS_LIB", h, re.S).group(1)
    # strict parse like a browser (reject NaN/Infinity)
    return json.loads(blob, parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError("bad const " + c)))


def main():
    data = load_data()
    titles = data["titles"]
    nfeat = data["nFeat"]
    by_id = {t["id"]: t for t in titles}
    tag_idx = {n: i for i, n in enumerate(data["tags"])}
    disp = lambda t: t["e"] or t["r"] or f"#{t['id']}"

    for t in titles:
        t["norm"] = math.sqrt(sum(v * v for _, v in t["v"])) or 1e-9

    print(f"STRICT JSON parse OK | titles={len(titles)} tags={len(data['tags'])} "
          f"nFeat={nfeat} genres={len(data['genres'])} "
          f"votableTags={len(data['tagMeta'])} seed={len(data['seed'])}")

    # ---- 1. anchor cosine matches original berserk_cos ----
    def centroid(ids):
        c = np.zeros(nfeat); n = 0
        for i in ids:
            for j, v in by_id[i]["v"]:
                c[j] += v
            n += 1
        if n:
            c /= n
        return c, (math.sqrt(float(c @ c)) or 1e-9)

    def cos(cen, t):
        cv, cn = cen
        return sum(cv[j] * v for j, v in t["v"]) / (cn * t["norm"])

    cen = centroid([data["berserkId"]])
    art = np.load("data/model_artifacts.npz", allow_pickle=True)
    orig = {int(i): float(c) for i, c in zip(art["ids"], art["berserk_cos"])}
    diffs = [abs(orig[t["id"]] - cos(cen, t)) for t in titles if t["id"] in orig]
    print(f"max |JS anchor-cosine - Python berserk_cos| = {max(diffs):.2e}")

    # ---- 2. taste-fit: hating an OP-flavour tag penalizes its titles ----
    # AniList "Overpowered Main Characters" isn't ranked in this top-2000 corpus;
    # use the closest available high-signal power-fantasy tag instead.
    VOTE = {"love": 2, "like": 1, "neutral": 0, "dislike": -1, "hate": -2}
    op_name = next((n for n in ("Overpowered Main Characters", "Super Power",
                                "Cultivation") if n in tag_idx), None)
    print(f"OP-flavour tag used: {op_name!r}")

    def taste_fit(t, votes):
        p = np.zeros(nfeat)
        for name, v in votes.items():
            if name in tag_idx:
                p[tag_idx[name]] = VOTE[v]
        pn = math.sqrt(float(p @ p)) or 1e-9
        return sum(p[j] * val for j, val in t["v"]) / (pn * t["norm"])

    op_titles = [t for t in titles
                 if any(data["tags"][j] == op_name for j, _ in t["v"])]
    print(f"OP-carrying titles in corpus: {len(op_titles)}")
    print("Sample OP titles taste-fit (HATE op vs LOVE op):")
    for t in op_titles[:6]:
        hf = taste_fit(t, {op_name: "hate"})
        lf = taste_fit(t, {op_name: "love"})
        print(f"  {disp(t)[:38]:38s} hate={hf:+.3f}  love={lf:+.3f}")
    bad = [t for t in op_titles
           if taste_fit(t, {op_name: "hate"}) >= taste_fit(t, {op_name: "love"})]
    assert not bad, "hate did not penalize OP titles!"
    print("ASSERT OK: hating OP penalizes OP titles vs loving it.")

    # ---- 3. ranking shift with OP hated ----
    votes = {op_name: "hate"}
    scored = sorted(titles, key=lambda t: -taste_fit(t, votes))
    top20_op = sum(1 for t in scored[:20]
                   if any(data["tags"][j] == op_name for j, _ in t["v"]))
    bot20_op = sum(1 for t in scored[-20:]
                   if any(data["tags"][j] == op_name for j, _ in t["v"]))
    print(f"With OP hated: OP titles in top-20={top20_op}, bottom-20={bot20_op} "
          f"(expect more at bottom)")
    assert bot20_op > top20_op, "hating OP did not push OP titles down overall"
    print("ASSERT OK: hated tag pushes its titles to the bottom of the ranking.")

    # ---- 4. genre filter integrity ----
    print(f"Horror-genre titles: {sum('Horror' in t['g'] for t in titles)}")

    # ---- 5. artist index sanity ----
    at = data["artistTitles"]
    if at:
        b = by_id.get(data["berserkId"])
        print(f"Berserk artists: {b['ar']}")
        for a in b["ar"]:
            ids = at.get(a, [])
            print(f"  {a} -> {[disp(by_id[i]) for i in ids if i in by_id]}")
        # Vinland Saga -> Makoto Yukimura (and Planetes if present)
        vin = by_id.get(30642)
        if vin:
            print(f"Vinland Saga artists: {vin['ar']}")
            for a in vin["ar"]:
                ids = at.get(a, [])
                print(f"  {a} -> {[disp(by_id[i]) for i in ids if i in by_id]}")
    else:
        print("artistTitles empty (staff fetch not finished yet)")

    print("\nVALIDATION COMPLETE")


if __name__ == "__main__":
    main()
