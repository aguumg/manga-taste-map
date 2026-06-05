#!/usr/bin/env python3
"""Broaden the Western candidate pool beyond pure-dark: pull action/sci-fi/war/adventure/
crime Western GNs (not just horror) from quality publishers, so the Western universe gets
VARIETY and disperses. Still respects the floor (no romance/kids/humor; the LLM tagger
enforces the explicit-violence requirement downstream).

Run after parse_gcd.py genre. Output: data/gcd_broad_for_tagging.json
"""
import json, re, sqlite3, pathlib

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
DB = DATA / "gcd.sqlite"
CAP = 260

# action/violent-capable genres to INCLUDE (broader than the dark-only set)
INCLUDE = ["horror", "crime", "war", "noir", "detective", "suspense", "science fiction",
           "adventure", "fantasy", "western-frontier", "historical", "spy", "martial arts"]
# if the TOP genre is one of these, drop (floor: no romance/kids/comedy)
BAD_TOP = ["humor", "anthropomorphic-funny animals", "children", "teen", "romance",
           "non-fiction", "satire-parody", "biography", "religious", "sports", "erotica"]
GOOD_PUB = ["image", "vertigo", "dark horse", "idw", "boom", "avatar", "oni",
            "fantagraphics", "top shelf", "aftershock", "vault", "black mask", "scout",
            "tko", "humanoids", "dynamite", "valiant", "first comics", "eclipse",
            "black label", "berger", "dc", "marvel"]
BAD_NAME = ["reprint", "archive", "classics", "library", "album", "collection", "annual",
            "megazine", "omnibus", "complete", "best of", "reader", "sketchbook", "artist",
            "companion", "handbook", "who's who", "edition", "tpb", "treasury", "anthology",
            "presents", "spectacular", "quarterly", "showcase", "digest", "free comic book",
            "image firsts", "image+", "insider"]


def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())
def top_genre(g):
    m = re.match(r"([a-z \-]+)\(", g or ""); return m.group(1).strip() if m else ""


def main():
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row; cur = con.cursor()
    have = set()
    for f in ["western_llm_tags.json", "western_projected_tags.json"]:
        p = DATA / f
        if p.exists():
            for k in json.loads(p.read_text()):
                have.add(norm(k))

    rows = cur.execute("""SELECT s.id,s.name,s.year_began,s.issue_count,p.name AS pub,
        g.genres,g.synopsis FROM series s JOIN series_genre g ON g.series_id=s.id
        LEFT JOIN publisher p ON p.id=s.publisher_id
        WHERE s.year_began>=1980 AND s.issue_count BETWEEN 1 AND 80""").fetchall()

    cand = []
    for r in rows:
        name = r["name"] or ""; genres = (r["genres"] or "").lower()
        pub = (r["pub"] or "").lower(); syn = r["synopsis"] or ""; tg = top_genre(genres)
        if len(syn) < 60: continue
        if tg in BAD_TOP: continue
        if not any(g in genres for g in INCLUDE): continue
        # superhero allowed ONLY if it co-occurs with a mature/violent genre
        if tg == "superhero" and not any(x in genres for x in ["crime", "horror", "war", "noir"]):
            continue
        if any(b in name.lower() for b in BAD_NAME): continue
        if not any(gp in pub for gp in GOOD_PUB): continue
        if norm(name) in have: continue
        # score favors variety: reward non-horror action genres + quality pub
        variety = sum(genres.count(g) for g in ["science fiction", "adventure", "war",
                                                "crime", "western-frontier", "fantasy"])
        pubbonus = 2 if any(x in pub for x in ["image", "dark horse", "vertigo", "black label",
                                               "fantagraphics", "avatar", "tko"]) else 1
        cand.append({"name": name, "year": r["year_began"], "publisher": r["pub"],
                     "issue_count": r["issue_count"], "gcd_genres": r["genres"],
                     "synopsis": syn[:400], "_s": variety + pubbonus + min(r["issue_count"], 30)/15})

    best = {}
    for c in cand:
        k = norm(c["name"])
        if k not in best or c["_s"] > best[k]["_s"]:
            best[k] = c
    out = sorted(best.values(), key=lambda c: -c["_s"])[:CAP]
    for c in out: c.pop("_s")
    (DATA / "gcd_broad_for_tagging.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"broadened Western candidates (new, varied): {len(out)}")
    from collections import Counter
    tg = Counter(top_genre(c["gcd_genres"].lower()) for c in out)
    print("top-genre mix:", dict(tg.most_common(10)))
    print("\nsample 40:")
    for c in out[:40]:
        print(f"  {c['name']} ({c['year']}, {c['publisher']}) | {top_genre(c['gcd_genres'].lower())}")
    print("\nsaved -> data/gcd_broad_for_tagging.json")


if __name__ == "__main__":
    main()
