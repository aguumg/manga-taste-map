#!/usr/bin/env python3
"""Find NEW *curated* dark Western GNs in GCD (data/gcd.sqlite). v2: aggressive filtering
to cut anthologies / reprints / superhero noise that gamed v1's ranking.

Run AFTER parse_gcd.py genre. Output: data/gcd_dark_candidates.json
"""
import json, re, sqlite3, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB = DATA / "gcd.sqlite"

DARK = ["horror", "crime", "war", "noir", "detective", "suspense"]
# quality mature/indie houses (substring, lowercase)
GOOD_PUB = ["image", "vertigo", "dark horse", "idw", "boom", "avatar", "oni",
            "fantagraphics", "top shelf", "caliber", "aftershock", "vault",
            "black mask", "scout", "tko", "humanoids", "dynamite", "valiant",
            "first comics", "eclipse", "black label", "berger"]
# name patterns that signal reprint / anthology / reference -> drop
BAD_NAME = ["reprint", "archive", "classics", "library", "album", "collection",
            "annual", "megazine", "omnibus", "complete", "best of", "reader",
            "sketchbook", "artist", "companion", "encyclopedia", "handbook",
            "who's who", "edition", "tpb", "treasury", "anthology", "presents",
            "spectacular", "giant", "quarterly", "showcase", "digest"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def top_genre(genres):
    m = re.match(r"([a-z \-]+)\(", genres or "")
    return m.group(1).strip() if m else ""


def main():
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row; cur = con.cursor()
    have = set()
    for f in ["western_projected_tags.json", "western_corpus.json"]:
        p = DATA / f
        if p.exists():
            d = json.loads(p.read_text())
            for v in (d.values() if isinstance(d, dict) else d):
                nm = v if isinstance(v, str) else (v.get("name") or v.get("title") or "")
                have.add(norm(nm))

    rows = cur.execute("""
        SELECT s.id,s.name,s.year_began,s.issue_count,p.name AS pub,g.genres,g.synopsis
        FROM series s JOIN series_genre g ON g.series_id=s.id
        LEFT JOIN publisher p ON p.id=s.publisher_id
        WHERE s.year_began>=1980 AND s.issue_count BETWEEN 1 AND 80
    """).fetchall()

    cand = []
    for r in rows:
        name = r["name"] or ""; genres = (r["genres"] or "").lower()
        pub = (r["pub"] or "").lower(); syn = (r["synopsis"] or "")
        if len(syn) < 40:
            continue
        if not any(d in genres for d in DARK):
            continue
        if top_genre(genres) in ("superhero", "humor", "anthropomorphic-funny animals",
                                 "children", "teen", "romance", "non-fiction", "satire-parody"):
            continue
        if any(b in name.lower() for b in BAD_NAME):
            continue
        if not any(gp in pub for gp in GOOD_PUB):
            continue
        if norm(name) in have:
            continue
        darkhits = sum(genres.count(d) for d in DARK)
        pubbonus = 3 if any(x in pub for x in ["image", "vertigo", "dark horse", "avatar",
                                               "fantagraphics", "black label", "tko"]) else 1
        score = round(darkhits * 2 + pubbonus + min(r["issue_count"] or 0, 40) / 20, 2)
        cand.append({"gcd_id": r["id"], "name": name, "year": r["year_began"],
                     "publisher": r["pub"], "issue_count": r["issue_count"],
                     "genres": r["genres"], "synopsis": syn[:400], "score": score})

    # dedupe by normalized name (keep highest score)
    best = {}
    for c in cand:
        k = norm(c["name"])
        if k not in best or c["score"] > best[k]["score"]:
            best[k] = c
    cand = sorted(best.values(), key=lambda c: -c["score"])
    (DATA / "gcd_dark_candidates.json").write_text(json.dumps(cand, ensure_ascii=False, indent=2))
    print(f"curated dark candidates (new): {len(cand)}")
    print("\nTop 45:")
    for c in cand[:45]:
        print(f"  [{c['score']:>5}] {c['name']} ({c['year']}, {c['publisher']}) | "
              f"{top_genre(c['genres'].lower())} | {c['synopsis'][:64]}…")
    con.close()
    print("\nsaved -> data/gcd_dark_candidates.json")


if __name__ == "__main__":
    main()
