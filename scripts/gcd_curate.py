#!/usr/bin/env python3
"""Second-pass curation of gcd_dark_candidates.json: drop TV/movie/game tie-ins and
promo/magazine series (whose 'synopsis' is a letters page or ad), cap to a clean set
for LLM tagging. Output: data/gcd_for_tagging.json
"""
import json, re, pathlib

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
CAP = 120

# licensed tie-in franchises -> drop (Hellboy/BPRD/Mignolaverse kept: original + dark)
TIEIN = ["csi", "buffy", "angel", "tomb raider", "star wars", "star trek", "aliens",
         "predator", "terminator", "robocop", "godzilla", "transformers", "g.i. joe",
         "halo", "mass effect", "witcher", "assassin", "x-files", "planet of the apes",
         "evil dead", "army of darkness", "friday the 13th", "elm street", "chucky",
         "rambo", "conan", "red sonja", "vampirella", "shadow", "green hornet",
         "lone ranger", "zorro", "tarzan", "flash gordon", "battlestar", "doctor who",
         "simpsons", "disney", "looney", "pacific rim", "dark horse insider",
         "dark horse comics", "image+", "tomb raider", "buffyverse"]
# synopsis fingerprints of non-story (promo / letters / index) -> drop
PROMO = ["letters from", "letters of comment", "non-fiction comic", "talks about",
         "promo ads", "introduction and plot", "information on", "preview of",
         "behind the scenes", "annotations for", "back matter", "table of contents",
         "pin-up", "pinup gallery", "cover gallery", "sketchbook"]


def main():
    cand = json.loads((DATA / "gcd_dark_candidates.json").read_text())
    clean = []
    for c in cand:
        nm = c["name"].lower(); syn = (c["synopsis"] or "").lower()
        if any(t in nm for t in TIEIN):
            continue
        if any(p in syn for p in PROMO):
            continue
        if len(syn) < 60:
            continue
        clean.append(c)
    clean = clean[:CAP]
    out = [{"name": c["name"], "year": c["year"], "publisher": c["publisher"],
            "gcd_genres": c["genres"], "synopsis": c["synopsis"]} for c in clean]
    (DATA / "gcd_for_tagging.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"clean GCD titles for tagging: {len(out)} (from {len(cand)})")
    for c in out[:50]:
        print(f"  - {c['name']} ({c['year']}, {c['publisher']})")
    print("saved -> data/gcd_for_tagging.json")


if __name__ == "__main__":
    main()
