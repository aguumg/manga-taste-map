#!/usr/bin/env python3
"""Fetch full plot DESCRIPTIONS for the Western corpus from Comicvine volume-detail.

Gate-1 pivot: Comicvine 'concepts' are junk (cover/variant metadata), so we model
Western titles by their plot description -> embed -> project into the AniList tag space.
This script just guarantees every Western title has a clean description.

Reads key from comicvine_key.env. Caches to data/western_corpus.json. ~1 req/sec.
"""
import json, re, time, html, urllib.request, urllib.parse, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KEY = None
for line in (ROOT / "comicvine_key.env").read_text().splitlines():
    m = re.search(r"([0-9a-f]{30,})", line)
    if m:
        KEY = m.group(1); break
assert KEY, "no Comicvine key found in comicvine_key.env"

UA = {"User-Agent": "taste-model/1.0 (personal research)"}
BASE = "https://comicvine.gamespot.com/api"

def strip_html(s):
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def search_volume(query):
    q = urllib.parse.urlencode({"api_key": KEY, "format": "json",
                                "resources": "volume", "query": query, "limit": 5,
                                "field_list": "id,name,start_year,publisher,count_of_issues"})
    d = get(f"{BASE}/search/?{q}")
    return d.get("results", [])

def volume_detail(vid):
    q = urllib.parse.urlencode({"api_key": KEY, "format": "json",
                                "field_list": "id,name,start_year,publisher,deck,description"})
    d = get(f"{BASE}/volume/4050-{vid}/?{q}")
    return d.get("results", {})

# Seed list: all discussed titles + a broader notable dark/mature set.
SEED = [
    "Crossed", "Y: The Last Man", "Locke & Key", "Saga", "Sara", "Warren Ellis Crecy",
    "Scalped", "The Sheriff of Babylon", "Northlanders", "The Killer", "Criminal",
    "100 Bullets", "The Punisher MAX", "Fury MAX", "Die", "Bouncer", "Blacksad",
    "The Black Monday Murders", "The Realm", "Gideon Falls", "Manifest Destiny",
    "Pretty Deadly", "Birthright", "The Goddamned", "Sweet Tooth", "Wytches",
    "Outcast", "Deadly Class", "Southern Bastards", "East of West", "The Old Guard",
    "Pretty Deadly", "Sandman", "Preacher", "Hellblazer", "Transmetropolitan",
    "DMZ", "Ex Machina", "Wytches", "Gideon Falls", "Nameless", "Providence",
    "Neonomicon", "The Boys", "Crossed +100", "Uber", "Ferals", "Crossed: Badlands",
    "Two Brothers", "A History of Violence", "Road to Perdition", "Sin City",
    "300", "From Hell", "Watchmen", "V for Vendetta", "Berlin", "Maus",
    "Black Hole", "Hellboy", "B.P.R.D.", "The Goon", "Fatale", "Incognito",
    "Kill or be Killed", "My Heroes Have Always Been Junkies", "Reckless",
    "Pulp", "That Texas Blood", "Gideon Falls", "Lazarus", "Trees", "Injection",
    "Low", "Black Science", "Paper Girls", "We Stand on Guard", "The Nice House on the Lake",
    "Something is Killing the Children", "Department of Truth", "Ice Cream Man",
    "Gideon Falls", "Coffin Bound", "20th Century Men", "Undiscovered Country",
]

def main():
    cache_f = DATA / "western_corpus.json"
    corpus = {}
    if cache_f.exists() and "--refresh" not in sys.argv:
        corpus = json.loads(cache_f.read_text())
        print(f"loaded {len(corpus)} cached")
    seen_names = {v["name"].lower() for v in corpus.values()}
    for i, title in enumerate(dict.fromkeys(SEED)):  # dedupe preserve order
        if any(title.lower().split(":")[0] in n for n in seen_names):
            continue
        try:
            res = search_volume(title)
            time.sleep(1.0)
            if not res:
                print(f"  [{i}] NO MATCH: {title}"); continue
            # pick the result with most issues (the main series), tie-break earliest year
            res.sort(key=lambda r: (-(r.get("count_of_issues") or 0), r.get("start_year") or "9999"))
            v = res[0]
            det = volume_detail(v["id"])
            time.sleep(1.0)
            desc = strip_html(det.get("description") or det.get("deck") or "")
            if len(desc) < 40:
                print(f"  [{i}] thin desc, skip: {title}"); continue
            pub = (det.get("publisher") or {}).get("name") if isinstance(det.get("publisher"), dict) else None
            corpus[str(v["id"])] = {
                "id": v["id"], "name": det.get("name") or v.get("name"),
                "year": det.get("start_year"), "publisher": pub,
                "description": desc[:1200],
            }
            seen_names.add((det.get("name") or v["name"]).lower())
            print(f"  [{i}] ok: {corpus[str(v['id'])]['name']} ({pub}) — {len(desc)} chars")
            if i % 10 == 0:
                cache_f.write_text(json.dumps(corpus, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"  [{i}] ERR {title}: {e}")
            time.sleep(2.0)
    cache_f.write_text(json.dumps(corpus, ensure_ascii=False, indent=2))
    withdesc = sum(1 for v in corpus.values() if len(v.get("description", "")) >= 40)
    print(f"\nDONE: {len(corpus)} western titles, {withdesc} with usable descriptions")
    print("saved -> data/western_corpus.json")

if __name__ == "__main__":
    main()
