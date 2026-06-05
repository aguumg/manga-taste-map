"""
fetch_western.py — Build a Western (US/EU) comics corpus from Comicvine.

Two-stage, heavily cached (data/western_*.json; never re-hits unless --refresh):
  STAGE A: resolve the explicit must-include volumes by publisher+year.
  STAGE B: broaden with notable volumes from dark/mature-leaning publishers
           (Image, Vertigo/DC Black Label, Dark Horse, Avatar, IDW, BOOM!, Oni),
           sorted by issue count (popularity proxy), to ~300-500 total.

For each volume we fetch the DETAIL endpoint (id, name, publisher, start_year,
deck, description, concepts, character_credits count).

Comicvine etiquette: ~1 req/sec, ~200 req per resource-type per hour. We sleep
1.1s between requests, cache aggressively, and cap the corpus so we never blow
the hourly cap. A User-Agent header is mandatory (Comicvine blocks default).

The API key is read from comicvine_key.env and NEVER printed or hardcoded.

Outputs:
  data/western_corpus.json   list of volume dicts (the corpus)
  data/western_raw_detail.json  per-id detail cache (resume-safe)
  data/western_search_cache.json  per-query search cache

Usage:
  python fetch_western.py                 # use caches; fetch only what's missing
  python fetch_western.py --target 400    # broaden target size
  python fetch_western.py --refresh        # wipe caches, re-fetch
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ENV = ROOT / "comicvine_key.env"
API = "https://comicvine.gamespot.com/api"
HEADERS = {"User-Agent": "manga-taste-model/1.0 (personal taste research)"}
SLEEP = 1.1

CORPUS = DATA / "western_corpus.json"
DETAIL_CACHE = DATA / "western_raw_detail.json"
SEARCH_CACHE = DATA / "western_search_cache.json"

DETAIL_FIELDS = ("id,name,publisher,start_year,deck,description,concepts,"
                 "character_credits,count_of_issues")

# Must-include titles: (query, publisher_substr|None, year|None). The resolver
# picks the first search hit matching publisher+year (year within +/-1).
MUST_INCLUDE = [
    ("Crossed", "Avatar", 2008),
    ("Y The Last Man", "DC", 2002),          # Vertigo is under DC Comics publisher
    ("Locke & Key", "IDW", 2008),
    ("Saga", "Image", 2012),
    ("Sara", "TKO", None),
    ("Crecy", "Avatar", None),
    ("Scalped", "DC", 2007),                 # Vertigo
    ("The Sheriff of Babylon", "DC", 2015),  # Vertigo
    ("Northlanders", "DC", 2007),            # Vertigo
    ("The Killer", "Archaia", None),
    ("Criminal", "Marvel", 2006),            # Brubaker/Icon (Marvel)
    ("100 Bullets", "DC", 1999),             # Vertigo
    ("Punisher Max", "Marvel", 2004),        # Ennis MAX
    ("Fury Max", "Marvel", 2012),
    ("Die", "Image", 2018),
    ("The Bouncer", None, None),
    ("Once Upon a Time in France", None, None),
    ("Blacksad", None, None),
    ("The Black Monday Murders", "Image", 2016),
    ("The Realm", "Image", 2017),
]

# Broadening publishers (dark/mature-leaning). Comicvine publisher ids:
#   Image=513, DC=10, Dark Horse=364, Avatar=2640, IDW=1190, BOOM!=2350, Oni=1108
BROADEN_PUBLISHERS = {
    "Image": 513, "DC (Vertigo/Black Label)": 10, "Dark Horse": 364,
    "Avatar": 2640, "IDW": 1190, "BOOM! Studios": 2350, "Oni": 1108,
}

# noise concepts = cover/variant/award/event metadata, NOT thematic content
NOISE = re.compile(
    r"variant|cover|exclusive|comic-?con|expo|award|homage|sketch|virgin art|"
    r"wraparound|blank|connecting|anniversary|reprint|signed|retailer|incentive|"
    r"logo|skybound|imprint|pride month|comics-unrelated|foil|hologram|"
    r"crossover event|second print|\d+th print|gatefold|lenticular|metal cover",
    re.I)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def load_key():
    txt = ENV.read_text()
    m = re.search(r'([0-9a-fA-F]{20,})', txt)
    if not m:
        raise RuntimeError("Comicvine key not found in comicvine_key.env")
    return m.group(1)


KEY = load_key()


def clean_desc(html, maxlen=600):
    if not html:
        return ""
    d = _TAG.sub(" ", html)
    d = (d.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
          .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " "))
    d = _WS.sub(" ", d).strip()
    if len(d) > maxlen:
        d = d[:maxlen].rsplit(" ", 1)[0] + "…"
    return d


def _get(url, params, tries=5):
    backoff = 3.0
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=50)
        except requests.RequestException as e:
            print(f"  net error {e}; retry {backoff:.0f}s")
            time.sleep(backoff); backoff *= 2; continue
        if r.status_code == 420 or r.status_code == 429:
            print(f"  {r.status_code} rate-limited; sleeping 30s")
            time.sleep(30); continue
        if r.status_code >= 500:
            time.sleep(backoff); backoff *= 2; continue
        r.raise_for_status()
        j = r.json()
        if j.get("error") not in ("OK", None):
            print(f"  API error: {j.get('error')}")
            return None
        return j
    return None


def load_json(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def save_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False))


# ---- search & detail (cached) ----
search_cache = {}
detail_cache = {}


def search_volumes(query, limit=12):
    key = f"{query}::{limit}"
    if key in search_cache:
        return search_cache[key]
    j = _get(f"{API}/search/", {"api_key": KEY, "format": "json",
             "resources": "volume", "query": query, "limit": limit})
    time.sleep(SLEEP)
    res = (j or {}).get("results", []) if j else []
    # keep only light fields in cache
    light = [{"id": v["id"], "name": v["name"],
              "publisher": (v.get("publisher") or {}).get("name", ""),
              "start_year": v.get("start_year"),
              "count_of_issues": v.get("count_of_issues")} for v in res]
    search_cache[key] = light
    return light


def list_publisher_volumes(pub_id, want, scan=600):
    """Get a publisher's notable volumes ranked by issue count (popularity proxy).
    Comicvine's /volumes/ publisher filter + sort are BROKEN, so instead:
      1. publisher DETAIL endpoint -> full id+name list for that publisher
      2. take the first `scan` of those ids, batch-fetch their issue counts
         (100 ids/request, pipe-separated) -- this works where sort doesn't
      3. rank by count_of_issues desc, return top `want`.
    Cached by pub_id."""
    ckey = f"pubvols:{pub_id}:{want}:{scan}"
    if ckey in search_cache:
        return search_cache[ckey]

    j = _get(f"{API}/publisher/4010-{pub_id}/",
             {"api_key": KEY, "format": "json", "field_list": "id,name,volumes"})
    time.sleep(SLEEP)
    vols = ((j or {}).get("results") or {}).get("volumes") or []
    # publisher.volumes is roughly chronological by id; scan a window and rank
    ids = [v["id"] for v in vols[:scan]]
    counts = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        jb = _get(f"{API}/volumes/", {"api_key": KEY, "format": "json",
                  "filter": "id:" + "|".join(str(x) for x in chunk), "limit": 100,
                  "field_list": "id,name,publisher,start_year,count_of_issues"})
        time.sleep(SLEEP)
        for v in (jb or {}).get("results", []):
            counts[v["id"]] = {"id": v["id"], "name": v["name"],
                               "publisher": (v.get("publisher") or {}).get("name", ""),
                               "start_year": v.get("start_year"),
                               "count_of_issues": v.get("count_of_issues") or 0}
    ranked = sorted(counts.values(), key=lambda v: -(v["count_of_issues"] or 0))
    out = ranked[:want]
    search_cache[ckey] = out
    return out


def get_detail(vid):
    sid = str(vid)
    if sid in detail_cache:
        return detail_cache[sid]
    j = _get(f"{API}/volume/4050-{vid}/", {"api_key": KEY, "format": "json",
             "field_list": DETAIL_FIELDS})
    time.sleep(SLEEP)
    v = (j or {}).get("results") if j else None
    if not v:
        detail_cache[sid] = None
        return None
    concepts_all = [c["name"] for c in (v.get("concepts") or [])]
    concepts_thematic = [n for n in concepts_all if not NOISE.search(n)]
    rec = {
        "id": v["id"],
        "name": v.get("name"),
        "publisher": (v.get("publisher") or {}).get("name", ""),
        "start_year": v.get("start_year"),
        "deck": v.get("deck") or "",
        "desc": clean_desc(v.get("description")),
        "concepts": concepts_all,
        "concepts_thematic": concepts_thematic,
        "n_chars": len(v.get("character_credits") or []),
        "count_of_issues": v.get("count_of_issues"),
    }
    detail_cache[sid] = rec
    return rec


def resolve_must(query, pub, year):
    hits = search_volumes(query, 12)
    def ok(v):
        if pub and pub.lower() not in (v["publisher"] or "").lower():
            return False
        if year and v["start_year"] and abs(int(v["start_year"]) - year) > 1:
            return False
        return True
    for v in hits:
        if ok(v):
            return v
    # relax year, keep publisher
    for v in hits:
        if not pub or pub.lower() in (v["publisher"] or "").lower():
            return v
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    global search_cache, detail_cache
    if not args.refresh:
        search_cache = load_json(SEARCH_CACHE, {})
        detail_cache = load_json(DETAIL_CACHE, {})

    chosen = {}       # id -> light record (with reason)
    must_report = []

    # STAGE A: must-include
    print("STAGE A: resolving must-include titles…")
    for query, pub, year in MUST_INCLUDE:
        v = resolve_must(query, pub, year)
        if v:
            chosen[v["id"]] = v
            must_report.append((query, v["id"], v["name"], v["publisher"], v["start_year"]))
            print(f"  [ok]   {query:32s} -> {v['id']} '{v['name']}' ({v['publisher']}, {v['start_year']})")
        else:
            must_report.append((query, None, None, None, None))
            print(f"  [MISS] {query}")
        save_json(SEARCH_CACHE, search_cache)

    # STAGE B: broaden by publisher (issue-count sorted)
    per_pub = max(20, (args.target - len(chosen)) // len(BROADEN_PUBLISHERS) + 8)
    print(f"\nSTAGE B: broadening ~{per_pub}/publisher across {len(BROADEN_PUBLISHERS)} publishers…")
    for pname, pid in BROADEN_PUBLISHERS.items():
        vols = list_publisher_volumes(pid, per_pub)
        added = 0
        for v in vols:
            if v["id"] in chosen:
                continue
            # prefer collected/multi-issue series; skip 1-issue floppies/specials
            ci = v.get("count_of_issues") or 0
            if ci < 2:
                continue
            chosen[v["id"]] = v
            added += 1
            if len(chosen) >= args.target:
                break
        print(f"  {pname:28s} +{added}  (total {len(chosen)})")
        save_json(SEARCH_CACHE, search_cache)
        if len(chosen) >= args.target:
            break

    # fetch DETAIL for all chosen (cached / resume-safe)
    print(f"\nFetching detail for {len(chosen)} volumes (cached ones skipped)…")
    corpus = []
    n = 0
    for vid in chosen:
        rec = get_detail(vid)
        n += 1
        if rec:
            corpus.append(rec)
        if n % 25 == 0:
            save_json(DETAIL_CACHE, detail_cache)
            print(f"  {n}/{len(chosen)} detailed")
    save_json(DETAIL_CACHE, detail_cache)
    save_json(SEARCH_CACHE, search_cache)
    save_json(CORPUS, corpus)

    print(f"\nDone. corpus={len(corpus)} volumes -> {CORPUS}")
    # must-include resolution summary
    miss = [m for m in must_report if m[1] is None]
    print(f"Must-include: {len(must_report)-len(miss)}/{len(must_report)} resolved"
          + (f"; MISSED: {[m[0] for m in miss]}" if miss else ""))


if __name__ == "__main__":
    main()
