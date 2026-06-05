"""
fetch_staff.py — Pull artist/author staff edges AND synopsis for every corpus
title, BATCHED 50 ids/request via Page(media(id_in:[...])) — ~40 requests for
the whole corpus, finishes in well under a minute (NOT one request per title).

For each title in data/corpus.parquet, fetch:
  - staff edges, keeping only the roles that identify who DREW or WROTE it:
      role contains "Story & Art"  -> artist (the hand) AND author
      role contains "Art"          -> artist  (NOT "Touch-up Art"/"Lettering")
      role contains "Story"        -> author / writer
    Translators, letterers, editors, supervisors, touch-up are excluded.
  - description(asHtml:false) -> synopsis. HTML/markdown stripped to plain text,
    truncated to ~600 chars + "…".

Outputs (merged into JSON the HTML builder embeds):
  data/staff.json
    {
      "title_artists": { "<id>": ["Kentarou Miura", ...] },   # artists only
      "title_authors": { "<id>": ["Makoto Yukimura", ...] },  # writers
      "artist_titles": { "Kentarou Miura": [<id>, ...] },      # reverse idx
      "synopsis":      { "<id>": "plain-text synopsis…" },
      "fetched_ids":   [<id>, ...],
      "no_staff_ids":  [<id>, ...],
      "no_desc_ids":   [<id>, ...]
    }

Caching: data/staff_raw.json holds per-title {staff:[[role,name]...], desc:str}.
Re-runs resume / skip already-fetched ids. --refresh wipes and re-fetches.

Usage:
  python fetch_staff.py            # resume; only fetch missing ids
  python fetch_staff.py --refresh  # re-fetch all
  python fetch_staff.py --limit 50 # debug: only first 50 missing
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://graphql.anilist.co"
SLEEP = 1.0          # one ~1s sleep per BATCH (50 ids), well within rate limit
MAX_RETRIES = 6
DESC_MAX = 600
BATCH = 50           # AniList Page perPage cap

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "staff_raw.json"        # id -> {"staff":[[role,name]...], "desc":str}
OUT = DATA / "staff.json"

# Batched: up to 50 media per request via id_in. perPage:50 returns all matches.
QUERY = """
query($ids:[Int]){
  Page(perPage:50){
    media(id_in:$ids, type:MANGA){
      id
      description(asHtml:false)
      staff(perPage:20){ edges{ role node{ name{ full } } } }
    }
  }
}
"""

_ART_NEG = re.compile(r"touch-?up|letter|assist|background art|color", re.I)
_ART_POS = re.compile(r"\bart\b|story\s*&\s*art|illustrat", re.I)
_STORY_POS = re.compile(r"\bstory\b|\boriginal\b|\bauthor\b|\bwriter\b|original creator", re.I)
_STORY_NEG = re.compile(r"storyboard", re.I)

_TAG = re.compile(r"<[^>]+>")
_MD = re.compile(r"(\*\*|__|~~|\[|\]|\(https?://[^)]+\))")
_WS = re.compile(r"\s+")


def classify(role):
    is_art = bool(_ART_POS.search(role)) and not _ART_NEG.search(role)
    is_story = bool(_STORY_POS.search(role)) and not _STORY_NEG.search(role)
    return is_art, is_story


def clean_desc(d):
    if not d:
        return ""
    d = _TAG.sub(" ", d)                  # strip HTML tags
    d = d.replace("<br>", " ").replace("&nbsp;", " ")
    d = (d.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
          .replace("&quot;", '"').replace("&#039;", "'"))
    d = _MD.sub("", d)                    # light markdown strip
    d = _WS.sub(" ", d).strip()
    if len(d) > DESC_MAX:
        d = d[:DESC_MAX].rsplit(" ", 1)[0] + "…"
    return d


def post(variables):
    backoff = 2.0
    for _ in range(MAX_RETRIES):
        try:
            r = requests.post(API, json={"query": QUERY, "variables": variables}, timeout=60)
        except requests.RequestException as e:
            print(f"  net error {e}; retry {backoff:.0f}s")
            time.sleep(backoff); backoff *= 2; continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", backoff))
            print(f"  429; sleep {wait}s"); time.sleep(wait + 1); backoff *= 2; continue
        if r.status_code >= 500:
            print(f"  {r.status_code}; retry {backoff:.0f}s")
            time.sleep(backoff); backoff *= 2; continue
        r.raise_for_status()
        payload = r.json()
        if payload.get("errors"):
            return None
        return payload["data"]
    raise RuntimeError("max retries")


def load_raw():
    if RAW.exists():
        raw = json.loads(RAW.read_text())
        # migrate any old staff-only entries ([[role,name]...]) to new dict shape
        fixed = {}
        for k, v in raw.items():
            if isinstance(v, list):
                fixed[k] = {"staff": v, "desc": None}   # desc unknown -> refetch
            else:
                fixed[k] = v
        return fixed
    return {}


def save_raw(raw):
    RAW.write_text(json.dumps(raw, ensure_ascii=False))


def needs_fetch(entry):
    # entry missing, or has no desc captured yet (old staff-only cache)
    return entry is None or entry.get("desc") is None


def build_indexes(raw):
    title_artists, title_authors, artist_titles = {}, {}, {}
    synopsis = {}
    no_staff, no_desc = [], []
    for sid, entry in raw.items():
        edges = entry.get("staff", [])
        desc = entry.get("desc") or ""
        arts, auths = [], []
        for role, name in edges:
            is_art, is_story = classify(role)
            if is_art and name not in arts:
                arts.append(name)
            if is_story and name not in auths:
                auths.append(name)
        if arts:
            title_artists[sid] = arts
            for a in arts:
                artist_titles.setdefault(a, [])
                if int(sid) not in artist_titles[a]:
                    artist_titles[a].append(int(sid))
        if auths:
            title_authors[sid] = auths
        if not arts and not auths:
            no_staff.append(int(sid))
        if desc:
            synopsis[sid] = desc
        else:
            no_desc.append(int(sid))
    return title_artists, title_authors, artist_titles, synopsis, no_staff, no_desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corpus = pd.read_parquet(DATA / "corpus.parquet").drop_duplicates("id")
    ids = [int(i) for i in corpus["id"].tolist()]

    raw = {} if args.refresh else load_raw()
    todo = [i for i in ids if needs_fetch(raw.get(str(i)))]
    if args.limit:
        todo = todo[:args.limit]
    chunks = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(ids)} titles; {len(raw)} cached; fetching {len(todo)} "
          f"missing in {len(chunks)} batched requests (50/req).")

    t0 = time.time()
    for ci, chunk in enumerate(chunks, 1):
        data = post({"ids": chunk})
        media = (data or {}).get("Page", {}).get("media", []) if data else []
        got = set()
        for m in media:
            mid = m["id"]
            got.add(mid)
            desc = clean_desc(m.get("description"))
            edges = []
            if m.get("staff") and m["staff"].get("edges"):
                for e in m["staff"]["edges"]:
                    nm = (e["node"]["name"] or {}).get("full")
                    if nm:
                        edges.append([e["role"] or "", nm])
            raw[str(mid)] = {"staff": edges, "desc": desc}
        # ids the API didn't return for this chunk: record empty so we don't loop
        for mid in chunk:
            if mid not in got:
                raw[str(mid)] = {"staff": [], "desc": ""}
        save_raw(raw)
        print(f"  batch {ci}/{len(chunks)}: requested {len(chunk)}, "
              f"got {len(media)} (cached {len(raw)})")
        time.sleep(SLEEP)
    print(f"  fetch wall time: {time.time()-t0:.1f}s")
    save_raw(raw)

    ta, tau, at, syn, no_staff, no_desc = build_indexes(raw)
    out = {
        "title_artists": ta,
        "title_authors": tau,
        "artist_titles": at,
        "synopsis": syn,
        "fetched_ids": [int(k) for k in raw.keys()],
        "no_staff_ids": no_staff,
        "no_desc_ids": no_desc,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False))
    print(f"\nDone. {len(raw)} fetched.")
    print(f"  titles with >=1 artist: {len(ta)}")
    print(f"  titles with NO story/art staff at all: {len(no_staff)}")
    print(f"  distinct artists: {len(at)}")
    print(f"  titles with synopsis: {len(syn)}  |  no description: {len(no_desc)}")
    for probe in (30002, 30642):
        a = ta.get(str(probe), [])
        print(f"  id {probe} artists: {a}")
        for nm in a:
            print(f"    {nm} -> {len(at.get(nm, []))} title(s)")


if __name__ == "__main__":
    main()
