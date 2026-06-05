"""
fetch_synonyms.py — Batch-fetch AniList `synonyms` (alternate/scanlation titles)
for every corpus title, so the app's search can match community names like
"Eternally Regressing Knight" -> "The Knight Only Lives Today".

Batched 50 ids/request via Page(media(id_in:[...])) — ~40 requests, fast.
Cached to data/synonyms.json as { "<id>": ["alt title", ...] }. Re-runs skip
the network unless --refresh.

Usage:
  python fetch_synonyms.py            # use cache if present
  python fetch_synonyms.py --refresh  # re-hit AniList
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://graphql.anilist.co"
SLEEP = 1.0
BATCH = 50
MAX_RETRIES = 6

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "synonyms.json"

QUERY = """
query($ids:[Int]){
  Page(perPage:50){
    media(id_in:$ids, type:MANGA){ id synonyms }
  }
}
"""


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if OUT.exists() and not args.refresh:
        d = json.loads(OUT.read_text())
        n = sum(1 for v in d.values() if v)
        print(f"Cached synonyms: {len(d)} titles ({n} with >=1). --refresh to re-fetch.")
        return

    corpus = pd.read_parquet(DATA / "corpus.parquet").drop_duplicates("id")
    ids = [int(i) for i in corpus["id"].tolist()]
    chunks = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]
    print(f"{len(ids)} titles -> {len(chunks)} batched requests (50/req).")

    syn = {}
    t0 = time.time()
    for ci, chunk in enumerate(chunks, 1):
        data = post({"ids": chunk})
        media = (data or {}).get("Page", {}).get("media", []) if data else []
        got = set()
        for m in media:
            got.add(m["id"])
            syn[str(m["id"])] = m.get("synonyms") or []
        for mid in chunk:
            if mid not in got:
                syn.setdefault(str(mid), [])
        if ci % 10 == 0 or ci == len(chunks):
            print(f"  batch {ci}/{len(chunks)} (collected {len(syn)})")
        time.sleep(SLEEP)

    OUT.write_text(json.dumps(syn, ensure_ascii=False))
    n_with = sum(1 for v in syn.values() if v)
    print(f"\nDone in {time.time()-t0:.1f}s. {len(syn)} titles, {n_with} with >=1 synonym.")


if __name__ == "__main__":
    main()
