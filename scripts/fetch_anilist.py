"""
fetch_anilist.py — Pull a manga/manhwa/manhua corpus from the AniList GraphQL API.

Strategy:
  - Query type:MANGA sorted POPULARITY_DESC, 50 titles/page (perPage cap = 50).
  - Keep format == MANGA only (this still INCLUDES Korean manhwa and Chinese
    manhua, which AniList stores as format MANGA + countryOfOrigin KR/CN).
    Drop NOVEL and ONE_SHOT explicitly via the `format_not_in` filter.
  - Persist every title's tags as a list of {name, rank, category, spoiler}.
  - Robust to rate limits: honours Retry-After on HTTP 429 with exponential
    backoff, and sleeps SLEEP seconds between every request (AniList advertises
    ~90 req/min but the live X-RateLimit-Limit header reported 30/min, so we are
    conservative).

Outputs:
  data/corpus.parquet   (canonical, tags kept as Python objects)
  data/corpus.csv       (tags JSON-encoded into a string column)

Re-run is cached: if data/corpus.parquet exists we skip the network entirely
unless --refresh is passed.

Usage:
  python fetch_anilist.py            # use cache if present
  python fetch_anilist.py --refresh  # force re-fetch
  python fetch_anilist.py --pages 40 # ~2000 titles (default)
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://graphql.anilist.co"
SLEEP = 2.0          # seconds between requests (conservative for 30/min limit)
MAX_RETRIES = 6

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

PAGE_QUERY = """
query($page:Int,$perPage:Int){
  Page(page:$page, perPage:$perPage){
    pageInfo{ currentPage hasNextPage lastPage total }
    media(type:MANGA, sort:POPULARITY_DESC, format_not_in:[NOVEL, ONE_SHOT]){
      id
      title{ romaji english }
      countryOfOrigin
      format
      averageScore
      popularity
      genres
      startDate{ year }
      tags{ name rank isMediaSpoiler category }
    }
  }
}
"""


def post(query, variables):
    """POST with retry/backoff on 429 and transient 5xx."""
    backoff = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(API, json={"query": query, "variables": variables}, timeout=60)
        except requests.RequestException as e:
            print(f"  network error ({e}); retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 2
            continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", backoff))
            print(f"  429 rate-limited; sleeping {wait}s")
            time.sleep(wait + 1)
            backoff *= 2
            continue
        if r.status_code >= 500:
            print(f"  {r.status_code} server error; retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 2
            continue
        r.raise_for_status()
        payload = r.json()
        if "errors" in payload and payload["errors"]:
            raise RuntimeError(payload["errors"])
        return payload["data"]
    raise RuntimeError("exceeded max retries")


def media_to_row(m):
    return {
        "id": m["id"],
        "romaji": (m["title"] or {}).get("romaji"),
        "english": (m["title"] or {}).get("english"),
        "country": m["countryOfOrigin"],
        "format": m["format"],
        "averageScore": m["averageScore"],
        "popularity": m["popularity"],
        "year": (m["startDate"] or {}).get("year"),
        "genres": m["genres"] or [],
        "tags": [
            {
                "name": t["name"],
                "rank": t["rank"],
                "category": t["category"],
                "spoiler": bool(t["isMediaSpoiler"]),
            }
            for t in (m["tags"] or [])
        ],
    }


def fetch_corpus(pages):
    rows, seen = [], set()
    for page in range(1, pages + 1):
        data = post(PAGE_QUERY, {"page": page, "perPage": 50})
        pg = data["Page"]
        for m in pg["media"]:
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            rows.append(media_to_row(m))
        info = pg["pageInfo"]
        print(f"page {page}/{pages}  got {len(pg['media'])}  total so far {len(rows)}")
        if not info["hasNextPage"]:
            print("API reports no further pages; stopping.")
            break
        time.sleep(SLEEP)
    return rows


def save(rows):
    df = pd.DataFrame(rows)
    df.to_parquet(DATA / "corpus.parquet", index=False)
    csv = df.copy()
    csv["tags"] = csv["tags"].apply(json.dumps)
    csv["genres"] = csv["genres"].apply(json.dumps)
    csv.to_csv(DATA / "corpus.csv", index=False)
    return df


def country_breakdown(df):
    vc = df["country"].value_counts(dropna=False).to_dict()
    label = {"JP": "JP (manga)", "KR": "KR (manhwa)", "CN": "CN (manhua)"}
    print("\nCorpus country breakdown:")
    for k, v in sorted(vc.items(), key=lambda kv: -kv[1]):
        print(f"  {label.get(k, k)}: {v}")
    print(f"  TOTAL: {len(df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=40, help="50 titles/page (40 => ~2000)")
    ap.add_argument("--refresh", action="store_true", help="ignore cache, re-fetch")
    args = ap.parse_args()

    cache = DATA / "corpus.parquet"
    if cache.exists() and not args.refresh:
        df = pd.read_parquet(cache)
        print(f"Loaded cached corpus: {len(df)} titles (use --refresh to re-fetch).")
        country_breakdown(df)
        return

    print(f"Fetching ~{args.pages * 50} titles from AniList ...")
    rows = fetch_corpus(args.pages)
    df = save(rows)
    print(f"\nSaved {len(df)} titles to {cache} and corpus.csv")
    country_breakdown(df)


if __name__ == "__main__":
    main()
