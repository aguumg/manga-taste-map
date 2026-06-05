"""
resolve_labels.py — Resolve every labelled title to an AniList id and audit it.

For each (search, allowed_countries, approx_year, hint):
  1. AniList search query (type:MANGA) returns up to 10 candidates.
  2. Score each candidate: country must be in allowed set (hard filter);
     among survivors prefer the smallest |year - approx_year|, breaking ties by
     popularity. This guards against same-name collisions (e.g. a JP and a KR
     title sharing an English name).
  3. If no candidate passes the country filter, fall back to the most popular
     search hit but flag resolved=False for manual audit.

Any labelled title not already in data/corpus.parquet is fetched individually
and appended to the corpus (labels MUST be present in the feature matrix).

Outputs:
  data/labels_resolved.parquet   (id, group, label, resolved, matched fields)
  outputs/label_resolution.csv    (human-auditable)
  data/corpus.parquet             (augmented with any missing labelled titles)
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

from labels import STRONG_POS, NEG, MOD_POS, EXTRA_EXCLUDE

API = "https://graphql.anilist.co"
SLEEP = 2.0
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEARCH_QUERY = """
query($search:String){
  Page(page:1, perPage:10){
    media(search:$search, type:MANGA, sort:SEARCH_MATCH){
      id title{romaji english native} countryOfOrigin format
      averageScore popularity startDate{year}
    }
  }
}
"""

MEDIA_QUERY = """
query($id:Int){
  Media(id:$id, type:MANGA){
    id title{romaji english} countryOfOrigin format averageScore popularity
    genres startDate{year} tags{name rank isMediaSpoiler category}
  }
}
"""


def post(query, variables):
    backoff = 2.0
    for _ in range(6):
        r = requests.post(API, json={"query": query, "variables": variables}, timeout=60)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", backoff))
            print(f"  429; sleeping {wait}s")
            time.sleep(wait + 1)
            backoff *= 2
            continue
        r.raise_for_status()
        return r.json()["data"]
    raise RuntimeError("max retries")


def resolve_one(search, countries, approx_year, hint):
    cands = post(SEARCH_QUERY, {"search": search})["Page"]["media"]
    time.sleep(SLEEP)
    if not cands:
        return {"resolved": False, "reason": "no search hits"}

    # hard country filter
    in_country = [c for c in cands if c["countryOfOrigin"] in countries]
    pool = in_country if in_country else cands
    resolved = bool(in_country)

    def year_gap(c):
        y = (c["startDate"] or {}).get("year")
        if y is None or approx_year is None:
            return 9999
        return abs(y - approx_year)

    # prefer smallest year gap, then highest popularity
    pool = sorted(pool, key=lambda c: (year_gap(c), -(c["popularity"] or 0)))
    best = pool[0]
    if not resolved:
        reason = f"no candidate in countries {countries}; fell back to top hit"
    elif year_gap(best) > 6 and approx_year is not None:
        # country matched but year is far off -> flag for audit but keep
        resolved = False
        reason = f"country OK but year gap {year_gap(best)} > 6"
    else:
        reason = "ok"

    return {
        "resolved": resolved,
        "id": best["id"],
        "matched_romaji": (best["title"] or {}).get("romaji"),
        "matched_english": (best["title"] or {}).get("english"),
        "matched_country": best["countryOfOrigin"],
        "matched_year": (best["startDate"] or {}).get("year"),
        "popularity": best["popularity"],
        "reason": reason,
        "n_candidates": len(cands),
    }


def to_plain(obj):
    """Recursively convert numpy arrays/scalars to plain Python for json.dumps.
    Needed because parquet round-trips list-of-dict columns as ndarrays."""
    import numpy as _np
    if isinstance(obj, _np.ndarray):
        return [to_plain(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    return obj


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
            {"name": t["name"], "rank": t["rank"], "category": t["category"],
             "spoiler": bool(t["isMediaSpoiler"])}
            for t in (m["tags"] or [])
        ],
    }


def main():
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    corpus_ids = set(corpus["id"].tolist())

    groups = [("STRONG_POS", 1, STRONG_POS), ("NEG", 0, NEG),
              ("MOD_POS", None, MOD_POS), ("EXCLUDE", None, EXTRA_EXCLUDE)]

    records = []
    for gname, label, items in groups:
        for search, countries, year, hint in items:
            res = resolve_one(search, countries, year, hint)
            rec = {"group": gname, "label": label, "query": search,
                   "expected_country": "/".join(sorted(countries)),
                   "approx_year": year, "hint": hint, **res}
            records.append(rec)
            status = "OK" if res.get("resolved") else "FLAG"
            print(f"[{status}] {gname:10s} {search[:35]:35s} -> "
                  f"id={res.get('id')} {res.get('matched_english') or res.get('matched_romaji')} "
                  f"({res.get('matched_country')}, {res.get('matched_year')}) [{res.get('reason')}]")

    res_df = pd.DataFrame(records)

    # Fetch any labelled/excluded title missing from the corpus so it lands in X.
    needed_ids = [int(i) for i in res_df["id"].dropna().tolist()]
    missing = [i for i in needed_ids if i not in corpus_ids]
    if missing:
        print(f"\nFetching {len(missing)} labelled titles missing from corpus ...")
        extra = []
        for mid in missing:
            m = post(MEDIA_QUERY, {"id": mid})["Media"]
            extra.append(media_to_row(m))
            time.sleep(SLEEP)
        corpus = pd.concat([corpus, pd.DataFrame(extra)], ignore_index=True)
        corpus = corpus.drop_duplicates(subset="id", keep="first")
        corpus.to_parquet(DATA / "corpus.parquet", index=False)
        csv = corpus.copy()
        csv["tags"] = csv["tags"].apply(lambda x: json.dumps(to_plain(x)))
        csv["genres"] = csv["genres"].apply(lambda x: json.dumps(to_plain(x)))
        csv.to_csv(DATA / "corpus.csv", index=False)
        print(f"Corpus now {len(corpus)} titles.")

    res_df.to_parquet(DATA / "labels_resolved.parquet", index=False)
    res_df.to_csv(OUT / "label_resolution.csv", index=False)

    n_flag = (~res_df["resolved"].fillna(False)).sum()
    print(f"\nResolved {len(res_df)} entries; {n_flag} flagged for audit.")
    if n_flag:
        print("FLAGGED:")
        for _, r in res_df[~res_df["resolved"].fillna(False)].iterrows():
            print(f"  {r['group']} '{r['query']}' -> {r['reason']}")


if __name__ == "__main__":
    main()
