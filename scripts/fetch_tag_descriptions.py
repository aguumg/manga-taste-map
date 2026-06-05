"""
fetch_tag_descriptions.py — One AniList call for all tag descriptions.

`query { MediaTagCollection { id name description category } }` returns every
AniList media tag (~423) with its official description in a single request.
Cached to data/tag_descriptions.json as { tagName: description }. The HTML
builder embeds only the descriptions for the ~254 tags actually used.

Usage:
  python fetch_tag_descriptions.py            # use cache if present
  python fetch_tag_descriptions.py --refresh  # re-hit AniList
"""
import argparse
import json
from pathlib import Path

import requests

API = "https://graphql.anilist.co"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "tag_descriptions.json"

QUERY = "query { MediaTagCollection { id name description category } }"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if OUT.exists() and not args.refresh:
        d = json.loads(OUT.read_text())
        print(f"Cached tag descriptions: {len(d)} tags (use --refresh to re-fetch).")
        return

    r = requests.post(API, json={"query": QUERY}, timeout=60)
    r.raise_for_status()
    tags = r.json()["data"]["MediaTagCollection"]
    desc = {t["name"]: (t.get("description") or "") for t in tags}
    OUT.write_text(json.dumps(desc, ensure_ascii=False))
    n_desc = sum(1 for v in desc.values() if v)
    print(f"Fetched {len(desc)} tags ({n_desc} with description) -> {OUT}")


if __name__ == "__main__":
    main()
