#!/usr/bin/env python3
"""Stream-extract the useful slice of the 3.5GB GCD MySQL dump into SQLite.

We don't need MySQL: parse INSERT ... VALUES (...),(...); statements ourselves and
write to a small SQLite DB. Two phases:
  phase 'series' : gcd_publisher + gcd_series (English) -> catalog            (fast)
  phase 'genre'  : gcd_issue (issue->series) + gcd_story (genre/synopsis)
                   -> aggregate genre + a sample synopsis per series          (~mins)

Usage: parse_gcd.py series   |   parse_gcd.py genre
Output: data/gcd.sqlite
"""
import sys, re, sqlite3, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DUMP = ROOT / "grand_comics_database" / "2026-06-01.sql"
DB = ROOT / "data" / "gcd.sqlite"
ENGLISH_LANG_ID = 25  # GCD language_id for English (seen in sample rows)


def iter_rows(table):
    """Yield each value-tuple (as a list of python scalars) from INSERT INTO `table`."""
    prefix = f"INSERT INTO `{table}` VALUES "
    with open(DUMP, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith(prefix):
                continue
            body = line[len(prefix):].rstrip().rstrip(";")
            yield from _split_tuples(body)


def _split_tuples(body):
    """Split (a,b),(c,d) respecting '...' strings with backslash escapes."""
    i, n = 0, len(body)
    while i < n:
        if body[i] != "(":
            i += 1
            continue
        i += 1
        fields, cur, in_str, esc = [], [], False, False
        while i < n:
            c = body[i]
            if in_str:
                if esc:
                    cur.append(c); esc = False
                elif c == "\\":
                    cur.append(c); esc = True
                elif c == "'":
                    in_str = False
                else:
                    cur.append(c)
            else:
                if c == "'":
                    in_str = True
                elif c == ",":
                    fields.append("".join(cur)); cur = []
                elif c == ")":
                    fields.append("".join(cur)); i += 1; break
                else:
                    cur.append(c)
            i += 1
        yield fields


def _to_int(s):
    try:
        return int(s)
    except Exception:
        return None


def phase_series():
    con = sqlite3.connect(DB); cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS publisher")
    cur.execute("CREATE TABLE publisher(id INTEGER PRIMARY KEY, name TEXT)")
    np = 0
    for r in iter_rows("gcd_publisher"):
        # gcd_publisher: id,name,... (name is field index 1)
        if len(r) > 1:
            cur.execute("INSERT OR REPLACE INTO publisher VALUES(?,?)", (_to_int(r[0]), r[1]))
            np += 1
    con.commit()
    print(f"publishers: {np}")

    cur.execute("DROP TABLE IF EXISTS series")
    cur.execute("""CREATE TABLE series(id INTEGER PRIMARY KEY, name TEXT, year_began INT,
                   publisher_id INT, language_id INT, issue_count INT, notes TEXT)""")
    # gcd_series field order (from schema): 0 id,1 name,2 sort_name,3 format,4 year_began,
    # 5 yb_unc,6 year_ended,7 ye_unc,8 pub_dates,9 first_issue,10 last_issue,11 is_current,
    # 12 publisher_id,13 country_id,14 language_id,15 tracking_notes,16 notes, ...,18 issue_count
    ns = nse = 0
    for r in iter_rows("gcd_series"):
        if len(r) < 19:
            continue
        ns += 1
        lang = _to_int(r[14])
        if lang != ENGLISH_LANG_ID:
            continue
        cur.execute("INSERT OR REPLACE INTO series VALUES(?,?,?,?,?,?,?)",
                    (_to_int(r[0]), r[1], _to_int(r[4]), _to_int(r[12]), lang,
                     _to_int(r[18]), r[16][:500]))
        nse += 1
        if nse % 50000 == 0:
            con.commit(); print(f"  ...{nse} english series")
    con.commit()
    print(f"series total: {ns} | english: {nse}")
    # quick look: biggest English series by issue_count joined to publisher
    print("\nTop 15 English series by issue_count:")
    for name, yr, pub, ic in cur.execute("""SELECT s.name, s.year_began, p.name, s.issue_count
            FROM series s LEFT JOIN publisher p ON s.publisher_id=p.id
            ORDER BY s.issue_count DESC LIMIT 15"""):
        print(f"  {ic:>5}  {name} ({yr}) — {pub}")
    con.close()
    print(f"\nsaved -> {DB}")


def phase_genre():
    con = sqlite3.connect(DB); cur = con.cursor()
    # issue -> series map (gcd_issue: 0 id,1 number,...; series_id position varies — find via schema)
    # gcd_issue schema: id,number,volume,no_volume,display_volume_with_number,series_id,...
    # series_id is field index 5 in current schema.
    print("building issue->series map (parsing gcd_issue, ~515MB)...")
    issue2series = {}
    for r in iter_rows("gcd_issue"):
        if len(r) > 5:
            iid, sid = _to_int(r[0]), _to_int(r[5])
            if iid is not None and sid is not None:
                issue2series[iid] = sid
    print(f"  issues mapped: {len(issue2series)}")

    # gcd_story: 0 id,1 title,2 title_inferred,3 feature,4 sequence,5 page_count,6 issue_id,
    # 7 script,8 pencils,9 inks,10 colors,11 letters,12 editing,13 genre,14 characters,15 synopsis
    print("aggregating genre + synopsis from gcd_story (~1GB)...")
    from collections import defaultdict, Counter
    genre_by_series = defaultdict(Counter)
    synopsis_by_series = {}
    n = 0
    for r in iter_rows("gcd_story"):
        if len(r) < 16:
            continue
        iid = _to_int(r[6]); genre = r[13].strip(); syn = r[15].strip()
        sid = issue2series.get(iid)
        if sid is None:
            continue
        if genre:
            for g in re.split(r"[;,]", genre):
                g = g.strip().lower()
                if g:
                    genre_by_series[sid][g] += 1
        if syn and len(syn) > len(synopsis_by_series.get(sid, "")):
            synopsis_by_series[sid] = syn[:1500]
        n += 1
        if n % 1000000 == 0:
            print(f"  ...{n} stories")
    print(f"  stories scanned: {n}")

    cur.execute("DROP TABLE IF EXISTS series_genre")
    cur.execute("CREATE TABLE series_genre(series_id INT PRIMARY KEY, genres TEXT, synopsis TEXT)")
    for sid, gc in genre_by_series.items():
        top = "; ".join(f"{g}({c})" for g, c in gc.most_common(6))
        cur.execute("INSERT OR REPLACE INTO series_genre VALUES(?,?,?)",
                    (sid, top, synopsis_by_series.get(sid, "")))
    con.commit()
    print(f"series with genre data: {len(genre_by_series)}")
    con.close()
    print(f"saved -> {DB}")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "series"
    {"series": phase_series, "genre": phase_genre}[phase]()
