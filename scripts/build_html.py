"""
build_html.py — Emit the single-file interactive taste-map app (v2).

Self-contained: inline Plotly + inline data + localStorage. No runtime network.

Embedded for all ~2007 titles:
  id, romaji, english, country, averageScore, popularity,
  x,y (FIXED PCA layout), s (supervised score), v (sparse tag vector),
  tt (top-5 tag string), g (genres list), syn (synopsis, may be ""),
  artists (list of names), authors (list of names)
Plus, embedded once:
  tags[]            tagIndex -> tagName (254)
  tagCat{}          tagName -> AniList category
  tagMeta[]         ranked tag-vote metadata (freq, discriminativeness, loved/dis counts)
  artistTitles{}    artist name -> [title ids]   (reverse index)
  genres[]          the 19 AniList genres
  seed{}            v1-style pre-seed (overall verdict) for first run

The feature matrix is rebuilt via build_feature_matrix from model.py so the JS
cosine == the Python centroid cosine EXACTLY.

Output: outputs/taste_map_app.html   (overwrites; static taste_map.html kept)
"""
import json
import re as _re_shelf
from pathlib import Path

import numpy as np
import pandas as pd

from model import build_feature_matrix, BERSERK_ID


def _norm_title(s):
    s = (s or "").lower()
    s = _re_shelf.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def _slug(s):
    return _re_shelf.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


# Reading Shelf: arbitrary reading-queue seed (many titles are anime/comics NOT
# in the corpus). Statuses: toread / reading / paused / done. Each entry has an
# editable note + optional read URL. Build-time we attach `cid` (corpus title id)
# and/or `wk` (Western key) when the title matches, so the app can offer "→ map".
SHELF_SEED = [
  # --- DONE / caught-up (his own) ---
  {"title":"Overgeared","status":"done","source":"mine","note":"Finished the webtoon; text novel exists but not the same","url":"https://w99.overgeared.club/overgeared-chapter-290/"},
  {"title":"Eternally Regressing Knight","status":"done","source":"mine","note":"Caught up (~ch73)","url":"https://w2.regressingknight.com/eternally-regressing-knight-chapter-73/"},
  {"title":"Pick Me Up: Infinite Gacha","status":"done","source":"mine","note":"Caught up (~ch165)","url":"https://w1.pickmeupgacha.com/pick-me-up-infinite-gacha-chapter-165/"},
  {"title":"Surviving the Game as a Barbarian","status":"done","source":"mine","note":"Caught up (~ch120)","url":"https://asuracomic.net/series/surviving-the-game-as-a-barbarian-ffa1cc42/chapter/120"},
  {"title":"Murim RPG Simulation","status":"done","source":"mine","note":"S2 finale","url":"https://www.webtoons.com/en/action/murim-rpg-simulation/s2-episode-144-season-2-finale/viewer?title_no=3779&episode_no=144"},
  {"title":"Taming Master","status":"done","source":"mine","note":"Caught up (~ch164); renews ~weekly","url":"https://xbato.com/title/90733-taming-master-official/3780444-ch_164"},
  {"title":"The Lone Necromancer","status":"done","source":"mine","note":"Caught up (~ch203)","url":"https://w1.thelonenecromancer.site/manga/the-lone-necromancer-chapter-203/"},
  {"title":"Berserk","status":"done","source":"mine","note":"All-time favorite","url":""},
  {"title":"Claymore","status":"done","source":"mine","note":"All-time favorite","url":""},
  {"title":"Blade of the Immortal","status":"done","source":"mine","note":"All-time favorite (La espada del inmortal)","url":""},
  {"title":"Vagabond","status":"done","source":"mine","note":"Loved — like the early bloody mangas","url":""},
  {"title":"Solo Leveling","status":"done","source":"mine","note":"Favorite","url":""},
  {"title":"Demon Slayer","status":"done","source":"mine","note":"Favorite","url":""},
  {"title":"Hell's Paradise","status":"done","source":"mine","note":"Watched all; caught up at S3 (not out)","url":""},
  {"title":"Ajin: Demi-Human","status":"done","source":"mine","note":"Anime complete — very good","url":""},
  {"title":"Attack on Titan","status":"done","source":"mine","note":"Anime complete — very good","url":""},
  # --- READING (in progress) ---
  {"title":"The Greatest Estate Developer","status":"reading","source":"mine","note":"~ch57","url":"https://greatestestatedeveloper.org/manga/the-greatest-estate-developer-chapter-57/"},
  {"title":"Skeleton Soldier Couldn't Protect the Dungeon","status":"reading","source":"mine","note":"~ch27","url":"https://skeleton-soldier.online/manga/skeleton-soldier-couldnt-protect-the-dungeon-chapter-27/"},
  {"title":"SSS-Class Suicide Hunter","status":"reading","source":"mine","note":"~ch23","url":"https://www.toongod.org/webtoon/sss-class-suicide-hunter/chapter-23/"},
  {"title":"Solo Max-Level Newbie","status":"reading","source":"mine","note":"~ch14","url":"https://solomaxlevel.club/manga/solo-max-level-newbie-chapter-14/"},
  {"title":"Reincarnation of the Suicidal Battle God","status":"reading","source":"mine","note":"~ch4 (early — watch for OP-MC boredom)","url":"https://reincarnationofthesuicidalbattlegod.club/manga/reincarnation-of-the-suicidal-battle-god-chapter-4/"},
  # --- PAUSED / inconclusive ---
  {"title":"Memorize","status":"paused","source":"mine","note":"Stopped ~ch1-3, inconclusive","url":"https://xbato.com/title/99274-memorize-official/1936054-ch_3"},
  {"title":"Omniscient Reader's Viewpoint","status":"paused","source":"mine","note":"Started, got bored — but reputedly one of the best; revisit","url":"https://www.webtoons.com/en/action/omniscient-reader/list?title_no=2154"},
  {"title":"Bleach","status":"paused","source":"mine","note":"Anime: loved it, dropped when MC goes to hell/dies/revives-from-future; wanted more fights","url":""},
  {"title":"Lone Wolf and Cub","status":"paused","source":"mine","note":"Few chapters, dropped","url":""},
  {"title":"Jormungand","status":"paused","source":"mine","note":"Anime: liked it, stopped","url":""},
  {"title":"Goblin Slayer","status":"paused","source":"mine","note":"Anime: good but got bored","url":""},
  {"title":"Golden Kamuy","status":"paused","source":"mine","note":"1 ep, too talky","url":""},
  {"title":"Absolute Sword Sense","status":"paused","source":"mine","note":"~ch23; loved talking-swords premise, bored in barbarian-master training arc","url":"https://asuracomic.net/series/absolute-sword-sense-a96416f4/chapter/23"},
  {"title":"Nano Machine","status":"paused","source":"mine","note":"Dropped — MC too OP, everything too easy -> boring [flagged dislike candidate]","url":"https://asuracomic.net/series/nano-machine-af0c03db/chapter/136"},
  {"title":"Golgo 13","status":"paused","source":"mine","note":"Few eps; old, MC too OP, too easy [flagged dislike candidate]","url":""},
  {"title":"Made in Abyss","status":"paused","source":"mine","note":"Expected traumatic/dark, found it cozy/tender — mismatch [flagged dislike candidate]","url":""},
  # --- TO-READ: friend recs ---
  {"title":"Steins;Gate","status":"toread","source":"friend","note":"Friend rec — time travel","url":""},
  {"title":"Frieren: Beyond Journey's End","status":"toread","source":"friend","note":"Friend rec","url":""},
  {"title":"Fate/Zero","status":"toread","source":"friend","note":"Friend rec — manga adaptation","url":"https://es.novelcool.com/chapter/Cap-tulo-1/2880616/"},
  {"title":"Puella Magi Madoka Magica","status":"toread","source":"friend","note":"Friend rec","url":""},
  {"title":"Saya no Uta","status":"toread","source":"friend","note":"Friend rec — manga 'Song of Saya' (Nitroplus horror)","url":"https://batcave.biz/25689-song-of-saya-2010.html"},
  # --- TO-READ: CBR viking list ---
  {"title":"Eternal","status":"toread","source":"cbr-viking","note":"CBR viking — shieldmaiden vs warlock, battle + family (Lindsay/Zawadzki)","url":""},
  {"title":"Viking: The Long Cold Fire","status":"toread","source":"cbr-viking","note":"CBR viking — two brothers, 9th-c, dark humor","url":""},
  {"title":"Sword Daughter","status":"toread","source":"cbr-viking","note":"CBR viking — Brian Wood, shieldmaiden vengeance","url":""},
  {"title":"Helheim","status":"toread","source":"cbr-viking","note":"CBR viking — Cullen Bunn, horror, zombies/witches","url":""},
  {"title":"Black Road","status":"toread","source":"cbr-viking","note":"CBR viking — Brian Wood, Norway Christianization, vengeance","url":""},
  {"title":"The Darkness: Lodbrok's Hand","status":"toread","source":"cbr-viking","note":"CBR viking — mythic Darkness one-shot","url":""},
  {"title":"Heathen","status":"toread","source":"cbr-viking","note":"CBR viking — Norse fantasy, shieldmaiden + valkyrie","url":""},
  # --- TO-READ: curated from 8 best-of lists, ranked STRONG fit (2026-06-06) ---
  {"title":"Oyasumi PunPun","status":"toread","source":"curated","note":"TOP BET — depression spiral, abuse survivor→abuser; pure suffering, zero power fantasy.","url":""},
  {"title":"Monster","status":"toread","source":"curated","note":"Urasawa slow-burn dread; dark moral weight; finished masterpiece.","url":""},
  {"title":"Homunculus","status":"toread","source":"curated","note":"Psychological body-horror; MC unravels; devastating, finished.","url":""},
  {"title":"Dorohedoro","status":"toread","source":"curated","note":"Gory dark-fantasy in a brutal megacity; finished.","url":""},
  {"title":"Parasyte","status":"toread","source":"curated","note":"Body-horror; MC suffers and grows the hard way; tight, finished.","url":""},
  {"title":"I Am a Hero","status":"toread","source":"curated","note":"Best-in-class zombie horror; unstable MC; real terror.","url":""},
  {"title":"Blood on the Tracks","status":"toread","source":"curated","note":"Creeping psychological horror of an abusive mother.","url":""},
  {"title":"Sweet Home","status":"toread","source":"curated","note":"Monster apocalypse; suicidal recluse loses everything; gory, finished.","url":""},
  {"title":"Uzumaki","status":"toread","source":"curated","note":"Ito spiral cosmic horror; grotesque, inescapable, finished.","url":""},
  {"title":"Second Life Ranker","status":"toread","source":"curated","note":"Tower-climb revenge; twin betrayed/killed; earned grind.","url":""},
  {"title":"Tomie","status":"toread","source":"curated","note":"Ito horror; murder/regeneration; disturbing.","url":""},
  {"title":"Gantz","status":"toread","source":"curated","note":"Brutal alien death-game; high mortality, real stakes.","url":""},
  # --- READING: own picks from flights (2026-06-06) ---
  {"title":"God Level Assassin","status":"reading","source":"mine","note":"Loved — read to ch91 (OP MC but I enjoy it).","url":"https://manhuaus.com/manga/god-level-assassin-im-the-shadow/chapter-91/"},
  {"title":"Long Way of the Warrior","status":"reading","source":"mine","note":"Read to ch150.","url":"https://manhwatop.com/manga-tag/long-way-of-the-warrior-chapters/"},
]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# The reader's real exported ratings (43 entries) seed My Library by default.
USER_SEED = DATA / "user_library_seed.json"


def s(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val)


def top_tags(tags, n=5):
    ts = sorted(tags, key=lambda t: -(t["rank"] or 0))
    return ", ".join(t["name"] for t in ts[:n])


def build_seed(res):
    """Seed My Library from the reader's REAL exported ratings
    (data/user_library_seed.json: 43 entries). rating -> overall (v2). Axes/note
    left empty for the user to fill in. Falls back to the label table only if the
    export is missing."""
    if USER_SEED.exists():
        payload = json.loads(USER_SEED.read_text())
        entries = payload.get("entries", payload)
        seed = {}
        for k, v in entries.items():
            seed[int(k)] = {"read": bool(v.get("read", True)),
                            "rating": v.get("rating")}  # loved/liked/meh/disliked/None
        return seed
    # fallback: derive from labels (STRONG_POS->loved, NEG->disliked)
    seed = {}
    for _, r in res.iterrows():
        if pd.isna(r["id"]):
            continue
        tid = int(r["id"])
        if r["group"] == "STRONG_POS":
            seed[tid] = {"read": True, "rating": "loved"}
        elif r["group"] == "NEG":
            seed[tid] = {"read": True, "rating": "disliked"}
    return seed


def build_tag_meta(Xdf, feat_cols, X, feat_row, res):
    """Rank tags for the voting UI: combined = frequency * discriminativeness
    between the user's Loved (STRONG_POS) and Disliked (NEG) sets, plus per-tag
    loved/disliked carrier counts and category. Only real tags (not the
    demo::/genre:: binary helper columns) are votable."""
    n = X.shape[0]
    loved_ids = [int(i) for i in res[res.group == "STRONG_POS"].dropna(subset=["id"])["id"]]
    dis_ids = [int(i) for i in res[res.group == "NEG"].dropna(subset=["id"])["id"]]
    loved_rows = [feat_row[i] for i in loved_ids if i in feat_row]
    dis_rows = [feat_row[i] for i in dis_ids if i in feat_row]
    Xl, Xd = X[loved_rows], X[dis_rows]

    meta = []
    for j, name in enumerate(feat_cols):
        if name.startswith("demo::") or name.startswith("genre::"):
            continue
        col = X[:, j]
        freq = float((col > 0).mean())                    # corpus frequency
        mean_l = float(Xl[:, j].mean()) if len(Xl) else 0.0
        mean_d = float(Xd[:, j].mean()) if len(Xd) else 0.0
        disc = abs(mean_l - mean_d)                        # separates fav vs dislike
        loved_ct = int((Xl[:, j] > 0).sum()) if len(Xl) else 0
        dis_ct = int((Xd[:, j] > 0).sum()) if len(Xd) else 0
        meta.append({
            "i": j, "name": name,
            "score": round(freq * disc, 5),
            "freq": round(freq, 4),
            "disc": round(disc, 4),
            "loved": loved_ct, "dis": dis_ct,
        })
    meta.sort(key=lambda m: -m["score"])
    return meta


EASTERN_DROP = DATA / "eastern_drop.json"
WEST_PROJECTED = DATA / "western_projected_tags.json"
WEST_CORPUS = DATA / "western_corpus.json"
# Cross-domain model scores for Western titles (same logistic, applied to the
# LLM-tagged Western vectors). {title: score 0..1}. Absent file -> no scores.
WEST_SCORES = DATA / "western_scores.json"
# Precomputed unified / IDF / Western-only map layouts + IDF tag weights.
# layout*.json keys: "e<id>" (Eastern) and "w<title>" (Western); values [x,y].
# layout_western.json: "w<title>" only. idf_weights.json: {tagName: weight}.
LAYOUT = DATA / "layout.json"
LAYOUT_IDF = DATA / "layout_idf.json"
LAYOUT_WESTERN = DATA / "layout_western.json"
IDF_WEIGHTS = DATA / "idf_weights.json"
# verdict seeds so these few show up pre-rated on first run (task-specified).
WEST_SEED_VERDICTS = {
    "Crossed": "loved", "Sara": "loved",
    "Y: The Last Man": "liked", "Saga": "liked",
    "Crécy": "meh", "Crecy": "meh", "Locke & Key": "liked",
}


# BUGFIX (Western dropped under genre filter): Western titles have no AniList
# genres, so any active genre chip excluded them from rankedList/drawPlot. Fix =
# DERIVE a `.g` for each Western from its tags so they are treated exactly like
# Eastern titles by genreActive(). Each (case-insensitive) tag SUBSTRING maps to
# one or more AniList genres that exist in the corpus genre list (DATA.genres).
# A Western gets a genre if it has any mapped tag with weight >= 0.4. Every
# Western ends with at least ["Action"] so it is never genre-orphaned.
WEST_TAG_GENRE_MAP = [
    ("gore", ["Horror"]),
    ("cosmic horror", ["Horror"]),
    ("body horror", ["Horror"]),
    ("zombie", ["Horror"]),
    ("cannibalism", ["Horror"]),
    ("sadism", ["Horror"]),
    ("pandemic", ["Horror"]),
    ("curses", ["Horror", "Supernatural", "Fantasy"]),
    ("demons", ["Supernatural", "Fantasy"]),
    ("magic", ["Supernatural", "Fantasy"]),
    ("mythology", ["Supernatural", "Fantasy"]),
    ("ghost", ["Supernatural", "Fantasy"]),
    ("gods", ["Supernatural", "Fantasy"]),
    ("afterlife", ["Supernatural", "Fantasy"]),
    ("vampire", ["Supernatural", "Fantasy"]),
    ("werewolf", ["Supernatural", "Fantasy"]),
    ("war", ["Action"]),
    ("military", ["Action"]),
    ("guns", ["Action"]),
    ("swordplay", ["Action"]),
    ("martial arts", ["Action"]),
    ("revenge", ["Action"]),
    ("anti-hero", ["Action"]),
    ("survival", ["Action", "Adventure"]),
    ("gangs", ["Action"]),
    ("crime", ["Mystery", "Thriller"]),
    ("detective", ["Mystery", "Thriller"]),
    ("police", ["Mystery", "Thriller"]),
    ("conspiracy", ["Mystery", "Thriller"]),
    ("espionage", ["Mystery", "Thriller"]),
    ("spy", ["Mystery", "Thriller"]),
    ("mafia", ["Mystery", "Thriller"]),
    ("drugs", ["Mystery", "Thriller"]),
    ("space", ["Sci-Fi"]),
    ("aliens", ["Sci-Fi"]),
    ("post-apocalyptic", ["Sci-Fi"]),
    ("dystopian", ["Sci-Fi"]),
    ("cyborg", ["Sci-Fi"]),
    ("time", ["Sci-Fi"]),
    ("travel", ["Adventure"]),
    ("pirates", ["Adventure"]),
    ("tragedy", ["Drama", "Psychological"]),
    ("family life", ["Drama"]),
    ("parenthood", ["Drama"]),
    ("historical", ["Drama"]),
    ("philosophy", ["Psychological"]),
    ("memory", ["Psychological"]),
]


def _western_genres(tags):
    """Derive the AniList genre list for a Western item from its (name, weight)
    tags. A genre is included when any mapped tag SUBSTRING (case-insensitive)
    appears in a tag with weight >= 0.4. Falls back to ["Action"] so the title
    is never genre-orphaned. Returns a sorted, de-duplicated list."""
    out = set()
    for nm, w in (tags or []):
        try:
            wt = float(w)
        except (TypeError, ValueError):
            continue
        if wt < 0.4:
            continue
        low = str(nm).lower()
        for sub, genres in WEST_TAG_GENRE_MAP:
            if sub in low:
                out.update(genres)
    if not out:
        out.add("Action")
    return sorted(out)


def _clean_nearest(ne):
    """nearest_eastern may carry NaN titles (no resolved match). Keep only real
    string titles with a finite cosine; cap at 3 for the hover/list."""
    out = []
    for pair in ne or []:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        name, cos = pair[0], pair[1]
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            c = float(cos)
        except (TypeError, ValueError):
            continue
        if c != c:  # NaN
            continue
        out.append([name, round(c, 3)])
    return out


def build_western(feat_cols=None):
    """Load the projected Western titles, join publisher/year by title from the
    Comicvine corpus, scrub NaN, and shape a lean dict the HTML embeds as WEST.
    Returns None if the projected file is absent (Western layer just stays off).

    CHANGE 2: when feat_cols (the Eastern tag->column order) is supplied, each
    item also gets a sparse `v` = [[colIndex, weight], ...] in the SAME j-index
    basis as the Eastern titles, plus its L2 `norm`. This lets a Western title
    act as an anchor/centroid (cosine vs the Eastern cloud) exactly like an
    Eastern title does. Tags not present in feat_cols are skipped."""
    if not WEST_PROJECTED.exists():
        return None
    # name -> column index, in the Eastern feature basis (sparse j indices).
    name_to_col = ({name: idx for idx, name in enumerate(feat_cols)}
                   if feat_cols is not None else {})
    proj = json.loads(WEST_PROJECTED.read_text())
    # title -> cross-domain model score (0..1) from the same logistic. Absent
    # file or missing title -> the item's `s` stays None (no score under the
    # model-score ranking mode).
    scores = json.loads(WEST_SCORES.read_text()) if WEST_SCORES.exists() else {}
    # title -> {publisher, year} from the raw Comicvine corpus (keyed by id).
    meta = {}
    if WEST_CORPUS.exists():
        for _id, rec in json.loads(WEST_CORPUS.read_text()).items():
            nm = rec.get("name")
            if nm and nm not in meta:
                meta[nm] = {"publisher": rec.get("publisher") or "",
                            "year": str(rec.get("year") or "")}
    items = []
    for title, v in proj.items():
        coords = v.get("coords")
        if (not coords or len(coords) < 2
                or any((x is None or x != x) for x in coords)):
            continue  # no PCA placement -> can't put it on the map
        m = meta.get(title, {})
        seed_verdict = WEST_SEED_VERDICTS.get(title)
        # CHANGE 2: full tag list -> sparse vector in the Eastern j-index basis.
        sparse_v = []
        seen_cols = set()
        for nm, w in (v.get("tags") or []):
            col = name_to_col.get(str(nm))
            if col is None or col in seen_cols:
                continue
            seen_cols.add(col)
            sparse_v.append([col, round(float(w), 4)])
        wnorm = (sum(val * val for _, val in sparse_v)) ** 0.5
        sc = scores.get(title)
        items.append({
            "t": title,
            "s": (round(float(sc), 6) if sc is not None else None),  # cross-domain model score (est.)
            # BUGFIX: derived AniList genres so genreActive() treats Western like
            # Eastern. Derived from the FULL tag list (not the top-6 hover slice).
            "g": _western_genres(v.get("tags") or []),
            "v": sparse_v,
            "norm": round(wnorm, 6) if wnorm > 0 else 1e-9,
            "x": round(float(coords[0]), 4),
            "y": round(float(coords[1]), 4),
            "ev": v.get("est_verdict") or None,          # loved/liked/meh/null
            "seed": seed_verdict,                          # explicit first-run seed
            "known": bool(v.get("known")),
            "pub": m.get("publisher", ""),
            "yr": m.get("year", ""),
            "desc": v.get("description") or "",
            "note": v.get("note") or "",
            # top-6 tags by weight, [[name, weight], ...]
            "tags": [[str(nm), round(float(w), 3)]
                     for nm, w in (v.get("tags") or [])[:6]],
            "near": _clean_nearest(v.get("nearest_eastern"))[:3],
            "berserk": (None if v.get("berserk_cos") is None
                        else round(float(v["berserk_cos"]), 3)),
        })
    items.sort(key=lambda d: d["t"].lower())
    return {"items": items, "n": len(items)}


def main():
    corpus = pd.read_parquet(DATA / "corpus.parquet").drop_duplicates("id")
    art = np.load(DATA / "model_artifacts.npz", allow_pickle=True)
    res = pd.read_parquet(DATA / "labels_resolved.parquet")

    # staff + synopsis (optional; gracefully degrade if not fetched yet)
    staff_path = DATA / "staff.json"
    staff = json.loads(staff_path.read_text()) if staff_path.exists() else {}
    title_artists = staff.get("title_artists", {})
    title_authors = staff.get("title_authors", {})
    artist_titles = staff.get("artist_titles", {})
    synopsis = staff.get("synopsis", {})

    # official AniList tag descriptions (optional)
    td_path = DATA / "tag_descriptions.json"
    tag_desc_all = json.loads(td_path.read_text()) if td_path.exists() else {}

    # alternate/scanlation titles (synonyms) for search (optional)
    syn_path = DATA / "synonyms.json"
    synonyms_all = json.loads(syn_path.read_text()) if syn_path.exists() else {}

    # Western (LLM-tagged) titles, already PROJECTED into the SAME PCA space as
    # the Eastern corpus (data/western_projected_tags.json). They drop straight
    # onto the existing scatter via their `coords`. This is ADDITIVE: they are
    # NOT in the supervised model. Enrich each with publisher/year (joined by
    # title from western_corpus.json) and strip any NaN so allow_nan=False holds.
    # CHANGE 2: built AFTER feat_cols below so each Western item can carry a
    # sparse `v`/`norm` in the Eastern j-index basis (built just after Xdf).

    # Precomputed map layouts (CHANGE A/B/C). These DRIVE the plotted (x,y) for
    # every point client-side: Eastern id -> layout["e"+id], Western title ->
    # layout["w"+title]. layout.json is the DEFAULT; layout_idf the IDF view;
    # layout_western the Western-only sub-map. Absent file -> {} (feature off).
    layout = json.loads(LAYOUT.read_text()) if LAYOUT.exists() else {}
    layout_idf = json.loads(LAYOUT_IDF.read_text()) if LAYOUT_IDF.exists() else {}
    layout_western = (json.loads(LAYOUT_WESTERN.read_text())
                      if LAYOUT_WESTERN.exists() else {})
    idf_weights = json.loads(IDF_WEIGHTS.read_text()) if IDF_WEIGHTS.exists() else {}

    Xdf = build_feature_matrix(corpus)
    feat_cols = list(Xdf.columns)
    # CHANGE 2: Western items now embed a sparse `v`/`norm` in this same basis.
    western = build_western(feat_cols)
    X = Xdf.values
    ids_feat = Xdf.index.values
    feat_row = {int(i): r for r, i in enumerate(ids_feat)}

    coord_by_id = {int(i): (float(c[0]), float(c[1]))
                   for i, c in zip(art["ids"], art["coords"])}
    score_by_id = {int(i): float(v) for i, v in zip(art["ids"], art["model_score"])}

    # tag category lookup (from corpus tag objects)
    tag_cat = {}
    for tags in corpus["tags"]:
        for t in tags:
            if t.get("category") and t["name"] not in tag_cat:
                tag_cat[t["name"]] = t["category"]

    titles = []
    for _, row in corpus.iterrows():
        tid = int(row["id"])
        if tid not in feat_row or tid not in coord_by_id:
            continue
        vec = X[feat_row[tid]]
        nz = np.nonzero(vec)[0]
        sparse = [[int(j), round(float(vec[j]), 4)] for j in nz]
        x, y = coord_by_id[tid]
        titles.append({
            "id": tid,
            "r": s(row.get("romaji")),
            "e": s(row.get("english")),
            "c": s(row.get("country")),
            "as": None if pd.isna(row.get("averageScore")) else int(row["averageScore"]),
            "p": None if pd.isna(row.get("popularity")) else int(row["popularity"]),
            "x": round(x, 4),
            "y": round(y, 4),
            "s": round(score_by_id[tid], 4),
            "v": sparse,
            "tt": top_tags(list(row["tags"])),
            "g": list(row["genres"]) if row["genres"] is not None else [],
            "syn": synopsis.get(str(tid), ""),
            "alt": synonyms_all.get(str(tid), []),   # alternate/scanlation titles
            "ar": title_artists.get(str(tid), []),
            "au": title_authors.get(str(tid), []),
        })

    tag_meta = build_tag_meta(Xdf, feat_cols, X, feat_row, res)
    genres = sorted({g for t in titles for g in t["g"]})
    seed = build_seed(res)

    # Eastern "drop" list: AniList ids of Romance-without-action titles to hide
    # from the map / recs / counts (CHANGE 1). Plain JSON list of ints. Visible
    # only if the user has a library entry for them (handled JS-side).
    eastern_drop = []
    if EASTERN_DROP.exists():
        eastern_drop = [int(i) for i in json.loads(EASTERN_DROP.read_text())]

    # tag descriptions only for the tags we actually embed (keep blob lean)
    tag_desc = {name: tag_desc_all.get(name, "")
                for name in feat_cols if tag_desc_all.get(name)}

    # IDF weight per feature INDEX, aligned to feat_cols, so the client-side
    # anchor cosine (which works on tag indices, not names) can reweight each
    # tag value by its distinctiveness when the IDF toggle is on. Features with
    # no entry (e.g. demo::/genre:: helpers) default to 1.0 (no reweight).
    idf_by_index = [round(float(idf_weights.get(name, 1.0)), 4) for name in feat_cols]

    # Reading Shelf: link each seed title to a corpus id (cid) and/or Western key
    # (wk) by normalized-title match, so the app can offer a "→ map" button. We
    # index BOTH the english and romaji names (dispName prefers english, falls
    # back to romaji), so a seed like "Berserk"/"Vagabond" matches under either.
    # Index english + romaji + alternate/scanlation titles (synonyms). This lets
    # "Demon Slayer" match "Kimetsu no Yaiba" and "Hell's Paradise" match
    # "Jigokuraku" via the synonym list, not just the exact primary name.
    _corpus_by_norm = {}
    for _t in titles:
        _names = [_t.get("e"), _t.get("r")] + list(_t.get("alt") or [])
        for _name in _names:
            _nm = _norm_title(_name)
            if _nm and _nm not in _corpus_by_norm:
                _corpus_by_norm[_nm] = _t["id"]
    _west_by_norm = {}
    for _w in (western["items"] if western else []):
        _nm = _norm_title(_w.get("t"))
        if _nm and _nm not in _west_by_norm:
            _west_by_norm[_nm] = _w["t"]

    def _match_corpus(_n):
        # exact normalized match (incl. synonyms), then a word-boundary prefix
        # fallback so "demon slayer" -> "demon slayer kimetsu no yaiba". The
        # trailing space stops "eternal" from hitting "eternally regressing
        # knight"; the length floor avoids tiny generic prefixes.
        if _n in _corpus_by_norm:
            return _corpus_by_norm[_n]
        if len(_n) >= 6:
            _pfx = _n + " "
            for _cn, _cid in _corpus_by_norm.items():
                if _cn.startswith(_pfx):
                    return _cid
        return None

    def _match_west(_n):
        # exact, then word-boundary prefix so the shelf "Eternal" links to the
        # cataloged "Eternal (Black Mask)" without a duplicate dot.
        if _n in _west_by_norm:
            return _west_by_norm[_n]
        if len(_n) >= 6:
            _pfx = _n + " "
            for _wn, _wt in _west_by_norm.items():
                if _wn.startswith(_pfx):
                    return _wt
        return None

    shelf_seed = []
    for _e in SHELF_SEED:
        _n = _norm_title(_e["title"])
        # Western reading-list picks (CBR viking) must never link to an Eastern
        # manga that happens to share a name ("Helheim", "Black Road"): they can
        # only resolve to a Western map dot. Eastern/friend picks match the
        # corpus by exact primary, synonym, or word-boundary prefix.
        _is_west_pick = _e["source"] == "cbr-viking"
        shelf_seed.append({
            "sid": _slug(_e["title"]),
            "title": _e["title"], "status": _e["status"], "source": _e["source"],
            "note": _e["note"], "url": _e["url"],
            "cid": None if _is_west_pick else _match_corpus(_n),
            "wk": _match_west(_n),
        })

    payload = {
        "tags": feat_cols,
        "tagCat": tag_cat,
        "tagDesc": tag_desc,
        "tagMeta": tag_meta,
        "titles": titles,
        "artistTitles": artist_titles,
        "genres": genres,
        "seed": seed,
        "easternDrop": eastern_drop,
        "berserkId": BERSERK_ID,
        "nFeat": len(feat_cols),
        "hasStaff": bool(staff),
        "western": western,   # null if not built; LLM-tagged titles in SAME PCA space
        # CHANGE A/B/C: precomputed map layouts + per-index IDF weights.
        "layout": layout,             # DEFAULT unified RAW coords {"e<id>"/"w<title>": [x,y]}
        "layoutIdf": layout_idf,      # unified IDF-weighted coords (same keys)
        "layoutWestern": layout_western,  # Western-ONLY coords {"w<title>": [x,y]}
        "idfByIndex": idf_by_index,   # IDF weight per feature index (len == nFeat)
        "shelfSeed": shelf_seed,      # Reading Shelf seed (arbitrary queue; cid/wk linked)
    }
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)

    plotly_js = fetch_plotly()
    html = HTML_TEMPLATE.replace("/*PLOTLY_JS*/", plotly_js).replace(
        '"__DATA__"', blob)
    out = OUT / "taste_map_app.html"
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / 1024 / 1024
    n_artist = sum(1 for t in titles if t["ar"])
    n_syn = sum(1 for t in titles if t["syn"])
    print(f"Wrote {out}  ({size_mb:.2f} MB)")
    print(f"  titles={len(titles)} features={len(feat_cols)} votableTags={len(tag_meta)}")
    print(f"  pre-seeded={len(seed)} genres={len(genres)} easternDrop={len(eastern_drop)}")
    print(f"  titles with artist data: {n_artist} / {len(titles)} "
          f"(no artist: {len(titles)-n_artist})")
    print(f"  titles with synopsis: {n_syn} / {len(titles)} "
          f"(no description: {len(titles)-n_syn})")
    print(f"  tags with description: {len(tag_desc)} / {len(feat_cols)} "
          f"(used tags only; {len(tag_meta)} are votable)")
    n_alt = sum(1 for t in titles if t["alt"])
    print(f"  titles with >=1 synonym: {n_alt} / {len(titles)}")
    n_e_lay = sum(1 for k in layout if k.startswith("e"))
    n_w_lay = sum(1 for k in layout if k.startswith("w"))
    print(f"  layout entries: {len(layout)} (E={n_e_lay} W={n_w_lay}) · "
          f"idf={len(layout_idf)} · western-only={len(layout_western)} · "
          f"idfWeights={len(idf_weights)}")
    if western:
        n_seed = sum(1 for it in western["items"] if it["seed"])
        print(f"  western titles: {western['n']} (pre-seeded verdicts: {n_seed})")
    else:
        print("  western: none (western_projected_tags.json not found)")


def fetch_plotly():
    old = OUT / "taste_map.html"
    txt = old.read_text(encoding="utf-8")
    import re
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", txt, flags=re.S)
    biggest = max(scripts, key=len)
    if "Plotly" not in biggest:
        raise RuntimeError("could not locate inlined Plotly in taste_map.html")
    return biggest


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Manga / Manhwa Taste Map — Interactive</title>
<script>/*PLOTLY_JS*/</script>
<style>
  :root{
    --bg-deep:#16151a; --bg-panel:#201f25; --bg-elev:#2a2830;
    --line:#3a3640; --gold:#c9a44c; --gold-soft:#d8b86a; --amber:#e0a062;
    --oxblood:#8c3b3b; --text:#e9e3d7; --text-dim:#9c9488;
    --serif:'Cormorant Garamond','Playfair Display',Georgia,serif;
    --sans:system-ui,-apple-system,'Segoe UI',sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg-deep);color:var(--text);
    font:14px/1.45 var(--sans)}
  body{overflow-x:hidden}
  #app{display:flex;height:100vh;width:100vw;overflow:hidden}
  #left{flex:1 1 auto;min-width:0;display:flex;flex-direction:column}
  #plot{flex:1 1 auto;min-height:0;background:var(--bg-deep);touch-action:none}
  /* let Plotly own touch gestures (pinch-zoom / drag-pan) instead of the browser */
  #plot .plot-container,#plot .svg-container,#plot .main-svg,#plot .nsewdrag{touch-action:none}
  #right{width:454px;flex:0 0 454px;background:var(--bg-panel);
    border-left:1px solid var(--line);display:flex;flex-direction:column;
    box-shadow:-6px 0 22px rgba(0,0,0,.5)}
  header{padding:14px 18px;border-bottom:1px solid var(--line);background:var(--bg-panel)}
  header h1{margin:0;font-family:var(--serif);font-size:23px;font-weight:600;
    color:var(--gold);letter-spacing:.02em}
  header .sub{color:var(--text-dim);font-size:11.5px;margin-top:4px}
  .tabs{display:flex;border-bottom:1px solid var(--line)}
  .tab{flex:1;padding:11px 6px;text-align:center;cursor:pointer;color:var(--text-dim);
    font-weight:600;user-select:none;border-bottom:2px solid transparent;
    letter-spacing:.01em;font-size:12.5px}
  .tab.active{color:var(--gold);border-bottom-color:var(--gold)}
  .tabpane{display:none;flex-direction:column;min-height:0;flex:1}
  .tabpane.active{display:flex}
  .pad{padding:13px 18px}
  label.fld{display:block;color:var(--text-dim);font-size:11px;margin:9px 0 4px;
    text-transform:uppercase;letter-spacing:.07em}
  input[type=text],select,textarea{width:100%;padding:8px 10px;background:var(--bg-elev);
    color:var(--text);border:1px solid var(--line);border-radius:6px;font-size:13px;
    font-family:var(--sans)}
  textarea{resize:vertical;min-height:38px}
  input[type=text]:focus,select:focus,textarea:focus{outline:none;border-color:var(--amber);
    box-shadow:0 0 0 2px rgba(224,160,98,.25)}
  .row{display:flex;gap:8px;align-items:center}
  .btn{padding:7px 12px;background:transparent;color:var(--gold);
    border:1px solid var(--gold);border-radius:6px;cursor:pointer;font-size:12.5px;
    font-family:var(--sans);transition:background .12s,color .12s}
  .btn:hover{background:var(--gold);color:var(--bg-deep)}
  .btn.primary{background:var(--gold);color:var(--bg-deep);border-color:var(--gold);font-weight:600}
  .btn.primary:hover{background:var(--gold-soft);border-color:var(--gold-soft)}
  .btn.sm{padding:3px 8px;font-size:11px}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;max-height:90px;overflow:auto}
  .chip{background:var(--bg-elev);border:1px solid var(--line);border-radius:14px;
    padding:3px 10px;font-size:12px;display:flex;align-items:center;gap:7px}
  .chip b{font-weight:600;color:var(--text)}
  .chip .x{cursor:pointer;color:var(--oxblood);font-weight:700}
  .chip .x:hover{color:var(--gold-soft)}
  /* collapsible genre filter (CHANGE 3) */
  .gfhead{display:flex;align-items:baseline;gap:8px;cursor:pointer;user-select:none;
    margin:9px 0 0;color:var(--text-dim);font-size:11px;text-transform:uppercase;
    letter-spacing:.07em}
  .gfhead .arrow{display:inline-block;transition:transform .12s;color:var(--gold)}
  .gfhead.open .arrow{transform:rotate(90deg)}
  .gfhead .gfsum{flex:1;min-width:0;text-transform:none;letter-spacing:0;
    color:var(--text-dim);font-size:11px;white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis}
  .gfilter.collapsed{display:none}
  /* collapsible whole-options block (Anchor tab) — same look as gfhead */
  .opthead{display:flex;align-items:baseline;gap:8px;cursor:pointer;user-select:none;
    margin:0;padding:9px 18px 9px;color:var(--text-dim);font-size:11px;
    text-transform:uppercase;letter-spacing:.07em;border-bottom:1px solid var(--line)}
  .opthead .arrow{display:inline-block;transition:transform .12s;color:var(--gold)}
  .opthead.open .arrow{transform:rotate(90deg)}
  .opthead .optsum{flex:1;min-width:0;text-transform:none;letter-spacing:0;
    color:var(--text-dim);font-size:11px;white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis}
  .optbox.collapsed{display:none}
  /* genre filter chips */
  .gfilter{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
  .gchip{font-size:11px;padding:2px 9px;border-radius:11px;cursor:pointer;
    border:1px solid var(--line);color:var(--text-dim);background:transparent;
    user-select:none;transition:all .1s}
  .gchip.on{border-color:var(--gold);color:var(--gold);background:rgba(201,164,76,.12)}
  /* per-row genre pills (compact, read-only) */
  .gpill{display:inline-block;font-size:9.5px;padding:1px 6px;margin:1px 3px 1px 0;
    border-radius:9px;border:1px solid var(--line);color:var(--text-dim);white-space:nowrap}
  .results{flex:1 1 auto;overflow:auto;border-top:1px solid var(--line)}
  .item{padding:8px 18px;border-bottom:1px solid var(--line)}
  .item .head{display:flex;gap:8px;align-items:flex-start;cursor:pointer}
  .item:hover{background:var(--bg-elev)}
  .item .meta{flex:1;min-width:0}
  .item .tt{color:var(--text-dim);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .matchnote{color:var(--amber);font-size:10.5px;font-style:italic;opacity:.85;margin:1px 0}
  .item .sim{color:var(--gold);font-variant-numeric:tabular-nums;font-size:12px;flex:0 0 auto;text-align:right}
  .item .sim small{display:block;color:var(--text-dim);font-size:9.5px}
  .item .ttl{font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--text)}
  .cc{color:var(--text-dim);font-size:11px;font-weight:400}
  /* Reading Shelf (arbitrary queue) */
  .shelf-badge{display:inline-block;border:1px solid var(--line);border-radius:10px;
    padding:1px 7px;font-size:10px;color:var(--text-dim);vertical-align:middle}
  .shelf-grphead{padding:9px 18px 4px;color:var(--gold);font-size:11px;font-weight:600;
    text-transform:uppercase;letter-spacing:.07em;border-bottom:1px solid var(--line)}
  .shelf-row .head{align-items:center;cursor:default}
  .shelf-row select{width:auto;padding:4px 6px;font-size:11px}
  .shelf-row .snote{margin-top:6px}
  .shelf-x{cursor:pointer;color:var(--oxblood);font-weight:700;flex:0 0 auto;padding:0 4px}
  .shelf-x:hover{color:var(--gold-soft)}
  .shelfbtn{flex:0 0 auto;font-size:11px;padding:3px 7px;white-space:nowrap}
  .shelfbtn.on{color:var(--gold);border-color:var(--gold-soft)}
  .shelf-alt{color:var(--text-dim);font-size:11px;font-weight:400}
  .shelf-nomap{color:var(--text-dim);font-size:11px;opacity:.7;flex:0 0 auto;padding:0 4px;font-style:italic}
  .shelf-row.flash{box-shadow:0 0 0 2px var(--gold) inset;transition:box-shadow .2s}
  /* READ SOURCE: subtle gold per-title link */
  .readlink{color:var(--gold);font-size:10.5px;text-decoration:none;opacity:.78;
    white-space:nowrap;flex:0 0 auto;margin-left:6px;font-weight:500}
  .readlink:hover{opacity:1;text-decoration:underline;color:var(--gold-soft)}
  .expand .readlink{display:inline-block;margin:6px 0 2px 0;font-size:12px;opacity:1}
  .expand{margin-top:8px;padding-top:8px;border-top:1px dashed var(--line);display:none}
  .expand.open{display:block}
  .syn{font-size:12px;line-height:1.5;color:var(--text);background:var(--bg-deep);
    border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin-top:6px}
  .synmeta{font-size:11px;color:var(--text-dim);margin-top:6px}
  .axisgrid{display:grid;grid-template-columns:auto 1fr;gap:5px 10px;align-items:center;margin-top:8px}
  .axisgrid .alab{font-size:11.5px;color:var(--text-dim)}
  .stars{display:flex;gap:2px}
  .stars span{cursor:pointer;font-size:15px;color:var(--line);line-height:1}
  .stars span.on{color:var(--gold)}
  .stars span.clr{font-size:11px;color:var(--text-dim);margin-left:4px;align-self:center}
  .rate{display:flex;gap:3px;flex:0 0 auto}
  .rate button{width:25px;height:25px;border-radius:5px;border:1px solid var(--line);
    background:var(--bg-elev);cursor:pointer;font-size:13px;padding:0;color:var(--text-dim);transition:all .1s}
  .rate button.on.loved{background:var(--gold);color:var(--bg-deep);border-color:var(--gold)}
  .rate button.on.liked{background:var(--amber);color:var(--bg-deep);border-color:var(--amber)}
  .rate button.on.disliked{background:var(--oxblood);color:var(--text);border-color:var(--oxblood)}
  .rate button.on.meh{background:#5a5650;color:var(--text);border-color:#5a5650}
  .rd{cursor:pointer;font-size:16px;width:24px;text-align:center;color:var(--text-dim);flex:0 0 auto}
  .rd.on{color:var(--gold)}
  .hint{color:var(--text-dim);font-size:11px;margin-top:8px;line-height:1.55}
  .hint.westconf{border-left:2px solid #5fa8a0;padding-left:9px;color:var(--text-dim);font-style:italic}
  .stat{color:var(--text-dim);font-size:11.5px;padding:7px 18px;border-top:1px solid var(--line);font-variant-numeric:tabular-nums}
  .toggle{display:flex;align-items:center;gap:7px;margin-top:10px;font-size:12.5px;color:var(--text-dim)}
  .toggle input{width:auto;accent-color:var(--gold)}
  /* tag-vote rows */
  .tvrow{display:flex;align-items:center;gap:8px;padding:6px 18px;border-bottom:1px solid var(--line)}
  .tvrow:hover{background:var(--bg-elev)}
  .tvmeta{flex:1;min-width:0}
  /* tag name carries its description on hover: dotted underline + help cursor */
  .tvname{font-size:12.5px;color:var(--text);font-weight:600;display:inline}
  .tvname.has-desc{border-bottom:1px dotted var(--text-dim);cursor:help}
  .tvinfo{display:inline-block;margin-left:6px;color:var(--text-dim);opacity:.45;
    font-size:11px;cursor:help;font-style:normal}
  .tvinfo:hover{opacity:.9;color:var(--gold)}
  /* themed tooltip popover (replaces the default browser box) */
  #tagtip{position:fixed;z-index:9999;max-width:280px;display:none;pointer-events:none;
    background:var(--bg-panel);border:1px solid var(--gold);color:var(--text);
    border-radius:7px;padding:8px 11px;font-size:12px;line-height:1.45;
    box-shadow:0 6px 22px rgba(0,0,0,.55)}
  #tagtip .tt-cat{display:block;color:var(--gold-soft);font-size:10px;
    text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
  .tvsub{font-size:10.5px;color:var(--text-dim)}
  .votes{display:flex;gap:2px;flex:0 0 auto}
  .votes button{width:24px;height:23px;border-radius:4px;border:1px solid var(--line);
    background:var(--bg-elev);cursor:pointer;font-size:12px;padding:0;color:var(--text-dim)}
  .votes button.on.love{background:var(--gold);color:var(--bg-deep);border-color:var(--gold)}
  .votes button.on.like{background:var(--amber);color:var(--bg-deep);border-color:var(--amber)}
  .votes button.on.dislike{background:#6b4a4a;color:var(--text);border-color:#6b4a4a}
  .votes button.on.hate{background:var(--oxblood);color:var(--text);border-color:var(--oxblood)}
  .artlist{margin-top:6px}
  .artlist .ai{font-size:11.5px;padding:3px 0;color:var(--text);border-bottom:1px solid var(--line)}
  .artlist .ai .cc{margin-left:6px}
  ::-webkit-scrollbar{width:10px;height:10px}
  ::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px}
  ::-webkit-scrollbar-track{background:transparent}

  /* ===== RESPONSIVE: phone (portrait) + iPad portrait (<=860px) ===== */
  /* Restacks the desktop side-by-side layout vertically. Desktop (>860px)
     is untouched. Map full-width on top, panel full-width below, page scrolls. */
  @media (max-width:860px){
    html,body{height:auto}
    #app{display:block;height:auto;width:100%;overflow:visible}
    #left{display:block;width:100%}
    #plot{width:100%;height:48vh;min-height:320px}
    #right{width:100%;max-width:100%;flex:none;border-left:none;
      border-top:1px solid var(--line);box-shadow:none}
    /* let the right panel + its result lists scroll with the page, not internally */
    .tabpane.active{display:block}
    .results{flex:none;max-height:none;overflow:visible}
    /* slightly larger base font for phone readability */
    html,body{font-size:15px}
    header{padding:13px 16px}
    header h1{font-size:21px}
    /* ----- touch-target bumps (mobile only) ----- */
    .tab{padding:14px 6px;font-size:13.5px;min-height:44px}
    .gchip{font-size:13px;padding:8px 13px;min-height:38px;display:inline-flex;
      align-items:center}
    .chip{padding:7px 12px;font-size:13px;min-height:38px}
    .chip .x{padding:0 4px;font-size:15px}
    .btn{padding:11px 15px;font-size:13.5px;min-height:42px}
    .btn.sm{padding:8px 12px;font-size:12.5px;min-height:36px}
    .opthead,.gfhead{padding-top:13px;padding-bottom:13px;min-height:44px;
      align-items:center}
    .item{padding:12px 16px}
    .item .ttl{font-size:14px}
    .tvrow{padding:11px 16px}
    .rate button{width:34px;height:34px;font-size:15px}
    .votes button{width:34px;height:32px;font-size:14px}
    .rd{font-size:20px;width:32px}
    input[type=text],select,textarea{padding:10px 11px;font-size:15px}
    /* keep map full-width: don't let Plotly legend/colorbar force overflow */
    #plot,#plot .plot-container,#plot .svg-container{max-width:100%}
  }
</style>
</head>
<body>
<div id="tagtip"></div>
<div id="app">
  <div id="left">
    <header>
      <h1>Manga / Manhwa Taste Map — Interactive</h1>
      <div class="sub" id="subtitle">anchor cosine · explicit tag-preference vector · multi-axis ratings · same-artist · genre filter — client-side, offline</div>
    </header>
    <div id="plot"></div>
  </div>
  <div id="right">
    <div class="tabs">
      <div class="tab active" data-tab="anchor">Anchor / Explore</div>
      <div class="tab" data-tab="lib">My Library</div>
      <div class="tab" data-tab="tags">Tag Preferences</div>
      <div class="tab tab-west" data-tab="west">Western</div>
      <div class="tab" data-tab="shelf">Reading Shelf</div>
    </div>

    <!-- ANCHOR TAB -->
    <div class="tabpane active" id="pane-anchor">
      <!-- collapsible wrapper for the whole control block (default collapsed) -->
      <div class="opthead" id="optHead"><span class="arrow">▸</span>
        <span>Options</span><span class="optsum" id="optSum"></span></div>
      <div class="optbox collapsed" id="optBox">
      <div class="pad">
        <label class="fld">Color points / rank by</label>
        <select id="colorBy">
          <option value="anchor">Closeness to selected anchor</option>
          <option value="taste">Taste-fit (your tag votes)</option>
          <option value="blend">Blend (anchor × taste-fit)</option>
          <option value="score">Supervised model score</option>
        </select>
        <label class="fld">Anchor titles (centroid = mean of these)</label>
        <input type="text" id="anchorSearch" placeholder="Search all titles to add…" autocomplete="off">
        <div id="anchorSugg" class="results" style="max-height:160px;display:none;border:1px solid var(--line);border-radius:6px;margin-top:4px"></div>
        <div class="chips" id="anchorChips"></div>
        <div class="gfhead" data-gftab="anchor"><span class="arrow">▸</span>
          <span>Genre filter</span><span class="gfsum" id="gfSumAnchor"></span></div>
        <div class="gfilter collapsed" id="genreFilterAnchor"></div>
        <div class="toggle">
          <input type="checkbox" id="highlightTop">
          <label for="highlightTop" style="margin:0;cursor:pointer">Highlight top-N nearest as a cluster</label>
        </div>
        <div class="toggle">
          <input type="checkbox" id="showWestern" checked>
          <label for="showWestern" style="margin:0;cursor:pointer">Show Western titles <span style="color:#5fa8a0">▲</span> (LLM-tagged, not in model)</label>
        </div>
        <div class="toggle">
          <input type="checkbox" id="idfWeight">
          <label for="idfWeight" style="margin:0;cursor:pointer">IDF · distinctive-theme weighting (re-layout + anchor matches on rare tags)</label>
        </div>
        <div class="toggle">
          <input type="checkbox" id="westernOnly">
          <label for="westernOnly" style="margin:0;cursor:pointer">Western-only universe <span style="color:#5fa8a0">▲</span> (306 titles on their own axes)</label>
        </div>
        <div class="row" style="margin-top:6px;flex-wrap:wrap">
          <span class="hint" style="margin:0">N =</span>
          <input type="text" id="topN" value="25" style="width:54px">
          <label class="toggle" style="margin:0"><input type="checkbox" id="genreHide"> hide (vs dim) filtered</label>
          <button class="btn sm" id="resetAnchor">Reset → Berserk</button>
        </div>
        <!-- READ SOURCE: configurable per-title "↗ Read" link templates -->
        <label class="fld" style="margin-top:10px">READ SOURCE</label>
        <label class="fld" style="font-weight:400;opacity:.85">Eastern reader template <span class="cc">— {q} = title, {id} = AniList id</span></label>
        <input type="text" id="readSrcE" placeholder="https://mangadex.org/titles?q={q}" autocomplete="off">
        <div class="row" style="margin-top:4px;flex-wrap:wrap;gap:4px">
          <button class="btn sm" id="readPresetMangadex">MangaDex</button>
          <button class="btn sm" id="readPresetComick">Comick</button>
          <button class="btn sm" id="readPresetAnilist">AniList</button>
        </div>
        <label class="fld" style="font-weight:400;opacity:.85;margin-top:8px">Western reader template <span class="cc">— {q} = title</span></label>
        <input type="text" id="readSrcW" placeholder="https://batcave.biz/search/{q}" autocomplete="off">
        <div class="hint" style="margin-top:4px">Tip: point this at your own Suwayomi server if you run one.</div>
      </div>
      </div><!-- /optBox -->
      <div class="stat" id="anchorStat"></div>
      <div class="results" id="nearestList"></div>
    </div>

    <!-- LIBRARY TAB -->
    <div class="tabpane" id="pane-lib">
      <div class="pad">
        <input type="text" id="libSearch" placeholder="Search all titles…" autocomplete="off">
        <div class="row" style="margin-top:8px;flex-wrap:wrap">
          <button class="btn primary" id="exportJson">⬇ Export JSON</button>
          <button class="btn primary" id="exportCsv">⬇ Export CSV</button>
          <button class="btn" id="importBtn">⬆ Import</button>
          <input type="file" id="importFile" accept=".json" style="display:none">
        </div>
        <div class="gfhead" data-gftab="lib"><span class="arrow">▸</span>
          <span>Genre filter</span><span class="gfsum" id="gfSumLib"></span></div>
        <div class="gfilter collapsed" id="genreFilterLib"></div>
        <div class="toggle">
          <input type="checkbox" id="libOnlyMarked">
          <label for="libOnlyMarked" style="margin:0;cursor:pointer">Show only read/rated</label>
        </div>
        <div class="hint">Click a row to expand: overall verdict, per-axis stars (Art / Story / Characters / Pacing), a note, synopsis, and “more by artist”. A title can be disliked overall but art=5 — that’s the point.</div>
      </div>
      <div class="stat" id="libStat"></div>
      <div class="results" id="libList"></div>
    </div>

    <!-- TAG PREFERENCES TAB -->
    <div class="tabpane" id="pane-tags">
      <div class="pad">
        <div class="hint" style="margin-top:0">Vote tropes you love/hate. A <b>hated</b> tag actively penalises every title that carries it (negative contribution), it isn’t merely ignored. Ranked by how strongly each tag separates your Loved vs Disliked titles. Counts show how many of your Loved / Disliked titles carry it.</div>
        <input type="text" id="tagSearch" placeholder="Search all votable tags…" autocomplete="off" style="margin-top:8px">
        <div class="row" style="margin-top:6px"><button class="btn sm" id="clearVotes">Clear all votes</button>
          <span class="hint" style="margin:0" id="voteCount"></span></div>
      </div>
      <div class="results" id="tagList"></div>
    </div>

    <!-- WESTERN TAB (LLM-tagged titles, projected into the same PCA space) -->
    <div class="tabpane" id="pane-west">
      <div class="pad">
        <div class="hint westconf" id="westHint" style="margin-top:0">Western titles are LLM-tagged and placed by theme (validated by nearest-neighbor); verdict-prediction is limited until you rate more — they are NOT in the supervised model.</div>
        <input type="text" id="westLibSearch" placeholder="Search Western titles…" autocomplete="off" style="margin-top:8px">
        <div class="toggle"><input type="checkbox" id="westOnlyMarked">
          <label for="westOnlyMarked" style="margin:0;cursor:pointer">Show only read/rated</label></div>
        <div class="hint">Click a row to expand: synopsis, theme tags, and the nearest Eastern titles. Read + overall verdict (loved / liked / meh / disliked) behave exactly like My Library, but persist under their own store.</div>
      </div>
      <div class="stat" id="westStat"></div>
      <div class="results" id="westList"></div>
    </div>

    <!-- READING SHELF TAB (arbitrary reading queue; many titles not in corpus) -->
    <div class="tabpane" id="pane-shelf">
      <div class="pad">
        <div class="hint" style="margin-top:0">Your reading queue — arbitrary titles (anime / comics / webtoons), most of them not in the map corpus. Set a status, edit the note, follow the ↗ link; if a title matches a corpus or Western item a “→ map” button focuses it on the map.</div>
        <div class="row" style="gap:6px;margin-top:8px;margin-bottom:0">
          <input id="shelfAdd" type="text" placeholder="+ add a title (type to search our database)" style="flex:1" autocomplete="off">
          <button class="btn primary" id="shelfAddBtn">Add</button>
        </div>
        <div id="shelfSugg" class="results" style="max-height:200px;display:none;border:1px solid var(--line);border-radius:6px;margin-top:4px;margin-bottom:8px"></div>
        <div class="row" id="shelfFilters" style="gap:6px;flex-wrap:wrap;margin-bottom:6px;margin-top:8px"></div>
      </div>
      <div class="stat" id="shelfStat"></div>
      <div class="results" id="shelfList"></div>
    </div>
  </div>
</div>

<script>
"use strict";
const DATA = "__DATA__";
const LS_LIB = "taste_library_v2";
const LS_LIB_V1 = "taste_library_v1";
const LS_VOTES = "taste_tagvotes_v1";
const LS_WEST = "taste_western_v1";
const LS_SHELF = "taste_shelf_v1";   // Reading Shelf (arbitrary reading queue)
const LS_SHELF_REMOVED = "taste_shelf_removed_v1";   // tombstones: seed sids the user deleted (so the merge doesn't resurrect them)
const LS_GENRE_OPEN = "ui_genre_open";   // CHANGE 3: collapsible genre filter state
const LS_CONTROLS_OPEN = "ui_controls_open";   // collapsible whole-options block (Anchor tab)
// READ SOURCE: per-title "↗ Read" link templates. {q}=URL-encoded title, {id}=AniList id.
const LS_READ_SRC_E = "ui_read_src_e";   // Eastern reader template
const LS_READ_SRC_W = "ui_read_src_w";   // Western reader template
const READ_SRC_E_DEFAULT = "https://mangadex.org/titles?q={q}";
const READ_SRC_W_DEFAULT = "https://batcave.biz/search/{q}";
let readSrcE = localStorage.getItem(LS_READ_SRC_E) || READ_SRC_E_DEFAULT;
let readSrcW = localStorage.getItem(LS_READ_SRC_W) || READ_SRC_W_DEFAULT;

const titles = DATA.titles;
const byId = new Map(titles.map(t => [t.id, t]));
const nFeat = DATA.nFeat;
// CHANGE 1: pure-romance Eastern ids to hide from the map / recs / counts,
// UNLESS the user has their own library entry for them (never hide their data).
const DROP = new Set(DATA.easternDrop || []);
// true => this dropped title should be hidden from map/recs/counts right now.
// Keep it visible if it has any library entry (read/overall/axes/note).
function dropHidden(id){
  if(!DROP.has(id)) return false;
  return !(id in lib);   // hidden only when the user has no library entry
}
const GENRES = DATA.genres;
const dispName = t => t.e || t.r || ("#" + t.id);

// READ SOURCE: build the online-reader URL for a title.
//   name   = display title (used for {q})
//   id     = AniList numeric id (Eastern only; for {id})
//   isWest = pick the Western template instead of the Eastern one
// Western templates never get an id — any {id} placeholder is stripped to "".
function readUrl(name, id, isWest){
  const tpl = isWest ? readSrcW : readSrcE;
  return tpl
    .replace(/\{q\}/g, encodeURIComponent(name==null?"":name))
    .replace(/\{id\}/g, (!isWest && id!=null) ? String(id) : "");
}
// READ SOURCE: the small subtle/gold "↗ Read" anchor markup for a row header.
// stopPropagation on click so it opens the source WITHOUT toggling expand/halo.
function readLinkHtml(name, id, isWest){
  const url = readUrl(name, id, isWest);
  return `<a class="readlink" href="${esc(url)}" target="_blank" rel="noopener"
    onclick="event.stopPropagation()" title="Open in your read source">↗ Read</a>`;
}

/* ---------- Western layer (LLM-tagged, projected into the SAME PCA space) ----
   ADDITIVE. These titles are NOT in the supervised model — placed by theme and
   validated by nearest-neighbor only. WEST is null if the data wasn't built. */
const WEST = DATA.western || null;
const westItems = WEST ? WEST.items : [];
// CHANGE 2: Western title -> item, so a Western title string can anchor the map.
// Each Western item now carries `.v` (sparse, Eastern j-index basis) + `.norm`,
// just like an Eastern title — so it can be used as an anchor/centroid.
const westByTitle = new Map(westItems.map(it => [it.t, it]));
// Normalized-title -> key maps so the Reading Shelf can auto-link a title to its
// map dot AT RUNTIME (mirrors the Python build-time matcher: exact, synonym, then
// word-boundary prefix). Lets manually-added shelf titles get a "→ map" too.
const _normShelf = s => (s||"").toLowerCase().replace(/[^a-z0-9]+/g," ").trim();
const corpusByNorm = new Map();
for(const t of titles){ for(const nm of [t.e, t.r, ...(t.alt||[])]){ const k=_normShelf(nm); if(k && !corpusByNorm.has(k)) corpusByNorm.set(k, t.id); } }
const westByNorm = new Map();
for(const it of westItems){ const k=_normShelf(it.t); if(k && !westByNorm.has(k)) westByNorm.set(k, it.t); }
function matchShelfTitle(title){
  const n=_normShelf(title);
  if(corpusByNorm.has(n)) return {cid:corpusByNorm.get(n), wk:null};
  if(westByNorm.has(n))   return {cid:null, wk:westByNorm.get(n)};
  if(n.length>=6){ const pfx=n+" ";
    for(const [cn,cid] of corpusByNorm){ if(cn.startsWith(pfx)) return {cid, wk:null}; }
    for(const [wn,wt] of westByNorm){ if(wn.startsWith(pfx)) return {cid:null, wk:wt}; }
  }
  return {cid:null, wk:null};
}
// Resolve an anchor key (number => Eastern id, string => Western title) to the
// underlying title-like object (both expose .v / .norm). null if unknown.
function anchorVec(key){
  return (typeof key === 'number') ? (byId.get(key) || null)
                                   : (westByTitle.get(key) || null);
}
// Display name for any anchor key (Eastern id or Western title string).
function anchorName(key){
  if(typeof key === 'number'){ const t=byId.get(key); return t?dispName(t):("#"+key); }
  return String(key);
}
// Country/source label for an anchor key (Eastern country, else "Western").
function anchorCountry(key){
  if(typeof key === 'number'){ const t=byId.get(key); return t?t.c:""; }
  return "Western";
}
// Western library keyed by TITLE (own store). Seeded from per-item est_verdict
// seeds so a handful (Crossed/Sara loved, Y/Saga liked, Crécy/Locke&Key meh)
// show up pre-rated on first run. Edits win over the seed thereafter.
let westLib = loadWest();
// CHANGE 2: Western entries now carry the same per-axis ratings as My Library
// (art/story/characters/pacing 1-5 + note), alongside read/overall.
function blankWest(){ return {read:false, overall:null, art:null, story:null,
  characters:null, pacing:null, note:""}; }
function loadWest(){
  const base = {};
  for(const it of westItems){
    if(it.seed){ base[it.t] = Object.assign(blankWest(), {read:true, overall:it.seed}); }
  }
  let stored = null;
  try{ const raw = localStorage.getItem(LS_WEST); if(raw) stored = JSON.parse(raw); }catch(e){}
  if(stored){
    for(const k in stored){
      base[k] = Object.assign(blankWest(), base[k]||{}, stored[k]||{});
    }
  }
  localStorage.setItem(LS_WEST, JSON.stringify(base));
  return base;
}
function saveWest(){ localStorage.setItem(LS_WEST, JSON.stringify(westLib)); }
function westEnt(title){ return westLib[title] || blankWest(); }
function westEmpty(e){ return !e.read && !e.overall && e.art==null && e.story==null &&
  e.characters==null && e.pacing==null && !e.note; }
function setWestEnt(title, patch){
  const cur = Object.assign(blankWest(), westLib[title]||{});
  Object.assign(cur, patch);
  westLib[title] = cur;
  if(westEmpty(cur)) delete westLib[title];
  saveWest();
}
const westVerdictOf = title => (westLib[title] && westLib[title].overall) || null;
const westReadOf = title => !!(westLib[title] && westLib[title].read);
let showWestern = true;   // "Show Western titles" toggle (default ON)

/* ---------- Reading Shelf (arbitrary reading queue; own store) --------------
   Holds ARBITRARY titles (many are anime/comics NOT in the corpus). Each entry:
     {sid, title, status, source, note, url, cid, wk}
   cid = corpus title id (number) or null; wk = Western key (string) or null —
   both baked at build time so a row can offer a "→ map" button. Seeded from
   DATA.shelfSeed on first run; MERGE on later runs (keep user edits, append any
   new seed sid). Persisted under LS_SHELF and included in Export/Import. */
const SHELF_STATUS_ORDER = ["toread","reading","paused","done"];
const SHELF_STATUS_LABEL = {toread:"To-read", reading:"Reading", paused:"Paused", done:"Done"};
const SHELF_SOURCE_LABEL = {mine:"mine", friend:"friend", "cbr-viking":"CBR-viking", curated:"curated"};
const shelfSeed = DATA.shelfSeed || [];
// Tombstone set: seed entries the user removed with ✕, so the merge in loadShelf
// doesn't resurrect them on reload. Must init BEFORE loadShelf() runs.
function loadShelfRemoved(){ try{ return new Set(JSON.parse(localStorage.getItem(LS_SHELF_REMOVED)||"[]")); }catch(e){ return new Set(); } }
let shelfRemoved = loadShelfRemoved();
function saveShelfRemoved(){ localStorage.setItem(LS_SHELF_REMOVED, JSON.stringify([...shelfRemoved])); }
let shelf = loadShelf();
function shelfFromSeed(e){
  return {sid:e.sid, title:e.title, status:e.status, source:e.source,
    note:e.note||"", url:e.url||"", cid:(e.cid==null?null:e.cid), wk:(e.wk==null?null:e.wk)};
}
function loadShelf(){
  let stored = null;
  try{ const raw = localStorage.getItem(LS_SHELF); if(raw) stored = JSON.parse(raw); }catch(e){}
  let out;
  if(Array.isArray(stored)){
    const seedBySid = new Map(shelfSeed.map(e=>[e.sid, e]));
    // Merge: REFRESH seed-owned fields (map links cid/wk, read url, title, source)
    // from the latest build so rebuilds actually reach the user; keep only the
    // user-editable fields (status, note). For user-added rows, re-resolve the
    // map link in case the dataset changed since they typed it.
    out = stored.map(x=>{
      const sd = seedBySid.get(x.sid);
      if(sd){
        return Object.assign(shelfFromSeed(sd), {
          status: x.status || sd.status,
          note:   (x.note!=null && x.note!=="") ? x.note : sd.note
        });
      }
      const m = matchShelfTitle(x.title || "");
      return Object.assign({}, x, {cid: x.cid!=null?x.cid:m.cid, wk: x.wk!=null?x.wk:m.wk});
    });
    const have = new Set(out.map(x=>x.sid));
    // append seed sids that are neither present NOR tombstoned (user-deleted)
    for(const e of shelfSeed){ if(!have.has(e.sid) && !shelfRemoved.has(e.sid)){ out.push(shelfFromSeed(e)); } }
  } else {
    out = shelfSeed.map(shelfFromSeed);
  }
  localStorage.setItem(LS_SHELF, JSON.stringify(out));
  return out;
}
function saveShelf(){ localStorage.setItem(LS_SHELF, JSON.stringify(shelf)); }

/* ---------- precomputed map layouts (CHANGE A/B/C) ---------------------------
   Every plotted (x,y) is resolved from these embedded objects:
     Eastern id    -> LAYOUT["e"+id]
     Western title -> LAYOUT["w"+title]
   A point with NO layout entry is simply not plotted (this is how the ~491
   romance Eastern titles drop out of layout.json naturally).
   - LAYOUT      : default unified RAW coords
   - LAYOUT_IDF  : unified IDF-weighted coords (IDF toggle ON)
   - LAYOUT_WEST : Western-only coords (Western-only toggle ON) */
const LAYOUT = DATA.layout || {};
const LAYOUT_IDF = DATA.layoutIdf || {};
const LAYOUT_WEST = DATA.layoutWestern || {};
const IDF_BY_INDEX = DATA.idfByIndex || null;   // length nFeat, or null
const HAS_LAYOUT = Object.keys(LAYOUT).length > 0;

const LS_IDF = "ui_idf_weight";        // persist IDF toggle
const LS_WONLY = "ui_western_only";     // persist Western-only toggle
const LS_SESSION = "ui_session_v1";     // persist full session: anchors, top-N, hide/dim, genres, lens, selection, tab
let idfOn = (localStorage.getItem(LS_IDF) === "1");
let westernOnly = (localStorage.getItem(LS_WONLY) === "1");

// active layout for the current toggle combination. Western-only stays on its
// OWN layout regardless of IDF (IDF mainly reshapes the unified view).
const HAS_IDF_LAYOUT = Object.keys(LAYOUT_IDF).length > 0;
const HAS_WEST_LAYOUT = Object.keys(LAYOUT_WEST).length > 0;
function activeLayout(){
  if(westernOnly && HAS_WEST_LAYOUT) return LAYOUT_WEST;
  if(idfOn && HAS_IDF_LAYOUT) return LAYOUT_IDF;
  return LAYOUT;
}
// (x,y) for an Eastern title object, from the active layout. null => don't plot.
function coordE(t){
  const c = activeLayout()["e"+t.id];
  return (c && c.length>=2) ? c : null;
}
// (x,y) for a Western item, from the active layout. null => don't plot.
function coordW(it){
  const c = activeLayout()["w"+it.t];
  return (c && c.length>=2) ? c : null;
}

titles.forEach(t => {
  let sm = 0; for (const [,val] of t.v) sm += val*val;
  t.norm = Math.sqrt(sm) || 1e-9;
  t.gset = new Set(t.g);
  // lowercased primary titles + synonyms for case-insensitive search
  t._lc = [(t.e||""), (t.r||"")].filter(Boolean).map(s=>s.toLowerCase());
  t._altlc = (t.alt||[]).map(s=>String(s).toLowerCase());
});

/* ---------- library state (v2 with migration from v1) ---------- */
function blankEntry(){ return {read:false, overall:null, art:null, story:null,
  characters:null, pacing:null, note:""}; }
let lib = loadLib();
function loadLib(){
  // 1. defaults from the reader's seed (read/overall). These are CANONICAL
  //    members of the library so they always export, even if never edited.
  const base = {};
  for(const [id,v] of Object.entries(DATA.seed)){
    const e = blankEntry(); e.read=!!v.read; e.overall=v.rating||null; base[id]=e;
  }
  // 2. prior stored v2 edits (if any) — these WIN over the seed defaults.
  let stored = null;
  try{ const raw = localStorage.getItem(LS_LIB); if(raw) stored = JSON.parse(raw); }catch(e){}
  // 3. one-time migration of a legacy v1 store (its `rating` -> `overall`).
  if(!stored){
    try{
      const v1 = localStorage.getItem(LS_LIB_V1);
      if(v1){ stored = {}; const old = JSON.parse(v1);
        for(const k in old){ const e=blankEntry();
          e.read=!!old[k].read; e.overall=old[k].rating||null; stored[k]=e; } }
    }catch(e){}
  }
  // MERGE: seed provides defaults, stored edits override per-field where present.
  if(stored){
    for(const k in stored){
      const cur = Object.assign(blankEntry(), base[k]||{}, stored[k]||{});
      base[k] = cur;
    }
  }
  // persist the unified store so `lib` == full effective state from here on.
  localStorage.setItem(LS_LIB, JSON.stringify(base));
  return base;
}
function saveLib(){ localStorage.setItem(LS_LIB, JSON.stringify(lib)); }
function ent(id){ return lib[id] || blankEntry(); }
function setEnt(id, patch){
  const cur = Object.assign(blankEntry(), lib[id]||{});
  Object.assign(cur, patch);
  lib[id] = cur;
  if(!cur.read && !cur.overall && cur.art==null && cur.story==null &&
     cur.characters==null && cur.pacing==null && !cur.note) delete lib[id];
  saveLib();
}
const isRead = id => !!(lib[id] && lib[id].read);
const overallOf = id => (lib[id] && lib[id].overall) || null;

/* ---------- tag votes (preference vector) ---------- */
const VOTE_VAL = {love:2, like:1, neutral:0, dislike:-1, hate:-2};
let votes = loadVotes();
function loadVotes(){
  try{ const r = localStorage.getItem(LS_VOTES); if(r) return JSON.parse(r); }catch(e){}
  return {};   // tagIndex -> 'love'|'like'|'neutral'|'dislike'|'hate'
}
function saveVotes(){ localStorage.setItem(LS_VOTES, JSON.stringify(votes)); }
// preference vector p over nFeat tags; titles' taste-fit = normalized dot(p, vec)
let prefVec = new Float64Array(nFeat), prefNorm = 1e-9;
function rebuildPref(){
  prefVec = new Float64Array(nFeat);
  for(const k in votes){ prefVec[+k] = VOTE_VAL[votes[k]] || 0; }
  let nm=0; for(let j=0;j<nFeat;j++) nm += prefVec[j]*prefVec[j];
  prefNorm = Math.sqrt(nm) || 1e-9;
}
// taste-fit in [-1,1] (hated tags subtract). Scaled to [0,1] for coloring.
function tasteFit(t){
  if(prefNorm < 1e-6) return 0;
  let dot=0; for(const [j,val] of t.v) dot += prefVec[j]*val;
  return dot / (prefNorm * t.norm);
}
function tasteFit01(t){ return (tasteFit(t)+1)/2; }

/* ---------- cosine / anchor ----------
   CHANGE B: when the IDF toggle is on, each tag value is scaled by its IDF
   weight (idfW(j)) before the dot product, so anchor matches lean on
   distinctive tags. The centroid AND each title's norm are reweighted with the
   SAME factor so the cosine stays in [-1,1] and is internally consistent.
   When OFF, idfW returns 1 and this reduces to the original raw cosine. */
function idfW(j){
  return (idfOn && IDF_BY_INDEX) ? (IDF_BY_INDEX[j] || 1) : 1;
}
function centroidOf(ids){
  // CHANGE 2: anchor keys may be Eastern ids (number) OR Western titles (string).
  // anchorVec() resolves both to an obj exposing `.v`. idfW(j) still applies.
  const c = new Float64Array(nFeat); let n=0;
  for(const id of ids){ const t=anchorVec(id); if(!t || !t.v) continue;
    for(const [j,val] of t.v) c[j]+=val*idfW(j); n++; }
  if(n>0) for(let j=0;j<nFeat;j++) c[j]/=n;
  let nm=0; for(let j=0;j<nFeat;j++) nm+=c[j]*c[j];
  return {c, norm:Math.sqrt(nm)||1e-9};
}
// title norm under the active (possibly IDF) weighting.
function titleNorm(t){
  if(!(idfOn && IDF_BY_INDEX)) return t.norm;   // raw precomputed
  let nm=0; for(const [j,val] of t.v){ const w=val*idfW(j); nm+=w*w; }
  return Math.sqrt(nm) || 1e-9;
}
function cosineTo(cen, t){ let d=0; for(const [j,val] of t.v) d+=cen.c[j]*val*idfW(j);
  return d/(cen.norm*titleNorm(t)); }

let anchorIds = [DATA.berserkId];
let colorBy = "anchor";          // anchor | taste | blend | score
let simCache = new Map();        // id -> anchor cosine (Eastern)
// WESTERN AS FIRST-CLASS: parallel cache, Western title -> anchor cosine. Western
// items carry the same `.v`/`.norm` basis, so cosineTo(cen, item) is valid.
let westSim = new Map();          // Western title -> anchor cosine
function recomputeSim(){
  const cen = centroidOf(anchorIds);
  simCache.clear();
  for(const t of titles) simCache.set(t.id, cosineTo(cen, t));
  westSim.clear();
  for(const it of westItems) westSim.set(it.t, cosineTo(cen, it));
}

/* ---------- genre filter ---------- */
let activeGenres = new Set(GENRES);   // default all selected = no filter
const genreActive = t => activeGenres.size===GENRES.length ||
  t.g.some(g => activeGenres.has(g));

/* ---------- score selectors ---------- */
function rankVal(t){
  if(colorBy==="score") return t.s;
  if(colorBy==="taste") return tasteFit01(t);
  if(colorBy==="blend"){ const a=simCache.get(t.id); return 0.5*a + 0.5*tasteFit01(t); }
  return simCache.get(t.id);   // anchor
}
// WESTERN AS FIRST-CLASS: rank metric for a Western item. Western have `.v`
// (taste-fit works) but NO `.s` (model score) — score mode excludes them upstream,
// so this is only ever called for anchor/taste/blend.
function westRankVal(it){
  if(colorBy==="score") return it.s;   // cross-domain model score (est.); null when absent
  if(colorBy==="taste") return tasteFit01(it);
  const a = westSim.get(it.t);
  if(colorBy==="blend") return 0.5*a + 0.5*tasteFit01(it);
  return a;   // anchor
}

/* ---------- plot ---------- */
let plotInit = false;
// list<->map link: the row clicked anywhere (My Library / Western / Anchor list).
// Eastern id (number) OR Western title (string). null => nothing haloed.
let selectedKey = null;
let suppressSel = false;   // true while expandRow() synthetically clicks .head, so the programmatic expand doesn't toggle the selection off
const GOLD_SCALE = [[0,'#2c2a32'],[0.4,'#6b5836'],[0.7,'#c9a44c'],[1,'#f0c878']];
const COL = {gold:'#c9a44c', oxblood:'#8c3b3b', readGrey:'#5a5650',
             bg:'#16151a', text:'#e9e3d7', line:'#262430', panel:'#201f25'};
const WEST_COL = '#5fa8a0';   // cool teal — sets Western apart from gold/charcoal

function westHoverFor(it){
  const v = westVerdictOf(it.t) || it.ev;
  const lab = v ? (" · "+String(v).toUpperCase()) : "";
  const src = [it.pub, it.yr].filter(Boolean).join(" · ");
  const tagStr = (it.tags||[]).slice(0,6).map(p=>p[0]).join(", ");
  const nearStr = (it.near||[]).map(p=>`${p[0]} (${p[1].toFixed(2)})`).join(", ");
  return `<b>${esc(it.t)}</b>${lab}<br>`
       + `<span style="color:${WEST_COL}">Western (LLM-tagged)</span>`
       + (src? " · "+esc(src):"")
       + (tagStr? `<br><i>${esc(tagStr)}</i>`:"")
       + (nearStr? `<br>near: ${esc(nearStr)}`:"");
}

function hoverFor(t){
  const sim = simCache.get(t.id);
  const tf = tasteFit(t);
  const lab = overallOf(t.id) ? (" · "+overallOf(t.id).toUpperCase()) : (isRead(t.id)?" · read":"");
  const gen = t.g.length? "<br>"+esc(t.g.join(", ")) : "";
  return `<b>${esc(dispName(t))}</b>${lab}<br>${t.c} · model=${t.s.toFixed(2)}`
       + ` · anchor=${sim!==undefined?sim.toFixed(2):"–"} · taste=${tf.toFixed(2)}`
       + gen + `<br><i>${esc(t.tt)}</i>`;
}
function nearSetForHighlight(){
  if(!document.getElementById("highlightTop").checked) return null;
  const N = Math.max(1, parseInt(document.getElementById("topN").value)||25);
  // WESTERN AS FIRST-CLASS: rankedList now yields mixed keys — numeric Eastern ids
  // AND Western title strings. The returned Set holds both; drawPlot's Eastern loop
  // tests nearSet.has(t.id) and its Western loop tests nearSet.has(it.t).
  return new Set(rankedList(N, true).map(o=>o.id));
}
function drawPlot(){
  // effective Western-only: only when we actually have a Western layout.
  const westOnlyEff = westernOnly && HAS_WEST_LAYOUT;
  const nearSet = nearSetForHighlight();
  const hideFiltered = document.getElementById("genreHide").checked;
  // CHANGE 4: when "Highlight top-N" + "hide (vs dim) filtered" are BOTH on,
  // show ONLY the top-N nearest (the nearSet) plus the anchor titles; hide rest.
  const onlyNearHide = !!nearSet && hideFiltered;
  const anchorSet = new Set(anchorIds);
  const base=[], loved=[], disliked=[], readUnrated=[];
  for(const t of titles){
    // CHANGE C: Western-only universe hides every Eastern point.
    if(westOnlyEff) break;
    // CHANGE A: coords come from the active layout. No entry => not plotted
    // (this naturally drops the ~491 romance titles absent from layout.json).
    const c = coordE(t);
    if(!c) continue;   // absent from layout => not plotted (even if anchored)
    t._px = c[0]; t._py = c[1];
    if(dropHidden(t.id) && !anchorSet.has(t.id)) continue; // CHANGE 1: hide pure-romance drops (keep anchors)
    if(onlyNearHide && !nearSet.has(t.id) && !anchorSet.has(t.id)) continue; // CHANGE 4
    if(hideFiltered && !genreActive(t)) continue;
    const rt = overallOf(t.id);
    if(rt==="loved") loved.push(t);
    else if(rt==="disliked") disliked.push(t);
    else if(isRead(t.id)) readUnrated.push(t);
    else base.push(t);
  }
  // CHANGE A: plotted coords now come from the active layout (t._px/_py),
  // not the embedded PCA t.x/t.y.
  const xs=a=>a.map(t=>t._px), ys=a=>a.map(t=>t._py), hv=a=>a.map(hoverFor);
  // FEATURE B: per-point customdata so a map click can resolve the title.
  // Eastern points => [id, 0]; Western points => [title, 1]. Read in plotly_click.
  const cdE=a=>a.map(t=>[t.id,0]), cdW=a=>a.map(it=>[it.t,1]);
  const opac = t => genreActive(t) ? 0.82 : 0.07;   // dim non-matching

  const cmode = (colorBy==="score"||colorBy==="taste"||colorBy==="blend");
  const baseTrace = {
    type:"scattergl", mode:"markers", name:"corpus",
    x:xs(base), y:ys(base),
    marker:{
      symbol:"circle",
      size: base.map(t=> nearSet && nearSet.has(t.id) ? 11 : 5),
      color: base.map(rankVal),
      colorscale: GOLD_SCALE,
      cmin: cmode?0:undefined, cmax: cmode?1:undefined,
      showscale:true, opacity:0.82,
      // per-point opacity not supported on scattergl marker.opacity array reliably;
      // emulate dim via separate dimmed trace below.
      line:{ width: base.map(t=> nearSet && nearSet.has(t.id) ? 1.6 : 0.2),
             color: base.map(t=> nearSet && nearSet.has(t.id) ? COL.gold : 'rgba(0,0,0,.35)') },
      colorbar:{ title:{text: colorBarLabel(), side:"right"},
        tickfont:{color:COL.text}, titlefont:{color:COL.text},
        outlinecolor:COL.line, bgcolor:'rgba(0,0,0,0)', thickness:12 }
    },
    text:hv(base), hoverinfo:"text", customdata:cdE(base)
  };
  // dim layer: faint grey for filtered-out base points (when not hiding)
  let dimTrace=null;
  if(!hideFiltered){
    const dim = base.filter(t=>!genreActive(t));
    dimTrace = { type:"scattergl", mode:"markers", name:"", showlegend:false,
      hoverinfo:"skip", x:xs(dim), y:ys(dim), customdata:cdE(dim),
      marker:{symbol:"circle", size:4, color:'#2a2830', opacity:0.5} };
    // overlay: re-plot matching base on top so colors stay vivid
    const matchBase = base.filter(genreActive);
    baseTrace.x = xs(matchBase); baseTrace.y = ys(matchBase);
    baseTrace.text = hv(matchBase); baseTrace.customdata = cdE(matchBase);
    baseTrace.marker.size = matchBase.map(t=> nearSet && nearSet.has(t.id) ? 11 : 5);
    baseTrace.marker.color = matchBase.map(rankVal);
    baseTrace.marker.line.width = matchBase.map(t=> nearSet && nearSet.has(t.id) ? 1.6 : 0.2);
    baseTrace.marker.line.color = matchBase.map(t=> nearSet && nearSet.has(t.id) ? COL.gold : 'rgba(0,0,0,.35)');
  }
  const lovedShown = loved.filter(t=>!hideFiltered || genreActive(t));
  const lovedGlow = { type:"scattergl", mode:"markers", name:"", showlegend:false,
    hoverinfo:"skip", x:xs(lovedShown), y:ys(lovedShown),
    marker:{symbol:"star", size:24, color:'rgba(201,164,76,0.28)'} };
  const lovedTrace = { type:"scattergl", mode:"markers", name:"♥ Loved",
    x:xs(lovedShown), y:ys(lovedShown),
    marker:{symbol:"star", size: lovedShown.map(t=>genreActive(t)?15:7),
      color:COL.gold, line:{width:1.2, color:'#16150f'}},
    text:hv(lovedShown), hoverinfo:"text", customdata:cdE(lovedShown) };
  const disShown = disliked.filter(t=>!hideFiltered || genreActive(t));
  const disTrace = { type:"scattergl", mode:"markers", name:"✕ Disliked",
    x:xs(disShown), y:ys(disShown),
    marker:{symbol:"x", size: disShown.map(t=>genreActive(t)?11:5), color:COL.oxblood,
      line:{width:1.4, color:COL.oxblood}}, text:hv(disShown), hoverinfo:"text",
    customdata:cdE(disShown) };
  const readShown = readUnrated.filter(t=>!hideFiltered || genreActive(t));
  const readTrace = { type:"scattergl", mode:"markers", name:"✕ Read",
    x:xs(readShown), y:ys(readShown),
    marker:{symbol:"x", size: readShown.map(t=>genreActive(t)?8:4), color:COL.readGrey,
      line:{width:1, color:COL.readGrey}}, text:hv(readShown), hoverinfo:"text",
    customdata:cdE(readShown) };

  // Western layer: distinct teal triangles, own legend entry. Always shown
  // unless the "Show Western titles" toggle is off (they have no AniList
  // genres, so the Eastern genre filter doesn't gate them).
  // In Western-only mode they are ALWAYS shown (the "Show Western" toggle is
  // moot); otherwise they obey "Show Western titles". Coords come from the
  // active layout (CHANGE A/C); items without a coord are skipped.
  // CHANGE 2: a Western title can now be the anchor (anchorSet holds strings for
  // Western, numbers for Eastern — no collision in one mixed Set). When the
  // Western layer is hidden we still surface an *anchored* Western title here.
  let westTrace = null;
  const westLayerOn = WEST && (westOnlyEff || showWestern) && westItems.length;
  if(WEST && westItems.length && (westLayerOn || westItems.some(it=>anchorSet.has(it.t)))){
    // BUGFIX: Western now carry derived genres (it.g), so they genre-dim/genre-hide
    // EXACTLY like Eastern via genreActive(it) — no more special-casing.
    //  - keepWest: a Western title we never hide (current anchor or the selected halo).
    //  - onlyNearHide (top-N hide): hide Western not in the top-N nearSet.
    //  - genre filter: !genreActive(it) -> HIDE when hideFiltered is on, else DIM.
    const wshow=[], wx=[], wy=[], wop=[];
    for(const it of westItems){
      if(!westLayerOn && !anchorSet.has(it.t)) continue;   // hidden layer: anchors only
      const keepWest = anchorSet.has(it.t) || selectedKey === it.t;
      // top-N hide: Western not in the top-N nearest (and not anchored) drop out.
      if(onlyNearHide && !keepWest && !(nearSet && nearSet.has(it.t))) continue;
      const matches = genreActive(it) || keepWest;
      // genre hide: hide on + non-matching -> drop out (except kept).
      if(hideFiltered && !matches) continue;
      const c=coordW(it); if(!c) continue;
      // genre dim: hide off + non-matching -> faint (mirror Eastern's dim opacity).
      wshow.push(it); wx.push(c[0]); wy.push(c[1]); wop.push(matches?0.9:0.12); }
    if(wshow.length){
      westTrace = { type:"scattergl", mode:"markers", name:"Western (LLM-tagged)",
        x: wx, y: wy,
        marker:{ symbol:"triangle-up",
          size: wshow.map(it=> (nearSet && nearSet.has(it.t)) ? 16 : (westOnlyEff?10:9)),
          color:WEST_COL,
          line:{ width: wshow.map(it=> (nearSet && nearSet.has(it.t)) ? 2.6 : 0.8),
                 color: wshow.map(it=> (nearSet && nearSet.has(it.t)) ? '#bdeee7' : '#16150f') },
          opacity:wop },
        text: wshow.map(westHoverFor), hoverinfo:"text", customdata:cdW(wshow) };
    }
  }

  // CHANGE 1: a dedicated "Anchor" trace, drawn LAST so it sits on top, marking
  // every current anchor in a vivid magenta open-dot. Coords pulled the SAME way
  // the rest of the map resolves them: Eastern anchor -> coordE, Western anchor
  // -> coordW. An anchor with no coord in the active layout is simply skipped.
  let anchorTrace = null;
  {
    const ax=[], ay=[], atxt=[], acd=[];
    for(const key of anchorIds){
      let c=null, nm="";
      if(typeof key === 'number'){ const t=byId.get(key); if(t){ c=coordE(t); nm=dispName(t); } }
      else { const it=westByTitle.get(key); if(it){ c=coordW(it); nm=it.t; } }
      if(!c) continue;   // no coord in the active layout => hidden
      ax.push(c[0]); ay.push(c[1]); atxt.push(`<b>${esc(nm)}</b><br>★ anchor`);
      acd.push([key, (typeof key==='number')?0:1]);   // FEATURE B: route anchor clicks too
    }
    if(ax.length){
      anchorTrace = { type:"scattergl", mode:"markers", name:"Anchor",
        x:ax, y:ay,
        marker:{ symbol:"circle-open-dot", size:18, color:"#ff5ec7",
          line:{width:2, color:"#ffffff"} },
        text:atxt, hoverinfo:"text", customdata:acd };
    }
  }

  // FEATURE A: a dedicated "Selected" trace (the row clicked in any list), drawn
  // LAST so it sits on top of EVERYTHING (incl. the Anchor trace). A bright white
  // open-circle halo around the single selectedKey. Coord resolved on the active
  // layout via coordE (Eastern id) / coordW (Western title); skipped if no coord.
  let selTrace = null;
  if(selectedKey !== null){
    let c=null, nm="";
    if(typeof selectedKey === 'number'){ const t=byId.get(selectedKey); if(t){ c=coordE(t); nm=dispName(t); } }
    else { const it=westByTitle.get(selectedKey); if(it){ c=coordW(it); nm=it.t; } }
    if(c){
      selTrace = { type:"scattergl", mode:"markers", name:"Selected",
        x:[c[0]], y:[c[1]],
        marker:{ symbol:"circle-open", size:26, color:"#ffffff",
          line:{width:3, color:"#ffffff"} },
        text:[`<b>${esc(nm)}</b><br>◎ selected`], hoverinfo:"text" };
    }
  }

  const traces = [];
  if(dimTrace) traces.push(dimTrace);
  traces.push(baseTrace, lovedGlow, lovedTrace, disTrace, readTrace);
  if(westTrace) traces.push(westTrace);
  if(anchorTrace) traces.push(anchorTrace);   // CHANGE 1: on top of everything
  if(selTrace) traces.push(selTrace);          // FEATURE A: on top of the anchor too
  const layout = {
    paper_bgcolor:COL.bg, plot_bgcolor:COL.bg,
    font:{color:COL.text, family:"system-ui,-apple-system,'Segoe UI',sans-serif"},
    dragmode:"pan",   // one-finger drag pans (touch) instead of box-select
    margin:{l:52,r:12,t:14,b:44},
    xaxis:{title: westOnlyEff
        ? "PC1 · Western-only view  (horror · apocalyptic  ⟷  crime · noir)"
        : "PC1  (action · fantasy · adventure  ⟷  romance · slice-of-life)",
      gridcolor:COL.line, zerolinecolor:COL.line, tickfont:{color:COL.text}, linecolor:COL.line},
    yaxis:{title: westOnlyEff
        ? "PC2 · Western-only view  (cerebral horror  ⟷  action · violence)"
        : "PC2  (seinen · drama · tragedy  ⟷  comedy)",
      gridcolor:COL.line, zerolinecolor:COL.line, tickfont:{color:COL.text}, linecolor:COL.line},
    hoverlabel:{bgcolor:COL.panel, bordercolor:COL.gold, align:"left",
      font:{color:COL.text, family:"system-ui,sans-serif"}},
    legend:{orientation:"h", y:1.04, x:0, font:{color:COL.text}, bgcolor:'rgba(0,0,0,0)'},
    showlegend:true
  };
  if(!plotInit){ Plotly.newPlot("plot",traces,layout,{responsive:true,displaylogo:false,scrollZoom:true}); plotInit=true; }
  else { Plotly.react("plot",traces,layout,{responsive:true,displaylogo:false,scrollZoom:true}); }
  saveSession();   // persist anchors + all control state on every redraw
}
function colorBarLabel(){
  return {anchor:"anchor closeness", taste:"taste-fit", blend:"blend", score:"model score"}[colorBy];
}

/* ---------- ranked list (anchor / taste / blend / score) ---------- */
function rankedList(N, includeRead){
  const anchorSet = new Set(anchorIds);   // mixed: numbers (Eastern) + strings (Western)
  const arr=[];
  for(const t of titles){
    if(anchorSet.has(t.id)) continue;
    if(dropHidden(t.id)) continue;                 // CHANGE 1: exclude pure-romance drops
    if(!includeRead && isRead(t.id)) continue;
    if(!genreActive(t)) continue;
    arr.push({id:t.id, west:false, val:rankVal(t), anchor:simCache.get(t.id),
              taste:tasteFit(t), score:t.s});
  }
  // WESTERN AS FIRST-CLASS: Western items join the same pool — ALWAYS, no
  // genreAll gate. Western now carry DERIVED genres (it.g), so genreActive(it)
  // filters them exactly like Eastern: included when the active genre subset
  // intersects their derived genres, excluded otherwise (e.g. only Romance
  // selected -> dark Western drop out — correct).
  //  - colorBy==="score": only those with a cross-domain score (it.s != null).
  //  - anchor exclusion: skip a Western title that is itself an anchor.
  //  - read exclusion: skip read Western via westReadOf (unless includeRead).
  for(const it of westItems){
    if(anchorSet.has(it.t)) continue;
    if(!includeRead && westReadOf(it.t)) continue;
    if(!genreActive(it)) continue;
    if(colorBy === "score" && it.s == null) continue;   // no cross-domain score -> not rankable under model-score mode
    arr.push({id:it.t, west:true, val:westRankVal(it), anchor:westSim.get(it.t),
              taste:tasteFit(it), score:it.s});
  }
  arr.sort((a,b)=>b.val-a.val);
  return arr.slice(0,N);
}
function renderNearest(){
  const N = Math.max(1, parseInt(document.getElementById("topN").value)||25);
  const list = rankedList(N, false);
  const el = document.getElementById("nearestList"); el.innerHTML="";
  // WESTERN AS FIRST-CLASS: mixed pool — Eastern via rowEl, Western via westNearRowEl.
  for(const o of list){
    if(o.west){ const it=westByTitle.get(o.id); if(it) el.appendChild(westNearRowEl(it, o)); }
    else el.appendChild(rowEl(byId.get(o.id), o, false));
  }
  const cb = colorBarLabel();
  document.getElementById("anchorStat").textContent =
    `Anchor = ${anchorIds.map(anchorName).join(", ")} · `
    + `top ${list.length} by ${cb} · read excluded`
    + (activeGenres.size<GENRES.length? ` · genre-filtered`:``);
  applyControlsCollapse();   // keep the collapsed Options summary in sync
}

/* ---------- a title row (shared by nearest + library) ---------- */
function gpills(t){ return t.g.map(g=>`<span class="gpill">${esc(g)}</span>`).join(""); }
function rowEl(t, rankObj, isLib){
  const d = document.createElement("div"); d.className="item"; d.dataset.id=t.id;
  const rt = overallOf(t.id);
  const simStr = rankObj ? `<div class="sim">${(rankObj.val).toFixed(3)}
      <small>a${rankObj.anchor!==undefined?rankObj.anchor.toFixed(2):'–'} ·
      t${rankObj.taste.toFixed(2)}</small></div>` : "";
  const readToggle = `<div class="rd ${isRead(t.id)?'on':''}" data-act="read" title="toggle read">${isRead(t.id)?'✓':'○'}</div>`;
  const matchNote = t._match ? `<div class="matchnote">matched: ${esc(t._match)}</div>` : "";
  const onShelf = inShelf(t.id, null, dispName(t));
  const shelfBtn = `<button class="btn sm shelfbtn${onShelf?' on':''}" data-act="addshelf" title="add to reading shelf">${onShelf?'✓ shelf':'+ shelf'}</button>`;
  d.innerHTML = `<div class="head">
      ${isLib?readToggle:""}
      <div class="meta"><div class="ttl">${esc(dispName(t))} <span class="cc">${t.c}${t.as?(' · '+t.as):''}</span> ${readLinkHtml(t.e||t.r||dispName(t), t.id, false)}</div>
        ${matchNote}
        <div>${gpills(t)}</div>
        <div class="tt">${esc(t.tt)}</div></div>
      ${simStr}
      ${shelfBtn}
    </div>
    <div class="expand" data-exp="${t.id}"></div>`;
  // toggle expand on head click (but not when clicking the read dot)
  d.querySelector(".head").addEventListener("click", e=>{
    if(e.target.dataset.act==="read"){ setEnt(t.id,{read:!isRead(t.id)});
      refreshAll(); return; }
    if(e.target.dataset.act==="addshelf"){ addToShelf(dispName(t), t.id, null);
      e.target.textContent="✓ shelf"; e.target.classList.add("on"); return; }
    const ex = d.querySelector(".expand");
    const open = ex.classList.toggle("open");
    if(open) fillExpand(ex, t, isLib);
    // FEATURE A: clicking a list row (My Library OR Anchor nearest) halos it on
    // the map. ADDITIVE — expand still works exactly as before.
    // FIX 2: toggle — clicking the already-selected row clears the halo.
    if(!suppressSel){ selectedKey = (selectedKey === t.id) ? null : t.id; drawPlot(); }
  });
  return d;
}

// WESTERN AS FIRST-CLASS: a Western row inside the Anchor nearest list. Mirrors
// rowEl's structure but marks the title with a teal ▲ + "Western" badge, shows the
// Western tag string, and on click sets selectedKey to the Western TITLE (halo),
// exactly like the Western library row does. Reuses fillWestExpand for the body.
function westNearRowEl(it, rankObj){
  const d = document.createElement("div"); d.className="item"; d.dataset.west=it.t;
  // Under model-score mode the shown number is it.s (a cross-domain estimate),
  // so flag it "(est.)" to make the Western score's provenance clear.
  const estStr = (colorBy==="score") ? ' <small style="color:'+WEST_COL+'">(est.)</small>' : '';
  const simStr = rankObj ? `<div class="sim">${(rankObj.val).toFixed(3)}${estStr}
      <small>a${rankObj.anchor!==undefined&&rankObj.anchor!==null?rankObj.anchor.toFixed(2):'–'} ·
      t${rankObj.taste.toFixed(2)}</small></div>` : "";
  const tagStr = (it.tags||[]).map(p=>p[0]).join(", ");
  d.innerHTML = `<div class="head">
      <div class="meta"><div class="ttl">${esc(it.t)}
        <span class="cc" style="color:${WEST_COL}">▲ Western</span> ${readLinkHtml(it.t, null, true)}</div>
        <div class="tt">${esc(tagStr)}</div></div>
      ${simStr}
    </div>
    <div class="expand" data-wexp="${esc(it.t)}"></div>`;
  d.querySelector(".head").addEventListener("click", e=>{
    const ex=d.querySelector(".expand");
    if(ex.classList.toggle("open")) fillWestExpand(ex, it);
    // halo this Western title on the map — same selectedKey wiring as westRowEl.
    if(!suppressSel){ selectedKey = (selectedKey === it.t) ? null : it.t; drawPlot(); }
  });
  return d;
}

function starRow(id, axis, val){
  let h = `<div class="alab">${axis[0]}</div><div class="stars" data-axis="${axis[1]}" data-id="${esc(id)}">`;
  for(let i=1;i<=5;i++) h += `<span class="${val&&i<=val?'on':''}" data-v="${i}">★</span>`;
  h += `<span class="clr" data-v="0">clear</span></div>`;
  return h;
}
function fillExpand(ex, t, isLib){
  const e = ent(t.id);
  let h = "";
  if(isLib){
    h += `<div class="row" style="gap:6px;flex-wrap:wrap">
      <div class="rate" data-id="${t.id}">
        <button data-rt="loved" class="${e.overall==='loved'?'on loved':''}" title="Loved">★</button>
        <button data-rt="liked" class="${e.overall==='liked'?'on liked':''}" title="Liked">👍</button>
        <button data-rt="disliked" class="${e.overall==='disliked'?'on disliked':''}" title="Disliked">👎</button>
        <button data-rt="meh" class="${e.overall==='meh'?'on meh':''}" title="Meh">😐</button>
      </div></div>
      <div class="axisgrid">
        ${starRow(t.id,['Art','art'],e.art)}
        ${starRow(t.id,['Story','story'],e.story)}
        ${starRow(t.id,['Characters & MC','characters'],e.characters)}
        ${starRow(t.id,['Pacing','pacing'],e.pacing)}
      </div>
      <textarea data-note="${t.id}" placeholder="note…" style="margin-top:8px">${esc(e.note||"")}</textarea>`;
  }
  h += `<div>${readLinkHtml(t.e||t.r||dispName(t), t.id, false)}</div>`;
  if(t.syn) h += `<div class="syn">${esc(t.syn)}</div>`;
  else h += `<div class="synmeta">No synopsis available.</div>`;
  const artists = t.ar||[];
  if(artists.length){
    h += `<div class="synmeta">Art/Story: ${artists.map(esc).join(", ")}
      <button class="btn sm" data-artist="${t.id}" style="margin-left:6px">🖌 more by artist</button></div>
      <div class="artlist" data-artwrap="${t.id}" style="display:none"></div>`;
  } else {
    h += `<div class="synmeta">No artist data.</div>`;
  }
  ex.innerHTML = h;

  // wire overall buttons
  ex.querySelectorAll(".rate button").forEach(b=>b.onclick=ev=>{
    ev.stopPropagation();
    const rt=b.dataset.rt, cur=e.overall, next=cur===rt?null:rt;
    setEnt(t.id,{overall:next, read: next?true:isRead(t.id)});
    fillExpand(ex,t,isLib); refreshAll(false);
  });
  // wire stars
  ex.querySelectorAll(".stars span").forEach(sp=>sp.onclick=ev=>{
    ev.stopPropagation();
    const wrap=sp.parentElement, axis=wrap.dataset.axis, v=+sp.dataset.v;
    setEnt(t.id,{[axis]: v===0?null:v, read:true});
    fillExpand(ex,t,isLib); libStat();
  });
  // wire note
  const ta = ex.querySelector(`textarea[data-note]`);
  if(ta) ta.addEventListener("input",()=>{ setEnt(t.id,{note:ta.value}); });
  if(ta) ta.addEventListener("click",ev=>ev.stopPropagation());
  // wire artist lookup
  const ab = ex.querySelector(`button[data-artist]`);
  if(ab) ab.onclick=ev=>{
    ev.stopPropagation();
    const wrap = ex.querySelector(`div[data-artwrap]`);
    if(wrap.style.display==="block"){ wrap.style.display="none"; return; }
    wrap.style.display="block"; wrap.innerHTML = artistTitlesHtml(t);
  };
}
function artistTitlesHtml(t){
  const seen=new Set([t.id]); const out=[];
  for(const a of (t.ar||[])){
    const ids = DATA.artistTitles[a] || [];
    for(const oid of ids){ if(seen.has(oid)) continue; seen.add(oid);
      const o=byId.get(oid); if(!o) continue;
      out.push(`<div class="ai">${esc(dispName(o))}<span class="cc">${o.c}${o.as?(' · '+o.as):''}</span>
        <span class="cc"> — ${esc(a)}</span></div>`); }
  }
  return out.length? out.join("") :
    `<div class="ai cc">No other titles by ${esc((t.ar||[]).join(", "))} in this corpus.</div>`;
}

/* ---------- anchor chips & search ---------- */
function renderChips(){
  // CHANGE 2: anchorIds may hold Eastern ids (number) and Western titles (string).
  // Encode the key + its kind in data-attrs so removal can rebuild it precisely.
  const el=document.getElementById("anchorChips"); el.innerHTML="";
  for(const key of anchorIds){
    const west = (typeof key !== 'number');
    const c=document.createElement("div"); c.className="chip";
    const tri = west ? ` <span class="cc" style="color:${WEST_COL}">▲ Western</span>` : "";
    const cc = west ? "" : ` <span class="cc">${esc(anchorCountry(key))}</span>`;
    c.innerHTML=`<b>${esc(anchorName(key))}</b>${cc}${tri} `
      + `<span class="x" data-key="${esc(String(key))}" data-west="${west?1:0}">✕</span>`;
    el.appendChild(c);
  }
  el.querySelectorAll(".x").forEach(x=>x.onclick=()=>{
    const west = x.dataset.west==="1";
    const key = west ? x.dataset.key : +x.dataset.key;
    anchorIds = anchorIds.filter(a=> !(a===key));
    if(anchorIds.length===0) anchorIds=[DATA.berserkId]; refreshAnchor();
  });
}
// Search romaji + english + ALL synonyms (substring, case-insensitive).
// Ranks primary-title matches before synonym-only matches. When a result
// matched via a synonym, sets t._match = the matched alt title (transient,
// for the muted "matched: …" hint). Returns title objects (back-compat).
function searchTitles(q, limit){
  q=q.trim().toLowerCase(); if(!q) return [];
  const primary=[], secondary=[];
  for(const t of titles){
    t._match = null;
    // tier 1: primary romaji/english
    if(t._lc.some(s=>s.includes(q))){ primary.push(t); continue; }
    // tier 2: synonyms
    const hit = (t._altlc||[]).findIndex(s=>s.includes(q));
    if(hit>=0){ t._match = t.alt[hit]; secondary.push(t); }
  }
  const out = primary.concat(secondary);
  return limit? out.slice(0,limit) : out;
}
// CHANGE 2: anchor search also matches Western titles (title substring), so a
// Western title can be picked as the anchor/centroid. Returns lean records
// {kind:'e'|'w', key, name, country, match} ranked Eastern-primary, then
// Eastern-synonym, then Western. (Western items have no synonyms in the data.)
function searchAnchors(q, limit){
  q=(q||"").trim().toLowerCase(); if(!q) return [];
  const east = searchTitles(q, 0).map(t=>(
    {kind:'e', key:t.id, name:dispName(t), country:t.c, match:t._match||null}));
  const west = [];
  for(const it of westItems){
    const nm = it.t;
    if(String(nm).toLowerCase().includes(q)){
      west.push({kind:'w', key:nm, name:nm, country:'Western', match:null});
    }
  }
  const out = east.concat(west);
  return limit? out.slice(0,limit) : out;
}
function wireAnchorSearch(){
  const inp=document.getElementById("anchorSearch"), sg=document.getElementById("anchorSugg");
  inp.addEventListener("input",()=>{
    const res=searchAnchors(inp.value,40);
    if(!res.length){ sg.style.display="none"; return; }
    sg.innerHTML=""; sg.style.display="block";
    for(const r of res){
      const d=document.createElement("div"); d.className="item"; d.style.cursor="pointer";
      const mh = r.match ? `<div class="matchnote">matched: ${esc(r.match)}</div>` : "";
      const cc = r.kind==='w'
        ? `<span class="cc" style="color:${WEST_COL}">▲ Western</span>`
        : `<span class="cc">${esc(r.country)}</span>`;
      d.innerHTML=`<div class="head"><div class="meta"><div class="ttl">${esc(r.name)} ${cc}</div>${mh}</div></div>`;
      d.onclick=()=>{ if(!anchorIds.includes(r.key)) anchorIds.push(r.key);
        inp.value=""; sg.style.display="none"; refreshAnchor(); };
      sg.appendChild(d);
    }
  });
  document.addEventListener("click",e=>{ if(!sg.contains(e.target)&&e.target!==inp) sg.style.display="none"; });
}

/* ---------- genre filter chips (both tabs share activeGenres) ---------- */
// CHANGE 3: collapsible. Default COLLAPSED; open/closed persisted in localStorage.
let genreOpen = (localStorage.getItem(LS_GENRE_OPEN) === "1");
function genreSummary(){
  if(activeGenres.size===GENRES.length) return "All";
  // preserve GENRES order for a stable, readable summary
  return GENRES.filter(g=>activeGenres.has(g)).join(", ") || "All";
}
function applyGenreCollapse(){
  for(const [headSel, listId, sumId] of [
      ['[data-gftab="anchor"]',"genreFilterAnchor","gfSumAnchor"],
      ['[data-gftab="lib"]',"genreFilterLib","gfSumLib"]]){
    const head=document.querySelector(headSel);
    const list=document.getElementById(listId);
    const sum=document.getElementById(sumId);
    if(!head||!list) continue;
    head.classList.toggle("open", genreOpen);
    list.classList.toggle("collapsed", !genreOpen);
    if(sum) sum.textContent = genreOpen ? "" : ("Genre filter: "+genreSummary());
  }
}

/* ---------- collapsible whole-options block (Anchor tab) ---------- */
// Default COLLAPSED (user wants the list up). Open/closed persisted in localStorage.
let controlsOpen = (localStorage.getItem(LS_CONTROLS_OPEN) === "1");
// Compact summary shown when collapsed: anchor set + top-N + active toggles.
function optSummary(){
  const N = Math.max(1, parseInt(document.getElementById("topN").value)||25);
  const anchors = anchorIds.map(anchorName).join(", ");
  let s = `${anchors} · top ${N}`;
  if(idfOn) s += " · IDF";
  if(westernOnly) s += " · Western-only";
  return s;
}
function applyControlsCollapse(){
  const head=document.getElementById("optHead");
  const box=document.getElementById("optBox");
  const sum=document.getElementById("optSum");
  if(!head||!box) return;
  head.classList.toggle("open", controlsOpen);
  box.classList.toggle("collapsed", !controlsOpen);
  if(sum) sum.textContent = controlsOpen ? "" : optSummary();
}
function renderGenreFilters(){
  for(const elId of ["genreFilterAnchor","genreFilterLib"]){
    const el=document.getElementById(elId); el.innerHTML="";
    const all=document.createElement("span");
    all.className="gchip"+(activeGenres.size===GENRES.length?" on":"");
    all.textContent="All"; all.onclick=()=>{ activeGenres=new Set(GENRES); afterGenreChange(); };
    el.appendChild(all);
    for(const g of GENRES){
      const c=document.createElement("span");
      c.className="gchip"+(activeGenres.has(g)?" on":"");
      c.textContent=g;
      c.onclick=()=>{
        if(activeGenres.size===GENRES.length){ activeGenres=new Set([g]); }
        else if(activeGenres.has(g)){ activeGenres.delete(g); if(activeGenres.size===0) activeGenres=new Set(GENRES); }
        else activeGenres.add(g);
        afterGenreChange();
      };
      el.appendChild(c);
    }
  }
  applyGenreCollapse();   // refresh collapse state + summary text
}
function afterGenreChange(){ renderGenreFilters(); drawPlot(); renderNearest(); renderLib(); }

function refreshAnchor(){ recomputeSim(); renderChips(); drawPlot(); renderNearest(); }
// refreshAll: after rating/vote changes. recolor optional (votes need recompute)
function refreshAll(recolorOnly){ drawPlot(); renderNearest(); renderLib(); libStat(); }

/* ---------- library tab ---------- */
function renderLib(){
  const q=document.getElementById("libSearch").value.trim().toLowerCase();
  const onlyMarked=document.getElementById("libOnlyMarked").checked;
  const el=document.getElementById("libList"); el.innerHTML="";
  let shown=0;
  const pool = q? searchTitles(q,800) : titles;
  if(!q){ for(const t of pool) t._match=null; }  // no stale "matched:" while browsing
  for(const t of pool){
    if(!genreActive(t)) continue;
    const marked = isRead(t.id)||overallOf(t.id)||ent(t.id).art!=null||ent(t.id).note;
    if(onlyMarked && !marked) continue;
    if(!q && !onlyMarked && shown>=300) break;
    el.appendChild(rowEl(t, null, true));
    shown++;
  }
  libStat();
}
function libStat(){
  let read=0,loved=0,liked=0,dis=0,meh=0,axes=0;
  for(const k in lib){ const v=lib[k];
    if(v.read) read++;
    if(v.overall==="loved")loved++; else if(v.overall==="liked")liked++;
    else if(v.overall==="disliked")dis++; else if(v.overall==="meh")meh++;
    if(v.art!=null||v.story!=null||v.characters!=null||v.pacing!=null) axes++; }
  document.getElementById("libStat").textContent =
    `${read} read · ★${loved} · 👍${liked} · 👎${dis} · 😐${meh} · ${axes} with axis ratings`;
}

/* ---------- Western tab (LLM-tagged titles, own library store) ---------- */
function westRefresh(){ renderWest(); westStat(); drawPlot(); }
function renderWest(){
  const el=document.getElementById("westList"); if(!el) return;
  el.innerHTML="";
  if(!WEST || !westItems.length){
    el.innerHTML=`<div class="item cc" style="padding:14px 18px">No Western data embedded.</div>`;
    westStat(); return;
  }
  const q=document.getElementById("westLibSearch").value.trim().toLowerCase();
  const onlyMarked=document.getElementById("westOnlyMarked").checked;
  let shown=0;
  for(const it of westItems){
    if(q && !it.t.toLowerCase().includes(q)) continue;
    const marked = westReadOf(it.t)||westVerdictOf(it.t);
    if(onlyMarked && !marked) continue;
    el.appendChild(westRowEl(it));
    shown++;
  }
  if(!shown) el.innerHTML=`<div class="item cc" style="padding:14px 18px">No matches.</div>`;
  westStat();
}
function westStat(){
  const elS=document.getElementById("westStat"); if(!elS) return;
  let read=0,loved=0,liked=0,dis=0,meh=0;
  for(const k in westLib){ const v=westLib[k];
    if(v.read) read++;
    if(v.overall==="loved")loved++; else if(v.overall==="liked")liked++;
    else if(v.overall==="disliked")dis++; else if(v.overall==="meh")meh++; }
  elS.textContent = `${westItems.length} Western titles · ${read} read · `
    + `★${loved} · 👍${liked} · 👎${dis} · 😐${meh}`;
}
function westRowEl(it){
  const d=document.createElement("div"); d.className="item"; d.dataset.west=it.t;
  const v = westVerdictOf(it.t) || it.ev;
  const verdictTag = v
    ? `<span class="sim" style="color:${WEST_COL}">${esc(String(v))}${westVerdictOf(it.t)?"":" <small>est.</small>"}</span>`
    : "";
  const src = [it.pub, it.yr].filter(Boolean).join(" · ");
  const readToggle = `<div class="rd ${westReadOf(it.t)?'on':''}" data-act="wread" title="toggle read">${westReadOf(it.t)?'✓':'○'}</div>`;
  const onShelfW = inShelf(null, it.t, it.t);
  const shelfBtnW = `<button class="btn sm shelfbtn${onShelfW?' on':''}" data-act="addshelf" title="add to reading shelf">${onShelfW?'✓ shelf':'+ shelf'}</button>`;
  d.innerHTML = `<div class="head">
      ${readToggle}
      <div class="meta"><div class="ttl">${esc(it.t)} <span class="cc">${esc(src)}</span> ${readLinkHtml(it.t, null, true)}</div>
        <div class="tt">${esc((it.tags||[]).map(p=>p[0]).join(", "))}</div></div>
      ${verdictTag}
      ${shelfBtnW}
    </div>
    <div class="expand" data-wexp="${esc(it.t)}"></div>`;
  d.querySelector(".head").addEventListener("click", e=>{
    if(e.target.dataset.act==="wread"){ setWestEnt(it.t,{read:!westReadOf(it.t)}); westRefresh(); return; }
    if(e.target.dataset.act==="addshelf"){ addToShelf(it.t, null, it.t);
      e.target.textContent="✓ shelf"; e.target.classList.add("on"); return; }
    const ex=d.querySelector(".expand");
    if(ex.classList.toggle("open")) fillWestExpand(ex, it);
    // FEATURE A: halo this Western title on the map. ADDITIVE.
    // FIX 2: toggle — clicking the already-selected row clears the halo.
    if(!suppressSel){ selectedKey = (selectedKey === it.t) ? null : it.t; drawPlot(); }
  });
  return d;
}
function fillWestExpand(ex, it){
  const e = westEnt(it.t);
  let h = `<div class="row" style="gap:6px;flex-wrap:wrap">
      <div class="rate" data-west="${esc(it.t)}">
        <button data-rt="loved" class="${e.overall==='loved'?'on loved':''}" title="Loved">★</button>
        <button data-rt="liked" class="${e.overall==='liked'?'on liked':''}" title="Liked">👍</button>
        <button data-rt="disliked" class="${e.overall==='disliked'?'on disliked':''}" title="Disliked">👎</button>
        <button data-rt="meh" class="${e.overall==='meh'?'on meh':''}" title="Meh">😐</button>
      </div></div>
      <div class="axisgrid">
        ${starRow(it.t,['Art','art'],e.art)}
        ${starRow(it.t,['Story','story'],e.story)}
        ${starRow(it.t,['Characters & MC','characters'],e.characters)}
        ${starRow(it.t,['Pacing','pacing'],e.pacing)}
      </div>
      <textarea data-wnote="${esc(it.t)}" placeholder="note…" style="margin-top:8px">${esc(e.note||"")}</textarea>`;
  h += `<div>${readLinkHtml(it.t, null, true)}</div>`;
  if(it.ev && !e.overall) h += `<div class="synmeta">est. verdict (theme-projected): <b>${esc(it.ev)}</b> — LLM guess, not a model prediction.</div>`;
  if(it.desc) h += `<div class="syn">${esc(it.desc)}</div>`;
  else h += `<div class="synmeta">No synopsis available.</div>`;
  if(it.note) h += `<div class="synmeta">${esc(it.note)}</div>`;
  if((it.tags||[]).length){
    h += `<div class="synmeta">Theme tags</div><div class="chips" style="max-height:none">`
       + it.tags.map(p=>`<span class="chip" style="border-color:var(--gold)"><b style="color:var(--gold-soft)">${esc(p[0])}</b> <span class="cc">${p[1].toFixed(2)}</span></span>`).join("")
       + `</div>`;
  }
  if((it.near||[]).length){
    h += `<div class="synmeta" style="margin-top:8px">Nearest Eastern titles</div><div class="artlist">`
       + it.near.map(p=>`<div class="ai">${esc(p[0])}<span class="cc"> — cos ${p[1].toFixed(2)}</span></div>`).join("")
       + `</div>`;
  }
  if(it.berserk!=null) h += `<div class="synmeta">cos to Berserk: ${it.berserk.toFixed(2)}</div>`;
  ex.innerHTML = h;
  ex.querySelectorAll(".rate button").forEach(b=>b.onclick=ev=>{
    ev.stopPropagation();
    const rt=b.dataset.rt, cur=westEnt(it.t).overall, next=cur===rt?null:rt;
    setWestEnt(it.t,{overall:next, read: next?true:westReadOf(it.t)});
    fillWestExpand(ex,it); westRefresh();
  });
  // CHANGE 2: per-axis stars (same behavior as My Library, own store)
  ex.querySelectorAll(".stars span").forEach(sp=>sp.onclick=ev=>{
    ev.stopPropagation();
    const axis=sp.parentElement.dataset.axis, v=+sp.dataset.v;
    setWestEnt(it.t,{[axis]: v===0?null:v, read:true});
    fillWestExpand(ex,it); westStat();
  });
  const wta = ex.querySelector(`textarea[data-wnote]`);
  if(wta){ wta.addEventListener("input",()=>{ setWestEnt(it.t,{note:wta.value}); });
    wta.addEventListener("click",ev=>ev.stopPropagation()); }
}

/* ---------- Reading Shelf tab (arbitrary reading queue; own store) ---------- */
let shelfFilter = "all";   // "all" or one of SHELF_STATUS_ORDER
function shelfRefresh(){ renderShelf(); }
function shelfStat(){
  const elS=document.getElementById("shelfStat"); if(!elS) return;
  const c={toread:0,reading:0,paused:0,done:0};
  for(const e of shelf){ if(e.status in c) c[e.status]++; }
  elS.textContent = `${shelf.length} titles · ${c.toread} to-read · ${c.reading} reading · `
    + `${c.paused} paused · ${c.done} done`;
}
function renderShelfFilters(){
  const el=document.getElementById("shelfFilters"); if(!el) return;
  const c={toread:0,reading:0,paused:0,done:0};
  for(const e of shelf){ if(e.status in c) c[e.status]++; }
  const mk=(key,label,n)=>`<button class="btn sm${shelfFilter===key?' primary':''}" `
    + `data-shfilter="${key}">${esc(label)}${n!=null?` (${n})`:""}</button>`;
  let h = mk("all","All",shelf.length);
  for(const st of SHELF_STATUS_ORDER) h += mk(st, SHELF_STATUS_LABEL[st], c[st]);
  el.innerHTML = h;
  el.querySelectorAll("button[data-shfilter]").forEach(b=>b.onclick=()=>{
    shelfFilter = b.dataset.shfilter; renderShelf();
  });
}
function shelfRowEl(e){
  const d=document.createElement("div"); d.className="item shelf-row"; d.dataset.sid=e.sid;
  // status <select>
  let opts="";
  for(const st of SHELF_STATUS_ORDER)
    opts += `<option value="${st}"${e.status===st?" selected":""}>${esc(SHELF_STATUS_LABEL[st])}</option>`;
  const srcLabel = SHELF_SOURCE_LABEL[e.source] || e.source || "";
  const badge = srcLabel ? `<span class="shelf-badge">${esc(srcLabel)}</span>` : "";
  const readlink = e.url
    ? `<a class="readlink" href="${esc(e.url)}" target="_blank" rel="noopener" title="Open">↗</a>`
    : "";
  const mapBtn = (e.cid!=null)
    ? `<button class="btn sm" data-shmap="e">→ map</button>`
    : (e.wk!=null ? `<button class="btn sm" data-shmap="w">→ map</button>`
    : `<span class="shelf-nomap" title="not in the dataset yet">· not on map</span>`);
  // Show the database's title when it differs from the shelf title (alternate
  // official translation), e.g. "Doom Breaker" for "Reincarnation of the Suicidal…".
  let altname="";
  if(e.cid!=null){ const t=byId.get(e.cid); const nm=t?dispName(t):""; if(nm && _normShelf(nm)!==_normShelf(e.title)) altname=nm; }
  else if(e.wk!=null && _normShelf(e.wk)!==_normShelf(e.title)) altname=e.wk;
  const altspan = altname ? ` <span class="shelf-alt">(maps to: ${esc(altname)})</span>` : "";
  d.innerHTML = `<div class="head">
      <div class="meta">
        <div class="ttl">${esc(e.title)}${altspan} ${badge} ${readlink}</div>
      </div>
      <select data-shstatus title="status">${opts}</select>
      ${mapBtn}
      <span class="shelf-x" data-shdel title="remove">✕</span>
    </div>
    <input class="snote" type="text" data-shnote placeholder="note…" value="${esc(e.note||"")}">`;
  // status change
  d.querySelector("select[data-shstatus]").onchange=ev=>{
    ev.stopPropagation();
    e.status = ev.target.value; saveShelf(); renderShelf();
  };
  // note edit (persist on change/blur)
  const ni=d.querySelector("input[data-shnote]");
  ni.addEventListener("change",()=>{ e.note=ni.value; saveShelf(); });
  ni.addEventListener("click",ev=>ev.stopPropagation());
  // → map
  const mb=d.querySelector("button[data-shmap]");
  if(mb) mb.onclick=ev=>{
    ev.stopPropagation();
    if(mb.dataset.shmap==="e" && e.cid!=null) focusFromMap(e.cid,false);
    else if(mb.dataset.shmap==="w" && e.wk!=null) focusFromMap(e.wk,true);
  };
  // ✕ remove
  d.querySelector(".shelf-x").onclick=ev=>{
    ev.stopPropagation();
    shelfRemoved.add(e.sid); saveShelfRemoved();   // tombstone so it stays gone after reload
    shelf = shelf.filter(x=>x!==e); saveShelf(); renderShelf();
  };
  return d;
}
function renderShelf(){
  const el=document.getElementById("shelfList"); if(!el) return;
  el.innerHTML="";
  renderShelfFilters();
  const srcRank = s => { const i=["mine","friend","cbr-viking"].indexOf(s);
    return i<0 ? 99 : i; };
  let shown=0;
  for(const st of SHELF_STATUS_ORDER){
    if(shelfFilter!=="all" && shelfFilter!==st) continue;
    const group = shelf.filter(e=>e.status===st);
    if(!group.length) continue;
    group.sort((a,b)=>{ const r=srcRank(a.source)-srcRank(b.source);
      return r!==0 ? r : a.title.toLowerCase().localeCompare(b.title.toLowerCase()); });
    const hd=document.createElement("div"); hd.className="shelf-grphead";
    hd.textContent = `${SHELF_STATUS_LABEL[st]} (${group.length})`;
    el.appendChild(hd);
    for(const e of group){ el.appendChild(shelfRowEl(e)); shown++; }
  }
  if(!shown) el.innerHTML=`<div class="item cc" style="padding:14px 18px">Nothing here yet.</div>`;
  shelfStat();
}
function shelfAdd(){
  const inp=document.getElementById("shelfAdd"); if(!inp) return;
  const val=inp.value.trim(); if(!val) return;
  const m=matchShelfTitle(val);              // auto-link to a map dot if we have it
  const sid=slug(val)+"-"+shelf.length;
  shelf.push({sid, title:val, status:"toread",
    source:"mine", note:"", url:"", cid:m.cid, wk:m.wk});
  saveShelf(); inp.value="";
  shelfFilter="all";                         // ensure the new (To-read) row is visible
  renderShelf();
  const row=document.querySelector('#shelfList .shelf-row[data-sid="'+cssEsc(sid)+'"]');
  if(row){ row.scrollIntoView({block:"center"}); row.classList.add("flash");
    setTimeout(()=>row.classList.remove("flash"),1200); }
}
// Is a title already on the shelf? (match by corpus id, Western key, or name)
function shelfHas(cid, wk, title){
  const tn=_normShelf(title||"");
  return shelf.find(e =>
    (cid!=null && e.cid===cid) ||
    (wk!=null && e.wk===wk) ||
    (tn && _normShelf(e.title)===tn)) || null;
}
function inShelf(cid, wk, title){ return !!shelfHas(cid, wk, title); }
// Add a title (from a row button or a search pick) to the shelf, linked to its
// map dot. Dedups, un-tombstones, persists. Returns true if newly added.
function addToShelf(title, cid, wk){
  if(shelfHas(cid, wk, title)) return false;
  let sid = slug(title) || ("t-"+shelf.length);
  if(shelf.some(e=>e.sid===sid)) sid = sid+"-"+shelf.length;
  if(shelfRemoved.has(sid)){ shelfRemoved.delete(sid); saveShelfRemoved(); }
  shelf.push({sid, title, status:"toread", source:"mine", note:"", url:"", cid:cid==null?null:cid, wk:wk==null?null:wk});
  saveShelf();
  const at=document.querySelector(".tab.active");
  if(at && at.dataset.tab==="shelf") renderShelf();
  return true;
}
// Autocomplete on the shelf "add" box: same DB search as Library/anchor. Picking
// a result links it to its dot; typing a non-match still adds via shelfAdd().
function wireShelfSearch(){
  const inp=document.getElementById("shelfAdd"), sg=document.getElementById("shelfSugg");
  if(!inp||!sg) return;
  inp.addEventListener("input",()=>{
    const res=searchAnchors(inp.value,40);
    if(!res.length){ sg.style.display="none"; return; }
    sg.innerHTML=""; sg.style.display="block";
    for(const r of res){
      const d=document.createElement("div"); d.className="item"; d.style.cursor="pointer";
      const already=inShelf(r.kind==='e'?r.key:null, r.kind==='w'?r.key:null, r.name);
      const mh=r.match?`<div class="matchnote">matched: ${esc(r.match)}</div>`:"";
      const cc=r.kind==='w'?`<span class="cc" style="color:${WEST_COL}">▲ Western</span>`:`<span class="cc">${esc(r.country)}</span>`;
      const chk=already?' <small style="color:var(--gold)">✓ in shelf</small>':'';
      d.innerHTML=`<div class="head"><div class="meta"><div class="ttl">${esc(r.name)} ${cc}${chk}</div>${mh}</div></div>`;
      d.onclick=()=>{
        addToShelf(r.name, r.kind==='e'?r.key:null, r.kind==='w'?r.key:null);
        inp.value=""; sg.style.display="none"; shelfFilter="all"; renderShelf();
        const sid=slug(r.name);
        const row=document.querySelector('#shelfList .shelf-row[data-sid="'+cssEsc(sid)+'"]');
        if(row){ row.scrollIntoView({block:"center"}); row.classList.add("flash"); setTimeout(()=>row.classList.remove("flash"),1200); }
      };
      sg.appendChild(d);
    }
  });
  document.addEventListener("click",e=>{ if(!sg.contains(e.target)&&e.target!==inp) sg.style.display="none"; });
}

/* ---------- tag preferences tab ---------- */
function renderTags(){
  const q=document.getElementById("tagSearch").value.trim().toLowerCase();
  const el=document.getElementById("tagList"); el.innerHTML="";
  let pool = DATA.tagMeta;
  if(q) pool = pool.filter(m=>m.name.toLowerCase().includes(q));
  else pool = pool.slice(0,120);   // top-ranked first; search reveals the rest
  for(const m of pool) el.appendChild(tagRow(m));
  const nv = Object.values(votes).filter(v=>v&&v!=="neutral").length;
  document.getElementById("voteCount").textContent = `${nv} tag(s) voted`;
}
/* themed tooltip for tag descriptions (replaces the browser's yellow box) */
const tagDesc = DATA.tagDesc || {};
const _tip = () => document.getElementById("tagtip");
function showTip(ev, name){
  const desc = tagDesc[name];
  const cat = DATA.tagCat[name] || "";
  const tip = _tip();
  tip.innerHTML = (cat? `<span class="tt-cat">${esc(cat)}</span>`:"")
    + (desc? esc(desc) : "No description.");
  tip.style.display = "block";
  moveTip(ev);
}
function moveTip(ev){
  const tip = _tip(); if(tip.style.display!=="block") return;
  const pad=14, w=tip.offsetWidth, h=tip.offsetHeight;
  let x=ev.clientX+pad, y=ev.clientY+pad;
  if(x+w>window.innerWidth-8) x=ev.clientX-w-pad;
  if(y+h>window.innerHeight-8) y=ev.clientY-h-pad;
  tip.style.left=Math.max(8,x)+"px"; tip.style.top=Math.max(8,y)+"px";
}
function hideTip(){ _tip().style.display="none"; }

function tagRow(m){
  const cat = DATA.tagCat[m.name] || "";
  const cur = votes[m.i] || "neutral";
  const hasDesc = !!tagDesc[m.name];
  const d=document.createElement("div"); d.className="tvrow";
  d.innerHTML=`<div class="tvmeta">
      <span class="tvname${hasDesc?' has-desc':''}" data-tag="${esc(m.name)}">${esc(m.name)}</span>${
        hasDesc?`<span class="tvinfo" data-tag="${esc(m.name)}">ⓘ</span>`:""}
      <div class="tvsub">${esc(cat)} · loved ${m.loved} / disliked ${m.dis} · freq ${(m.freq*100).toFixed(0)}%</div>
    </div>
    <div class="votes" data-i="${m.i}">
      <button data-v="love" class="${cur==='love'?'on love':''}" title="Love +2">♥</button>
      <button data-v="like" class="${cur==='like'?'on like':''}" title="Like +1">+</button>
      <button data-v="dislike" class="${cur==='dislike'?'on dislike':''}" title="Dislike −1">−</button>
      <button data-v="hate" class="${cur==='hate'?'on hate':''}" title="Hate −2">✕</button>
    </div>`;
  // hover affordances -> themed tooltip
  d.querySelectorAll("[data-tag]").forEach(elm=>{
    elm.addEventListener("mouseenter", ev=>showTip(ev, elm.dataset.tag));
    elm.addEventListener("mousemove", moveTip);
    elm.addEventListener("mouseleave", hideTip);
  });
  d.querySelectorAll(".votes button").forEach(b=>b.onclick=()=>{
    const i=b.parentElement.dataset.i, v=b.dataset.v;
    votes[i] = (votes[i]===v)? "neutral" : v;
    if(votes[i]==="neutral") delete votes[i];
    saveVotes(); rebuildPref();
    renderTags();
    // recolor live if a taste-based view is active
    drawPlot(); renderNearest();
  });
  return d;
}

/* ---------- export / import (round-trips axes + votes) ---------- */
function libRows(){
  const rows=[];
  for(const k in lib){ const id=+k, t=byId.get(id); if(!t) continue; const e=lib[k];
    rows.push({id, title:dispName(t), country:t.c, read:!!e.read,
      overall:e.overall||"", art:e.art??"", story:e.story??"",
      characters:e.characters??"", pacing:e.pacing??"", note:e.note||""}); }
  return rows.sort((a,b)=>a.title.localeCompare(b.title));
}
function download(name, text, type){
  const blob=new Blob([text],{type}); const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download=name; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),2000);
}
function votesExport(){
  // export readable: tagName -> vote
  const out={};
  for(const k in votes){ const name=DATA.tags[+k]; if(name) out[name]=votes[k]; }
  return out;
}
function exportJson(){
  download("taste_library_v2.json", JSON.stringify({
    version:"taste_library_v2", exported:new Date().toISOString(),
    library:lib, tagVotes:votes, tagVotesByName:votesExport(),
    western: westLib,   // Western library (own corpus); empty {} if untouched
    shelf: shelf,       // Reading Shelf (arbitrary reading queue)
    shelfRemoved: [...shelfRemoved]   // seed titles the user deleted (tombstones)
  }, null, 2), "application/json");
}
function exportCsv(){
  const rows=libRows();
  const head="id,title,country,read,overall,art,story,characters,pacing,note\n";
  const body=rows.map(r=>[r.id,csv(r.title),r.country,r.read,r.overall,
    r.art,r.story,r.characters,r.pacing,csv(r.note)].join(",")).join("\n");
  download("taste_library_v2.csv", head+body, "text/csv");
}
function importJson(file){
  const fr=new FileReader();
  fr.onload=()=>{
    try{
      const obj=JSON.parse(fr.result);
      // library: accept v2 {library}, or v1 {entries}/raw map
      let L = obj.library || obj.entries || obj;
      lib={};
      for(const k in L){ const v=L[k]; const e=blankEntry();
        e.read=!!v.read; e.overall=v.overall||v.rating||null;
        e.art=v.art??null; e.story=v.story??null;
        e.characters=v.characters??null; e.pacing=v.pacing??null; e.note=v.note||"";
        lib[k]=e; }
      // votes: prefer index map, else map names back to indices
      if(obj.tagVotes){ votes=Object.assign({},obj.tagVotes); }
      else if(obj.tagVotesByName){ votes={};
        const idx={}; DATA.tags.forEach((n,i)=>idx[n]=i);
        for(const name in obj.tagVotesByName){ if(name in idx) votes[idx[name]]=obj.tagVotesByName[name]; } }
      saveLib(); saveVotes(); rebuildPref();
      // Western section (additive; older exports won't have it)
      let nWest = 0;
      if(obj.western && typeof obj.western==="object"){
        westLib = {};
        for(const k in obj.western){ const v=obj.western[k];
          westLib[k] = Object.assign(blankWest(), {read:!!v.read, overall:v.overall||null,
            art:v.art??null, story:v.story??null, characters:v.characters??null,
            pacing:v.pacing??null, note:v.note||""});
          if(westEmpty(westLib[k])) delete westLib[k]; }
        saveWest(); nWest = Object.keys(westLib).length;
      }
      // Reading Shelf (additive; older exports won't have it). Restore tombstones
      // first, then route the imported shelf through loadShelf so it picks up
      // fresh map links and respects the deletions.
      if(Array.isArray(obj.shelfRemoved)){ shelfRemoved = new Set(obj.shelfRemoved); saveShelfRemoved(); }
      if(Array.isArray(obj.shelf)){ localStorage.setItem(LS_SHELF, JSON.stringify(obj.shelf)); shelf = loadShelf(); }
      refreshAnchor(); renderLib(); renderTags(); libStat();
      if(WEST) westRefresh();
      renderShelf();
      alert("Imported "+Object.keys(lib).length+" Eastern entries, "
        +Object.keys(votes).length+" tag votes, "+nWest+" Western entries, "
        +shelf.length+" shelf titles.");
    }catch(e){ alert("Import failed: "+e.message); }
  };
  fr.readAsText(file);
}

/* ---------- helpers ---------- */
function esc(s){ return (""+(s==null?"":s)).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function csv(s){ s=(""+s); return /[",\n]/.test(s)? '"'+s.replace(/"/g,'""')+'"' : s; }
// Reading Shelf: title -> slug (mirrors the Python _slug; for new-entry sids).
function slug(s){ return (""+(s==null?"":s)).toLowerCase().replace(/[^a-z0-9]+/g,"-").replace(/^-+|-+$/g,""); }

/* ---------- wire UI ---------- */
// FEATURE B: activate a tab by its data-tab name (reused by the map-click router).
function activateTab(name){
  const tab = document.querySelector('.tab[data-tab="'+name+'"]');
  if(!tab) return;
  document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
  document.querySelectorAll(".tabpane").forEach(p=>p.classList.remove("active"));
  tab.classList.add("active");
  const pane=document.getElementById("pane-"+name); if(pane) pane.classList.add("active");
  if(name==="lib") renderLib();
  if(name==="tags") renderTags();
  if(name==="west") renderWest();
  if(name==="shelf") renderShelf();
  saveSession();   // remember the active tab across refreshes
}
document.querySelectorAll(".tab").forEach(tab=>tab.onclick=()=>activateTab(tab.dataset.tab));

// FEATURE B: open (expand) a list row's synopsis/tags without toggling it shut if
// already open. Triggers the same fill logic the user's click would.
function expandRow(d){
  if(!d) return;
  const ex = d.querySelector(".expand");
  if(ex && !ex.classList.contains("open")){ suppressSel=true; d.querySelector(".head").click(); suppressSel=false; }
}
// FEATURE B: route a map click to the right tab + row. dir 'w' (Western) or 'e'.
function focusFromMap(key, isWest){
  // FIX 2: clicking the already-selected point just clears the halo (no re-navigate).
  if(key === selectedKey){ selectedKey = null; drawPlot(); return; }
  if(isWest){
    selectedKey = key;
    activateTab("west");
    // FIX 1: write the Western title into the search bar (mirrors the Eastern branch)
    // so the list filters to it. Assign .value directly — do NOT dispatch 'input'
    // (that would trip the clear-on-edit check in the search handler).
    const ws=document.getElementById("westLibSearch");
    if(ws){ ws.value = String(key); }
    const om=document.getElementById("westOnlyMarked"); if(om && om.checked) om.checked=false;
    renderWest();
    const d=document.querySelector('#westList .item[data-west="'+cssEsc(String(key))+'"]');
    if(d){ d.scrollIntoView({block:"center"}); expandRow(d); }
    drawPlot();
  } else {
    selectedKey = key;
    activateTab("lib");
    // filter My Library to this exact title so it surfaces even if it was past the
    // 300-row browse cap or filtered out, then re-render and focus it.
    const t=byId.get(key);
    const sb=document.getElementById("libSearch");
    if(sb && t){ sb.value = dispName(t); }
    const om=document.getElementById("libOnlyMarked"); if(om && om.checked) om.checked=false;
    if(activeGenres.size<GENRES.length){ activeGenres=new Set(GENRES); renderGenreFilters(); }
    renderLib();
    const d=document.querySelector('#libList .item[data-id="'+key+'"]');
    if(d){ d.scrollIntoView({block:"center"}); expandRow(d); }
    drawPlot();
  }
}
// minimal CSS.escape fallback (Western titles can contain quotes/brackets).
function cssEsc(s){ return (window.CSS && CSS.escape) ? CSS.escape(s)
  : String(s).replace(/["\\\]\[]/g, "\\$&"); }
document.getElementById("colorBy").onchange=e=>{ colorBy=e.target.value; drawPlot(); renderNearest(); };
document.getElementById("highlightTop").onchange=()=>drawPlot();
document.getElementById("topN").addEventListener("input",()=>{ drawPlot(); renderNearest(); });
document.getElementById("genreHide").onchange=()=>drawPlot();
document.getElementById("resetAnchor").onclick=()=>{ anchorIds=[DATA.berserkId]; refreshAnchor(); };
document.getElementById("exportJson").onclick=exportJson;
document.getElementById("exportCsv").onclick=exportCsv;
document.getElementById("importBtn").onclick=()=>document.getElementById("importFile").click();
document.getElementById("importFile").onchange=e=>{ if(e.target.files[0]) importJson(e.target.files[0]); };
document.getElementById("libSearch").addEventListener("input",e=>{
  // FIX 3: if the user edits/deletes the exact selected (Eastern) title, clear the halo.
  // Fires only on real 'input' (user typing); focusFromMap assigns .value, which
  // doesn't dispatch 'input', so it won't trip this.
  if(typeof selectedKey === 'number'){
    const t=byId.get(selectedKey);
    if(!t || e.target.value !== dispName(t)){ selectedKey = null; drawPlot(); }
  }
  renderLib();
});
document.getElementById("libOnlyMarked").onchange=renderLib;
document.getElementById("tagSearch").addEventListener("input",renderTags);
document.getElementById("clearVotes").onclick=()=>{ votes={}; saveVotes(); rebuildPref();
  renderTags(); drawPlot(); renderNearest(); };
// Western tab + map toggle
{
  const sw=document.getElementById("showWestern");
  if(sw) sw.onchange=()=>{ showWestern=sw.checked; drawPlot(); };
  const wls=document.getElementById("westLibSearch");
  if(wls) wls.addEventListener("input",e=>{
    // FIX 3: if the user edits/deletes the exact selected (Western) title, clear the halo.
    // Fires only on real 'input'; focusFromMap assigns .value (no 'input' dispatch).
    if(typeof selectedKey === 'string' && e.target.value !== selectedKey){
      selectedKey = null; drawPlot();
    }
    renderWest();
  });
  const wom=document.getElementById("westOnlyMarked");
  if(wom) wom.onchange=renderWest;
}
// Reading Shelf: add-title button + Enter key
{
  const ab=document.getElementById("shelfAddBtn");
  if(ab) ab.onclick=shelfAdd;
  const ai=document.getElementById("shelfAdd");
  if(ai) ai.addEventListener("keydown",e=>{ if(e.key==="Enter"){ e.preventDefault(); shelfAdd(); } });
}
// CHANGE B: IDF distinctive-theme weighting. Re-layout (active layout switches
// to layoutIdf) AND reweight the anchor cosine -> recompute sim + redraw + rerank.
{
  const idf=document.getElementById("idfWeight");
  if(idf){
    idf.checked = idfOn;
    idf.disabled = !(IDF_BY_INDEX && Object.keys(LAYOUT_IDF).length);
    idf.onchange=()=>{ idfOn=idf.checked;
      localStorage.setItem(LS_IDF, idfOn?"1":"0");
      refreshAnchor(); };   // recomputeSim() + redraw + nearest
  }
}
// CHANGE C: Western-only universe. Switches active layout to layoutWestern and
// hides Eastern points. Cosine is layout-independent, so anchoring still ranks.
{
  const wo=document.getElementById("westernOnly");
  if(wo){
    wo.checked = westernOnly;
    wo.disabled = !(WEST && Object.keys(LAYOUT_WEST).length);
    wo.onchange=()=>{ westernOnly=wo.checked;
      localStorage.setItem(LS_WONLY, westernOnly?"1":"0");
      drawPlot(); };   // layout-only switch; sim unchanged
  }
}
// READ SOURCE: wire template inputs + presets. Persist + re-render visible rows
// so existing "↗ Read" links pick up the new template immediately.
{
  const eIn=document.getElementById("readSrcE");
  const wIn=document.getElementById("readSrcW");
  // re-render whichever lists are currently built so links rebuild with new URLs
  const refreshLinks=()=>{
    try{ renderNearest(); }catch(_){}
    try{ renderLib(); }catch(_){}
    try{ renderWest(); }catch(_){}
  };
  if(eIn){
    eIn.value = readSrcE;
    eIn.addEventListener("input",()=>{ readSrcE = eIn.value || READ_SRC_E_DEFAULT;
      localStorage.setItem(LS_READ_SRC_E, eIn.value); refreshLinks(); });
  }
  if(wIn){
    wIn.value = readSrcW;
    wIn.addEventListener("input",()=>{ readSrcW = wIn.value || READ_SRC_W_DEFAULT;
      localStorage.setItem(LS_READ_SRC_W, wIn.value); refreshLinks(); });
  }
  const setPresetE=(tpl)=>{ readSrcE = tpl;
    if(eIn) eIn.value = tpl;
    localStorage.setItem(LS_READ_SRC_E, tpl); refreshLinks(); };
  const pm=document.getElementById("readPresetMangadex");
  if(pm) pm.onclick=()=>setPresetE("https://mangadex.org/titles?q={q}");
  const pc=document.getElementById("readPresetComick");
  if(pc) pc.onclick=()=>setPresetE("https://comick.io/search?q={q}");
  const pa=document.getElementById("readPresetAnilist");
  if(pa) pa.onclick=()=>setPresetE("https://anilist.co/manga/{id}");
}
wireAnchorSearch();
wireShelfSearch();
// CHANGE 3: genre-filter collapse headers (both tabs share one open/closed state)
document.querySelectorAll('.gfhead').forEach(h=>h.onclick=()=>{
  genreOpen = !genreOpen;
  localStorage.setItem(LS_GENRE_OPEN, genreOpen?"1":"0");
  applyGenreCollapse();
});
// whole-options block collapse (Anchor tab) — default collapsed, persisted
{
  const oh=document.getElementById("optHead");
  if(oh) oh.onclick=()=>{
    controlsOpen = !controlsOpen;
    localStorage.setItem(LS_CONTROLS_OPEN, controlsOpen?"1":"0");
    applyControlsCollapse();
  };
}

/* ---------- session persistence: anchors + every control survive a refresh.
   Stored under LS_SESSION as one blob and restored before the first render, so
   the map boots in exactly the state the user left it. IDF + Western-only keep
   their own keys; this covers anchors, selection, top-N highlight + N, hide/dim,
   the genre subset, the score lens and the active tab. ---------- */
function saveSession(){
  try{
    const g=id=>document.getElementById(id);
    const at=document.querySelector(".tab.active");
    localStorage.setItem(LS_SESSION, JSON.stringify({
      anchorIds, colorBy, selectedKey, showWestern,
      activeGenres:[...activeGenres],
      highlightTop: !!(g("highlightTop") && g("highlightTop").checked),
      topN: g("topN") ? g("topN").value : "25",
      genreHide: !!(g("genreHide") && g("genreHide").checked),
      westOnlyMarked: !!(g("westOnlyMarked") && g("westOnlyMarked").checked),
      shelfFilter,
      tab: at ? at.dataset.tab : null
    }));
  }catch(_){}
}
function restoreSession(){
  let o=null;
  try{ const raw=localStorage.getItem(LS_SESSION); if(raw) o=JSON.parse(raw); }catch(_){ o=null; }
  if(!o || typeof o!=="object") return null;
  if(Array.isArray(o.anchorIds)){
    const v=o.anchorIds.filter(k=>(typeof k==="number"&&byId.has(k))||(typeof k==="string"&&westByTitle.has(k)));
    anchorIds = v.length ? v : [DATA.berserkId];   // drop stale ids; never leave it empty
  }
  if(typeof o.colorBy==="string") colorBy=o.colorBy;
  if(o.selectedKey===null||(typeof o.selectedKey==="number"&&byId.has(o.selectedKey))||(typeof o.selectedKey==="string"&&westByTitle.has(o.selectedKey)))
    selectedKey=o.selectedKey;
  if(typeof o.showWestern==="boolean"){ showWestern=o.showWestern; const sw=document.getElementById("showWestern"); if(sw) sw.checked=showWestern; }
  if(Array.isArray(o.activeGenres)){
    const gv=o.activeGenres.filter(x=>GENRES.includes(x));
    activeGenres = gv.length ? new Set(gv) : new Set(GENRES);
  }
  const chk=(id,val)=>{ const e=document.getElementById(id); if(e && typeof val==="boolean") e.checked=val; };
  chk("highlightTop", o.highlightTop);
  chk("genreHide", o.genreHide);
  chk("westOnlyMarked", o.westOnlyMarked);
  if(o.topN!=null){ const tn=document.getElementById("topN"); if(tn) tn.value=o.topN; }
  if(typeof o.shelfFilter==="string") shelfFilter=o.shelfFilter;
  return (typeof o.tab==="string") ? o.tab : null;
}

/* ---------- boot ---------- */
const _savedTab = restoreSession();   // apply saved state BEFORE first render
rebuildPref();
renderGenreFilters();
refreshAnchor();   // draws the plot once (incl. the Western trace) + nearest
// FEATURE B: map point -> list. Resolve the clicked title via customdata
// ([key, isWestern]) and route to the right tab + row. plotInit is true now
// because refreshAnchor() already ran drawPlot() once.
{
  const plotEl=document.getElementById("plot");
  if(plotEl && plotEl.on){
    plotEl.on("plotly_click", ev=>{
      const p = ev && ev.points && ev.points[0];
      if(!p || !p.customdata) return;
      const cd = p.customdata;               // [key, isWestern]
      const key = cd[0], isWest = cd[1]===1;
      focusFromMap(key, isWest);
    });
  }
}
libStat();
shelfStat();
if(WEST){ westStat(); }
else {
  // no Western data: hide the tab + the map toggle so the UI stays clean
  const wt=document.querySelector('.tab-west'); if(wt) wt.style.display="none";
  const swrow=document.getElementById("showWestern");
  if(swrow && swrow.closest(".toggle")) swrow.closest(".toggle").style.display="none";
}
// hide a toggle row entirely when its backing data is missing (keeps UI clean)
function hideToggleIf(id, cond){
  if(!cond) return; const el=document.getElementById(id);
  if(el && el.closest(".toggle")) el.closest(".toggle").style.display="none";
}
hideToggleIf("idfWeight", !(IDF_BY_INDEX && HAS_IDF_LAYOUT));
hideToggleIf("westernOnly", !(WEST && HAS_WEST_LAYOUT));
// restore the last-active tab (default "anchor" already active, needs no action)
if(_savedTab && _savedTab!=="anchor") activateTab(_savedTab);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
