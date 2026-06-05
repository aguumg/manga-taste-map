# manga-taste-map

**An interactive map of ~2,300 comics, positioned by how close each one is to your taste.**

Open `outputs/taste_map_app.html` in any browser — it's a single offline file. Every dot is a comic. Where it sits, what color it is, and what it's near are all computed from a model of what *you* like. Anchor it on *Berserk* and the whole map recolors by "how Berserk-like is this." Rate things and the model learns. Flip to a "Western-only universe" and watch your dark-comics collection unfold into sub-genres.

It started as a one-line request — *"recommend me manga like Berserk"* — and turned into a small, honest data-science project. This README is the whole story, including the math and the parts that failed.

---

## 1. Turning a comic into a vector

You can't do math on "Berserk." You can do math on a **vector**.

[AniList](https://anilist.co) lets thousands of readers vote on *tags* for each title, and — crucially — each tag carries an **intensity rank from 0–100**. Berserk isn't just "tagged horror." It's:

```
Tragedy 97 · Seinen 95 · Revenge 94 · Demons 92 · Cosmic Horror 90
Gore 88 · Philosophy 86 · Found Family 86 · Anti-Hero 83 · Swordplay 82 ...
```

Take the union of all tags across the corpus (≈230 of them), add demographics and genres, and every title becomes a point in **254-dimensional space**, where coordinate *j* is the crowd-voted intensity of tag *j* (0 if absent). That vector *is* the comic, as far as the model is concerned.

> **The unlock:** we never needed to scrape blurbs and run sentiment analysis. The intensity-ranked tag vote *is* a ready-made, human-labeled feature matrix. Step one was a query, not an NLP project.

The corpus: top ~2,000 manga/manhwa/manhua by popularity, pulled from the AniList GraphQL API (`scripts/fetch_anilist.py`).

---

## 2. Learning your taste (the supervised model)

Tags describe a comic. To describe a *person*, we need labels: titles you **loved** and titles you **disliked**.

With those, "taste" becomes a classifier. Two models, deliberately simple:

- **L2-regularized logistic regression** — learns a weight per tag; positive weight = "this tag predicts you'll like it."
- **Nearest-Shrunken-Centroid (diagonal LDA)** — per-tag score ∝ (mean₍liked₎ − mean₍disliked₎) / pooled-sd. A robust, shrinkage-friendly alternative.

### The honest problem: *p ≫ n*

There are **254 features** but only **~30 labeled titles**. With far more knobs than examples, a flexible model will happily fit noise. So we never report training accuracy — only **leave-one-out cross-validation**: hide one labeled title, train on the rest, predict the hidden one, repeat.

The honest numbers, as the label set grew:

| Labels | Logistic LOO-AUC | NSC LOO-AUC |
|-------:|-----------------:|------------:|
| 23     | 0.629            | 0.636       |
| 43     | 0.630            | **0.725**   |
| 59     | **0.670**        | 0.679       |

AUC of 0.5 is a coin flip. ~0.67 is *modestly* predictive — better than chance, not magic. The lesson is written into the project: **more labels move it, re-confirming titles the model already knows does not.** The bottleneck was never the method; it was rows.

---

## 3. You are not one person (bimodality)

Run **k-means (k=2)** on just the *liked* vectors and they split, cleanly, with zero errors:

- **Mode 0 — seinen-tragedy:** Berserk, Vinland Saga, Claymore, Attack on Titan, Blade of the Immortal, Hell's Paradise *(all Japanese)*.
- **Mode 1 — Korean RPG-progression:** Solo Leveling, Overgeared, Pick Me Up, Murim RPG Simulation, Surviving as a Barbarian *(all Korean)*.

This has a sharp consequence. The naive recommender — "find titles near the *average* of everything you like" — fails, because **the centroid of a bimodal cloud lands in the empty valley between the two peaks**, and the densest mass near that valley is generic isekai. Your average is not a thing you like. The fix is to recommend *per mode*, never blended.

---

## 4. Drawing the map (PCA)

The 254-D vectors get projected to 2D with **PCA** for display. The axes are interpretable:

- **PC1** ≈ action/fantasy ⟷ romance/slice-of-life
- **PC2** ≈ seinen/drama/tragedy ⟷ comedy

Your favorites cluster in one corner; the rest of the map is everything you *haven't* anchored on yet. The 2D position is a **lossy summary** — for "is this actually similar," the model uses full-254-D cosine, not the dot's location.

---

## 5. The Western problem (the arc where clever broke and simple won)

Western comics — *Crossed*, *Y: The Last Man*, *Saga*, *Scalped* — **are not on AniList**. No crowd tags, no intensity vectors. So how do you place them in the same space as the manga?

### Attempt 1: description embeddings — *failed*

Plan: embed each Western comic's plot description with a sentence-transformer, embed every AniList tag's definition, and project the comic onto the tags by cosine similarity. Elegant. It produced garbage:

- *Crossed* (extreme gore-horror) got tagged **Villainess / 4-koma / Satire** and placed next to villainess-romance manhwa.

Two reasons, both real: (a) the comics database served **German editions** — *Crossed*'s description was literally in German, which the English embedder choked on; (b) the descriptions were **publication metadata**, not plot ("collected into ten trade paperbacks, Unmanned #1–5, Cycles #6–10…"). You can't embed *theme* from a volume index.

### Attempt 2: LLM tag-assignment — *won*

Drop the embeddings. Have a language model read each title and **assign the real AniList tags from knowledge**:

> *Crossed* → Gore 1.0, Survival 0.9, Post-Apocalyptic 0.9, Pandemic 0.8, Tragedy 0.8, Sadism 0.8, Cannibalism 0.6…

Validation flipped completely. *Crossed*'s nearest Eastern neighbors became **Fire Punch, The Drifting Classroom, Leviathan** — exactly right. Those LLM-assigned vectors live in the same 254-tag basis as the manga, so Western titles drop straight onto the shared map.

**The lesson:** the elegant method (embeddings) lost to the "dumb" one (read it and tag it). Mechanism over elegance.

---

## 6. Mining 3.5 GB (the Grand Comics Database)

To go beyond a hand-list of Western titles, we pulled the **GCD** full database dump — a 3.5 GB MySQL file, 74 tables, ~108K English series. No MySQL server needed: a streaming parser (`scripts/parse_gcd.py`) reads the `INSERT` statements line-by-line into a small SQLite, joining *series → issue → story* to aggregate the per-story `genre` field up to series level.

The pipeline (`query_gcd_dark.py` → `gcd_curate.py` → `gcd_broaden.py`):

1. genre-filter to dark/action series → ~25K
2. quality gates (real publishers, GN-sized, real synopsis) → ~1K
3. an LLM tags the genuine ones and **skips the noise** — promos, "Free Comic Book Day," reprints, bloodless capes, licensed tie-ins.

Honest yield: **3.5 GB → ~120 usable titles.** Modest, exactly as predicted up front — but real, and several connect to creators already in the set (Eduardo Risso of *100 Bullets* → *Moonshine*, *Chicanos*; Brian Wood of *Northlanders* → *Briggs Land*, *Starve*).

---

## 7. IDF — weighting by what makes a title *distinctive*

A problem surfaced: **88% of the Western titles carry "Tragedy," 60% carry "Gore."** Those near-universal tags dominate the geometry while carrying almost no discriminating information — *everyone* has them.

Fix: **inverse document frequency.** Weight each tag by

```
idf(tag) = log( (N + 1) / (df + 1) ) + 1
```

where *df* is how many titles carry the tag. This **down-weights the common** (Tragedy, Gore) and **up-weights the rare** (Cosmic Horror, Space, Detective). The payoff: anchor-search sharpens — "find titles like Berserk" starts matching on Berserk's *distinctive* fingerprint (Cosmic Horror, Philosophy) instead of generic tragedy, surfacing truer neighbors (Vinland Saga, Übel Blatt, Fire Punch).

It's a toggle in the app, because it changes what "similar" *means* and you may want either.

> A thing IDF **didn't** fix: it barely spread the Western cluster (spread ratio 0.10 → 0.16). That cluster is *genuinely* tight — every Western title was selected to be dark+violent, so they really are similar. No reweighting separates a set that's actually homogeneous. Honest negative result.

---

## 8. The Western-only universe

Since the Western titles are squished into one corner of the *Eastern-defined* map, there's a second view: fit a **fresh PCA on the 306 Western vectors alone**. Now the axes are defined by *Western* variance, and the sub-structure appears:

- **PC1:** apocalyptic-horror (*'68*, *B.P.R.D.*) ⟷ crime-noir (*Sin City*, *100 Bullets*, *Powers*)
- **PC2:** cerebral/cosmic-horror (*Providence*, *Gideon Falls*) ⟷ action-violence (*Punisher MAX*, *Hellsing*)

Also a toggle.

---

## What this project actually taught me

1. **Clever loses to simple more often than you'd like.** Embeddings → mush; "read it and tag it" → correct.
2. **Productive paranoia pays.** Every method got a validation gate *before* it shipped (does *Crossed* land near *Fire Punch*?). Two methods died at the gate.
3. **The map's 2D dot is a lossy summary.** When it disagrees with the full-dimensional nearest-neighbors, trust the neighbors.
4. **More data ≠ more signal.** Adding 60 dark comics densified the blob instead of spreading it — because they were all the same *kind* of thing.
5. **State your bottleneck honestly.** This model is label-starved, not method-starved. Saying so is more useful than a prettier chart.

## Honest caveats

- The supervised model is **modestly predictive (~0.67 AUC)**, not an oracle. It needs more labeled titles.
- **Western titles are NOT in the supervised model** — they're a thematic overlay, placed by tag-similarity (validated by nearest-neighbor), not by predicted verdict.
- GCD's `genre` field is coarse; it's a *discovery filter*, not a tagging source.

---

## Run it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. pull the Eastern corpus + tags from AniList
python scripts/fetch_anilist.py
python scripts/fetch_staff.py          # artist / "more by this artist"
python scripts/fetch_synonyms.py       # alt-title search
python scripts/fetch_tag_descriptions.py

# 2. fit the taste model (edit data/user_library_seed.json with YOUR labels)
python scripts/model.py

# 3. Western layer (needs your own Comicvine key in comicvine_key.env)
python scripts/fetch_western_desc.py
python scripts/build_western_from_tags.py   # uses LLM-assigned tags

# 4. (optional) GCD mining — drop the dump in grand_comics_database/
python scripts/parse_gcd.py series
python scripts/parse_gcd.py genre
python scripts/gcd_broaden.py

# 5. layouts + the app
python scripts/build_layout.py
python scripts/build_html.py           # -> outputs/taste_map_app.html
```

The LLM-tagging steps (Western titles → AniList tags) were done by hand-driving an LLM with the tag vocabulary in `data/anilist_tag_vocab.json`; the resulting tags live in `data/western_llm_tags.json`.

## Data sources & licenses

- **[AniList](https://anilist.co)** — tags, scores, staff, synopses (GraphQL API).
- **[Comicvine](https://comicvine.gamespot.com/api/)** — Western catalog (needs a free key).
- **[Grand Comics Database](https://www.comics.org)** — bulk Western catalog, CC-BY 3.0.

The 3.5 GB GCD dump and the derived `gcd.sqlite` are **not** committed (size + redistribution). Download the dump yourself from comics.org and run the parse scripts.

---

*Built across one long session of asking better questions, killing two approaches that didn't survive their own validation, and being honest about a 0.67.*
