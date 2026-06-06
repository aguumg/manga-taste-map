# manga-taste-map

**An interactive map of ~2,300 comics, positioned by how close each one is to your taste.**

Open `index.html` (or `outputs/taste_map_app.html`) in any browser — it's a single offline file. Every dot is a comic. Where it sits, what color it is, and what it's near are all computed from a model of what *you* like. Anchor it on *Berserk* — or on a mix like *Berserk + Crossed + Vinland Saga* — and the whole map recolors by "how close is this," ranking Eastern manga and Western comics in one pool. Rate things and the model learns.

It started as a one-line request — *"recommend me manga like Berserk"* — and turned into a small, honest data-science project. This README is the whole story, including the math and the parts that failed.

---

## 1. Turning a comic into a vector

You can't do math on "Berserk." You can do math on a **vector**.

[AniList](https://anilist.co) lets thousands of readers vote *tags* on each title, each with an **intensity rank 0–100**. Berserk is:

```
Tragedy 97 · Seinen 95 · Revenge 94 · Demons 92 · Cosmic Horror 90
Gore 88 · Philosophy 86 · Found Family 86 · Anti-Hero 83 · Swordplay 82 ...
```

Every title becomes a point in **254-dimensional space** (coordinate *j* = the crowd-voted intensity of tag *j*, 0 if absent). That vector *is* the comic, to the model.

> **The unlock:** no scraping, no sentiment analysis. The intensity-ranked tag vote *is* a ready-made, human-labeled feature matrix.

Corpus: top ~2,000 manga/manhwa/manhua by popularity from the AniList GraphQL API.

## 2. Learning your taste (the supervised model)

Give it titles you **loved** and **disliked**, and taste becomes a classifier — **L2-regularized logistic regression** + a robust **nearest-shrunken-centroid**.

**The honest problem (`p ≫ n`):** 254 features, ~30–60 labeled titles. More knobs than examples → overfitting. So we never report training accuracy — only **leave-one-out cross-validation**:

| Labels | Logistic LOO-AUC | NSC LOO-AUC |
|-------:|-----------------:|------------:|
| 23 | 0.629 | 0.636 |
| 43 | 0.630 | 0.725 |
| 59 | **0.670** | 0.679 |

~0.67 is *modestly* predictive — better than a coin flip, not magic. The lesson: **more labels move it; re-confirming titles the model already knows does not.** The bottleneck was always rows.

## 3. You are not one person (bimodality)

**k-means (k=2)** on the *liked* vectors splits cleanly, zero errors:

- **Mode 0 — seinen-tragedy:** Berserk, Vinland Saga, Claymore, Attack on Titan, Blade of the Immortal, Hell's Paradise *(all Japanese)*.
- **Mode 1 — Korean RPG-progression:** Solo Leveling, Overgeared, Pick Me Up, Murim RPG Simulation, Surviving as a Barbarian *(all Korean)*.

So "recommend the average of what I like" *fails*: the centroid of a bimodal cloud lands in the empty valley between the peaks, and the nearest mass there is generic isekai. **Your average is not a thing you like.** Recommend per-mode, never blended.

## 4. Drawing the map (PCA)

254-D vectors → 2D via PCA. **PC1** ≈ action/fantasy ⟷ romance/slice-of-life; **PC2** ≈ seinen/tragedy ⟷ comedy. The 2D dot is a *lossy summary* — for "is this actually similar," trust the full-254-D cosine, not the dot's location.

## 5. The Western problem (clever broke, simple won)

Western comics — *Crossed*, *Y: The Last Man*, *Saga*, *Scalped* — **aren't on AniList**. No tags.

**Attempt 1, description embeddings — failed.** Embed each comic's plot, project onto the tags by similarity. *Crossed* (one of the goriest comics ever) got tagged **Villainess / 4-koma / Satire** and placed next to romance manhwa. Why? The database served the **German edition** (plot in German) and the "description" was a **trade-paperback index**, not a plot. Garbage in.

**Attempt 2, LLM tag-assignment — won.** Have a model read each title and assign the real AniList tags from knowledge: *Crossed* → `Gore 1.0, Survival, Post-Apocalyptic, Pandemic, Cannibalism…`. Its nearest neighbors became **Fire Punch, The Drifting Classroom, Leviathan** — exactly right. Those vectors live in the same 254-tag basis, so Western drops onto the shared map. **Mechanism over elegance.**

## 6. Mining 3.5 GB (the Grand Comics Database)

The **GCD** full dump — 3.5 GB MySQL, 74 tables, ~108K English series. No server: a streaming parser reads the `INSERT`s into SQLite, joins *series→issue→story*, aggregates the `genre` field. Then quality gates + an LLM cull the noise (promos, reprints, bloodless capes). Honest yield: **3.5 GB → ~120 usable titles** — modest, but real, and several tie to creators already in the set (Eduardo Risso → *Moonshine*, *Chicanos*; Brian Wood → *Briggs Land*, *Starve*).

## 7. IDF — weighting by what makes a title *distinctive*

88% of the dark titles share the tag "Tragedy" — it dominates the geometry while saying nothing. Fix: `idf(tag) = log((N+1)/(df+1)) + 1`, down-weighting the common, up-weighting the rare. Now "like Berserk" matches on Cosmic Horror, not generic gore. *(It barely spread the Western cluster, though — those titles are genuinely homogeneous. No reweighting separates a set that's actually similar. A negative result is still a result.)*

## 8. The Western-only universe

A second view fits a fresh PCA on the 306 Western vectors *alone*, so they spread by their own variance: PC1 = apocalyptic-horror ⟷ crime-noir, PC2 = cerebral-horror ⟷ action-violence.

---

## What this taught me

1. **Clever loses to simple** — embeddings → mush; "read it and tag it" → correct.
2. **Validate before you ship** — two methods died at their own gate (does *Crossed* land near *Fire Punch*?).
3. **The 2D dot is lossy** — when it disagrees with the nearest-neighbors, trust the neighbors.
4. **More data ≠ more signal** — 60 more dark comics densified the blob instead of spreading it.
5. **Name your real bottleneck** — this model is label-starved, not method-starved.

## Honest caveats

- Supervised model is **~0.67 AUC** — modestly predictive, not an oracle.
- **Western titles** are LLM-tagged (lower fidelity than crowd tags); they carry a **cross-domain** model score, flagged "est." Folding their verdicts into training (down-weighted) was a measured **wash** (0.670 → 0.664) — too few labels, all positive.
- GCD's `genre` is a *discovery filter*, not a tagging source.

---

## Run it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_anilist.py        # corpus + tags
python scripts/model.py                # fit (edit data/user_library_seed.json with YOUR labels)
python scripts/build_western_from_tags.py   # Western layer (LLM-assigned tags)
python scripts/build_layout.py
python scripts/build_html.py           # -> outputs/taste_map_app.html
```

GCD mining (optional) needs the 3.5 GB dump in `grand_comics_database/`; Comicvine steps need your own key in `comicvine_key.env`. The dump and `gcd.sqlite` are **not committed** (size + redistribution).

## Data sources

[AniList](https://anilist.co) (tags/scores/staff) · [Comicvine](https://comicvine.gamespot.com/api/) (Western catalog, free key) · [Grand Comics Database](https://www.comics.org) (bulk Western catalog, CC-BY 3.0).

---

## Part 2 — the interactive app

The map became a full explorer:

- **Cross-universe anchoring** — pick *any* mix of titles across both worlds (Berserk + Crossed + Vinland Saga) as a centroid; the joint Eastern∪Western pool ranks against it.
- **IDF toggle** (distinctive-theme weighting) · **Western-only sub-map** (the 306 on their own axes).
- **Linked brushing** — click a list row to halo its dot; click a dot to open its row.
- **Per-axis ratings** (Art/Story/Characters/Pacing) · **tag-preference voting** · **same-artist lookup** · **synonym search** · collapsible controls.
- **Three score lenses** per title: `model` (supervised), `anchor` (cosine to your centroid), `taste` (your tag votes).
- **↗ Read links** (configurable source — MangaDex / batcave.biz / your own Suwayomi).
- **Responsive** — works on phone and iPad; add it to your home screen.

Western titles carry *derived genres* (from their tags), so the genre filter treats them exactly like Eastern. Built and debugged live — the Western-genre-exclusion bug only surfaced from a console screenshot.

*Built across one long session: asking better questions, killing two approaches that didn't survive their own validation, and being honest about a 0.67.*

---

## Part 3 — the reading shelf, and a bug that hid in plain sight

The obvious next thing was a place to keep *what to read*. A **Reading Shelf** — to-read, reading, paused, done — where every title links back to its dot on the map. One distinction mattered: **paused is not disliked.** "I bounced off it" and "I hated it" are different facts, so they're different states; only the second one feeds the model.

Then a useful failure. I kept fixing the shelf's map-links, rebuilding, verifying every title resolved — *Demon Slayer → Kimetsu no Yaiba*, *Hell's Paradise → Jigokuraku* — and the owner kept reporting them broken. We were both right. My new build was correct; his browser was running the old state. The shelf persists to `localStorage`, and the load step kept the **stored** entries verbatim — so every fix I shipped to the seed was silently overridden by a copy saved from older edits. The lesson is old and keeps being true: **the artifact you test is not always the artifact the user runs.** The fix: on load, refresh the *computed* fields (the map links, read URLs) from each new build, while keeping the *human* fields (status, notes).

Two checks now guard the lazy version of that mistake. Two independent sweeps (MAL's top lists, then eight "best dark manga" lists) both came back with the same answer — **every strong-fit title was already in the corpus**, so the bottleneck was never discovery, it was *using* the thing. And a **Playwright** harness drives the whole app headless before anything ships: it clicks every tab, adds a title, follows a link to the map, and even plants a stale shelf to prove the reload heals it. I trust those runs more than I trust "looks fine to me."
