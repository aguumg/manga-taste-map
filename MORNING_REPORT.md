# Morning Report — Western World Expansion

**Date:** 2026-06-05 · **Status:** Built, validated, integrated. Open `outputs/taste_map_app.html`.

## TL;DR
Your manga taste-model app now includes a **Western comics layer** — 183 titles (incl. all you've read + ~190 mined from your 3 forum PDFs), placed on the *same* taste map as your 2,007 Eastern titles, with their own ratings tab. Reload the app to see it.

## The one decision that mattered: I threw away my first approach
- **Original plan (description-embeddings):** embed each Western comic's plot description, project into your AniList tag-space. **It FAILED validation.** Comicvine served *German editions* (Crossed's description was literally in German) and *publication metadata* (Y: The Last Man's "description" was a trade-paperback index). The embedding tagged Crossed as *Villainess / 4-koma / Satire* and placed it next to villainess-romance manhwa. Garbage. You called this risk before I ran it.
- **The fix (LLM tag-assignment):** had an LLM read each title and assign real AniList tags from knowledge. Crossed → `Gore 1.0, Survival, Post-Apocalyptic, Pandemic, Tragedy, Sadism, Rape, Cannibalism, Torture`. This is the "smart hand-tagging" you suggested at the very start.

## Validation (honest)
**Placement works** — nearest-Eastern-neighbor test passes:
- Crossed → *Drifting Classroom, Fire Punch, Leviathan* (dark gore-survival) ✓
- Sara → *All You Need Is Kill, Saga of Tanya, Nausicaä* (war/military) ✓
- Y: The Last Man → *Heavenly Delusion, Leviathan, Eden* (post-apoc) ✓
- Scalped → *Death Note, Akira, Goth* (crime/dark) ✓

**Limit (stated plainly):** the map places Western titles by *theme* correctly, but it does **NOT predict your loved-vs-meh verdicts yet**. With only 6 Western labels, and because "meh" is about *execution* (Locke & Key is thematically dark but you found it meh→now liked), theme-tags can't separate them. Western titles are therefore **NOT in the supervised model** — they're a thematic overlay. This improves as you rate more Western titles.

## What's in the app
- **Map:** Western titles as teal triangles (▲) on the unified PCA map. Toggle "Show Western titles" in the Anchor tab. Hover = tags + nearest Eastern.
- **Western tab:** all 183, searchable, expand for synopsis + tag chips + nearest-Eastern + Berserk-cosine. Rate them (loved/liked/meh/disliked) — persists under `taste_western_v1`, exports in the `western` section.
- **Pre-seeded verdicts:** Crossed=loved, Sara=loved, Y=liked, Saga=liked, Crécy=meh, **Locke & Key=liked** (your update).

## Data sources evaluated
- **Comicvine** — used for the title catalog; its "concepts" are junk (cover-variant metadata), descriptions polluted. Not used for tags.
- **Grand Comics Database (comics.org)** — researched per your ask. Free CC-BY MySQL dump (bi-weekly, registration required), but genres are *coarse* (horror/crime/war) and it's a multi-GB all-or-nothing download. **Verdict: not worth it for tagging** (LLM-tags are finer); only worth it if you later want to auto-expand to *thousands* of Western titles as a catalog. Parked as "Phase 4: catalog expansion."

## What to verify (your eyes, not the math)
1. Open the app → Western tab → spot-check a few tag assignments against comics you know.
2. On the map, toggle Western on, anchor on Berserk, see which Western triangles fall near your seinen-dark cluster.
3. Rate 10–15 Western titles you've read → Export → send me the file → then it's worth me folding Western into the supervised model.

## Open questions for you
- Want Western titles eventually IN the supervised model? (needs ~15+ Western ratings first.)
- Want the GCD catalog-expansion (thousands of titles), or is the curated 183 enough?

## UPDATE — GCD catalog expansion (you said "add 1 + 2")
Mined your 3.5GB Grand Comics Database dump (no MySQL needed — streamed into `data/gcd.sqlite` via `parse_gcd.py`). Pipeline: 108K English series → genre-join → 24,657 dark → 971 curated (quality publishers, GN-sized, real synopsis) → 120 second-pass cleaned → **60 LLM-tagged** (tagger skipped 60 promos/FCBD/Image-Firsts/crossovers/Top-Cow-filler/licensed-manga). Merged → **map now has 242 Western titles** (183 + 59 new).

New dark gems added (several tied to creators you rate): **Moonshine** & **Chicanos** (Eduardo Risso, artist of your 100 Bullets), **Briggs Land** & **Starve** (Brian Wood, your Northlanders), **Revival, '68, Deadworld, Joe Golem, Criminal Macabre, The Divided States of Hysteria, Cry Havoc, Turf, Body Bags, Creepy, Frankenstein Mobster**, etc.

Honest yield: 3.5GB dump → ~60 usable titles. Modest for the effort (as predicted), but real, and the GCD pipeline now exists (`gcd.sqlite` + parse/query/curate scripts) to re-mine anytime. Caveat unchanged: placement is thematic; verdict-prediction still needs you to rate Western titles. Tag-vocab gaps noted by tagger: no Werewolf/Supernatural/Psychological/Death keys → substituted Curses/Demons/Philosophy/Tragedy (Moonshine/Redneck/Two Moons lean slightly off-genre as a result).
