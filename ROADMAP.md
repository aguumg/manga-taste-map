# Roadmap / Future features — taste map

Backlog of ideas not yet built. Pull from here when picking the next version.

## Big future versions (Agus's ideas, 2026-06-06)
- **Anime map** — same machinery, anime corpus. Note: anime adaptations often pad/filler vs the manga (Agus dislikes this: Terraformars vol 2-3, Golden Kamuy). A taste model could even score "is the anime worth it over the manga."
- **Light-novel map** — likely the *next* one after the manga map. LN-sourced titles currently can't go on the manga map (Re:Zero, 86, Steins;Gate, Apothecary Diaries, Mushoku Tensei, Monogatari, Legend of the Galactic Heroes).

## Curation principle (IMPORTANT — Agus feedback 2026-06-06)
- **Do NOT filter by "seinen" demographic.** He doesn't like seinen as a category. He likes **mature / dark / violent CONTENT**. Filter by content tags (Gore, Tragedy, Body Horror, Survival, Psychological…), not by demographic label.
- **MAL/global top-lists are low-yield** for him: every strong-fit title from MAL top-anime + top-manga (Monster, Goodnight Punpun, 20th Century Boys, Takopi's Original Sin, Hunter x Hunter, Chainsaw Man, Land of the Lustrous, Ashita no Joe) was **already in the corpus** (top-2000 by popularity covers the popular dark stuff). To find NEW titles, mine by dark content tags below the popularity cut, not from top-lists.

## Candidate titles to add (not yet in corpus)
Verified MISSING from the current map (2026-06-06):
- **Steel Ball Run** (JoJo Pt 7, manga) — "maybe" fit: brutal but stylized, not grimdark.
- **Lord of Mysteries / Guimi Zhi Zhu** (manhua) — "maybe" fit: dark Lovecraftian, slow-burn.
Graphic novels missing (from the 8-list curation, GN = allowed scope):
- **Arkham Asylum: A Serious House on Serious Earth** (Morrison/McKean) — clear dark fit; add via Western/LLM path.
- **Stitches** (David Small) — childhood medical/family-trauma memoir; softer, borderline.
- **Daytripper** (Bá/Moon) — mortality meditation; devastating but literary, borderline.
Decide per-title; only Arkham is a slam-dunk.

## KEY FINDING (2026-06-06): coverage is essentially complete for his taste
Two independent sweeps (MAL top lists + 8 best-of/horror/devastating lists) returned the SAME result:
**every strong-fit manga/manhwa they name is already in the corpus.** 24/24 from the 8-list curation
were already on the map (PunPun, Monster, Homunculus, Dorohedoro, Parasyte, Tomie, Gantz, Sweet Home…).
The top-2000-by-popularity corpus already saturates his dark/explicit taste. → Discovery is NOT the
bottleneck. Stop hunting external lists. The lever is (a) USING the app (anchor + read the dark cluster),
and (b) if expanding, mine BELOW the popularity cut by dark content tags, not best-of lists.
The 12 STRONG curated picks were loaded into the Reading Shelf as To-read (source "curated").

## UX backlog (from the proactive UX audit, not yet implemented)
P1:
- No-anchors / missing-anchor guard: if the centroid degenerates to a zero vector, show a hint instead of silent all-equal scores.
- Paused vs Disliked legibility: consider a 5th shelf status "Dropped" (oxblood) distinct from "Paused" (revisit). Surface first ~40 chars of the note on the collapsed row.
- `optSummary()`: add a `· highlight` token when highlight-top is on.
- Anchor stat bar: append `· (W scores estimated)` when Western items appear under model-score lens.
- Genre filter: animate the "All" chip when the active set snaps back to all (so the click reads as "did something").
- Disabled-not-hidden: when IDF/Western-only data is missing, show the toggle disabled with a tooltip instead of hiding it.
P2:
- "Read (unrated)" legend label uses the same ✕ glyph as Disliked — change symbol/label so they're distinguishable.
- Shelf note field is single-line `<input>`; switch to `<textarea rows=1>` to match Library/Western.
- Mobile: anchor-suggestion dropdown can clip inside the scroll container — portal it or raise z-index.

## QA harness (built 2026-06-06)
- `scripts/ux_smoke.mjs` — Playwright headless test. Loads the app, clicks every tab, checks shelf links / →map / add-title auto-link / session persistence, and captures JS console errors. Run after every build:
  `node scripts/ux_smoke.mjs outputs/taste_map_app.html`
- Standing rule: run this after each app change before declaring done.
- **Claude Chrome extension**: Agus has it. Complements this — it can drive his *real* logged-in browser (visual, interactive) for things headless can't (his actual localStorage state, real reading-source sites). Use the extension for exploratory "play with it like a human"; use ux_smoke.mjs for repeatable regression checks.
