# X thread draft — manga-taste-map

> Draft. Punchy version of the README story for a general/tech audience. Run it through your own voice before posting — swap in your screenshots at the marked spots. ~13 posts.

---

**1/**
I asked an AI to "recommend me manga like Berserk."

It turned into a 2,300-comic taste map you can explore in your browser, a supervised model of my own taste, and two approaches that failed in instructive ways.

The math thread 🧵

*[attach: screenshot of the map]*

---

**2/**
You can't do math on "Berserk." You can do math on a vector.

AniList readers vote tags with intensity 0–100. Berserk =
Tragedy 97 · Gore 88 · Cosmic Horror 90 · Philosophy 86 …

Every comic → a point in 254-dimensional tag space. No scraping. The crowd already labeled it.

---

**3/**
Give it titles you loved and disliked, and "taste" becomes a classifier.

Logistic regression + a nearest-shrunken-centroid model, scored by leave-one-out cross-validation.

Honest result: AUC ~0.67. Modestly predictive. Not magic. I refuse to report the flattering training number.

---

**4/**
Why only 0.67? **p ≫ n.**

254 features, ~30 labeled titles. More knobs than examples = overfitting heaven.

The bottleneck was never the method. It was rows. More labels moved it; re-rating titles it already knew did nothing.

---

**5/**
Then the model found something I didn't tell it:

k-means on my favorites splits PERFECTLY into two groups —
• Japanese seinen-tragedy (Berserk, Vinland, AoT)
• Korean RPG-progression (Solo Leveling, Overgeared)

I'm not one taste. I'm two.

---

**6/**
This breaks naive recommenders.

"Find titles near the average of what you like" fails, because the centroid of a 2-peak cloud lands in the empty valley between them — and the nearest stuff there is generic isekai.

Your average is not a thing you like.

*[attach: bimodal cluster screenshot]*

---

**7/**
Now the fun part. Western comics — Crossed, Saga, Scalped — aren't on AniList. No tags. How do you put them on the same map?

Attempt 1, the "smart" way: embed each comic's plot description, project onto the tags by similarity.

It produced garbage.

---

**8/**
Crossed — one of the goriest comics ever made — got tagged **Villainess / 4-koma / Satire** and placed next to romance manhwa.

Why? The database served the *German* edition (description in German), and the text was a *trade-paperback index*, not a plot.

Garbage in.

---

**9/**
Attempt 2, the "dumb" way: just have an LLM read each title and assign the real tags from knowledge.

Crossed → Gore 1.0, Survival, Post-Apocalyptic, Pandemic, Cannibalism…

Now its nearest neighbors are Fire Punch & The Drifting Classroom. Exactly right.

Clever lost to simple.

---

**10/**
To go bigger I pulled the Grand Comics Database — a **3.5 GB** SQL dump, 108K series.

Streamed it into SQLite (no server), joined series→issue→story, genre-filtered, then let an LLM cull the noise (promos, reprints, bloodless capes).

Yield: 3.5 GB → ~120 good titles. Modest. Honest.

---

**11/**
A subtle one: 88% of my dark comics share the tag "Tragedy." It dominates the geometry while telling you nothing — everyone has it.

Fix: IDF weighting. idf = log(N/df). Down-weight the universal, up-weight the rare.

Now "like Berserk" matches on Cosmic Horror, not generic gore.

---

**12/**
What IDF *didn't* do: spread my Western cluster apart.

Because the cluster is genuinely tight — I only ever pick dark+violent comics, so they really are similar. No math separates a set that's actually homogeneous.

A negative result is still a result.

---

**13/**
Lessons:
• Clever loses to simple more than you'd like
• Validate before you ship (2 methods died at the gate)
• The 2D dot is lossy — trust the neighbors
• More data ≠ more signal
• Name your real bottleneck

Code + the live map: [repo link]
