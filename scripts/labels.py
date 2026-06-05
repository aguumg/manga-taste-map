"""
labels.py — The reader's labelled titles + an AniList resolver.

Each label is (search_string, expected_country_set, approx_year, hint). The
resolver searches AniList, then VERIFIES the top candidates by country and
approximate start year to avoid same-name collisions, and records what matched
into label_resolution.csv for audit. Anything it cannot confidently match is
flagged (resolved=False) rather than silently guessed.

Three label groups:
  STRONG_POS  -> label 1, used to FIT the model
  NEG         -> label 0, used to FIT the model
  MOD_POS     -> held-out validation, NEVER used to fit; we only check that
                 the fitted model scores them above the corpus median.
"""

# (search term, {allowed countries}, approx year or None, free-text hint)
STRONG_POS = [
    ("Berserk",                              {"JP"}, 1989, "Miura seinen"),
    ("Vinland Saga",                         {"JP"}, 2005, "Yukimura"),
    ("Claymore",                             {"JP"}, 2001, "Yagi"),
    ("Mugen no Juunin",                      {"JP"}, 1993, "Blade of the Immortal"),
    ("Shingeki no Kyojin",                   {"JP"}, 2009, "Attack on Titan"),
    ("Solo Leveling",                        {"KR"}, 2018, "Na Honjaman Level-Up"),
    ("Jigokuraku",                           {"JP"}, 2018, "Hell's Paradise"),
    ("Overgeared",                           {"KR"}, 2017, "Korean RPG"),
    ("Pick Me Up",                           {"KR"}, 2022, "Infinite Gacha"),
    ("Murim RPG Simulation",                 {"KR"}, 2022, "murim rpg"),
    ("Surviving the Game as a Barbarian",    {"KR"}, 2022, "barbarian"),
    ("Eternally Regressing Knight",          {"KR"}, 2020, "regressing knight"),
]

NEG = [
    ("Hai to Gensou no Grimgar",             {"JP"}, 2013, "Grimgar"),
    ("Goblin Slayer",                        {"JP"}, 2016, "goblin slayer"),
    ("Jormungand",                           {"JP"}, 2006, "Takahashi"),
    ("Omniscient Reader",                    {"KR"}, 2020, "ORV"),
    ("Golden Kamuy",                         {"JP"}, 2014, "golden kamuy"),
    ("Made in Abyss",                        {"JP"}, 2012, "made in abyss"),
    ("Nano Machine",                         {"KR"}, 2020, "nano machine"),
    ("Absolute Sword Sense",                 {"KR"}, 2022, "absolute sword sense"),
    ("The Lone Necromancer",                 {"KR"}, 2021, "lone necromancer"),
    ("Second Life Ranker",                   {"KR"}, 2019, "second life ranker"),
    ("SSS-Class Suicide Hunter",             {"KR"}, 2020, "sss-class suicide hunter"),
]

MOD_POS = [
    ("Tower of God",                         {"KR"}, 2010, "ToG"),
    ("Terra ForMars",                        {"JP"}, 2011, "terraformars"),
    ("Solo Max-Level Newbie",                {"KR"}, 2021, "max level newbie"),
    ("Skeleton Soldier Couldn't Protect the Dungeon", {"KR"}, 2018, "skeleton soldier"),
    ("The Greatest Estate Developer",        {"KR"}, 2021, "estate developer"),
    ("Taming Master",                        {"KR"}, 2017, "taming master"),
    ("Arcane Sniper",                        {"KR"}, 2021, "arcane sniper"),
    ("Talent-Swallowing Magician",           {"KR"}, 2021, "talent swallowing"),
]

# Already-read titles that must NEVER be recommended (in addition to all
# labelled titles above). Resolved to ids the same way so we can exclude by id.
EXTRA_EXCLUDE = [
    ("Overlord",                             {"JP"}, 2012, ""),
    ("Arifureta",                            {"JP"}, 2016, ""),
    ("Jujutsu Kaisen",                       {"JP"}, 2018, ""),
    ("Memorize",                             {"KR"}, 2017, ""),
    ("Disastrous Necromancer",               {"KR"}, 2022, ""),
    ("Reincarnation of the Suicidal Battle God", {"KR"}, 2021, ""),
    ("The Novel's Extra",                    {"KR"}, 2017, ""),
    ("Bowblade Spirit",                      {"KR"}, 2016, ""),
    ("Lord Marksman and Vanadis",            {"JP"}, 2011, "Madan no Ou"),
    ("Tsurune",                              {"JP"}, 2016, ""),
]
