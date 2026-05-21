# Model changelog

A running log of *what the dynasty composite score actually changes* with
each PR. Read top-to-bottom to follow how the model evolves — what was
added, what shifted in the outputs, and the biggest player-level
movements.

Format for each entry:

- **What changed** — the mechanical change.
- **Why** — citation back to `docs/RESEARCH-sources.md` and/or external evidence.
- **Expected output shift** — qualitative and (where possible) quantitative
  predictions about which player cohorts move and in which direction.
- **Validation** — how we'll know in backtesting whether the change helped.

---

## v0.6.0 — Rookie college→NBA similarity chain (PR #7)

**Date:** 2026-05-21

**What changed.** The career-arc engine now projects rookies and current
draft prospects by chaining college similarity to NBA careers. The PR
#4/#5 NBA-only engine compared current NBA players against the
1980-present NBA corpus; PR #7 adds a parallel **NCAA D1 corpus**
(2008-2025, 55,528 player-seasons from barttorvik.com) and a
**college→NBA bridge** that maps every NBA player back to their college
season(s). For a draft-eligible player like Cooper Flagg the engine
now:

1. Vectorizes his 2024-25 Duke season in college feature space (per-36
   production + USG + class-relative age + conference strength).
2. KNNs over the NCAA corpus to find the top-20 most similar same-class
   (Fr/So/Jr/Sr) same-bucket player-seasons.
3. Looks up each comp's realized NBA career through the bridge.
4. Aggregates similarity-weighted, time-discounted 5%/yr.

The rookie projection (`rookie_dynasty_value`) is blended with the
existing NBA-side projection based on how many NBA seasons the target
has under his belt:

  * 0 NBA seasons → rookie projection only (pure college→NBA chain).
  * 1 NBA season  → 50/50 blend of rookie + NBA (the 1-year NBA sample
                    is noisy; college comps still informative).
  * 2+ NBA seasons → NBA-side projection only (PR #4 behavior).

**Why.** The PR #4 engine couldn't project pure rookies because they
have no NBA seasons to vectorize. It also misprojected 1-year players
because the lone NBA sample drove all the signal. The college side
restores meaningful comp data for players entering the draft and the
blend logic stabilizes the year-1 projection.

**Data sourcing.** The PR spec called for sports-reference.com/cbb but
that endpoint returns 403 from cloud / CI IPs (Cloudflare-style anti-
bot). We substituted **barttorvik.com**’s public `getadvstats.php`
endpoint, which:

  * Covers every D1 player-season 2008–present (~85K rows pre-filter).
  * Already exposes advanced stats (USG, TS%, BPM-equivalent).
  * Returns clean JSON in one round-trip per season.
  * Is bot-friendly and rate-limit-tolerant at 1s/request.

The substitution is documented in `RESEARCH-sources.md`. NCAA data
before 2008 is unavailable from barttorvik; the bridge naturally
limits to 2008+ NBA-bound college players. Pre-2008 NBA stars (Kobe,
KG, LeBron, much of the 2000s draft class) get NBA-only similarity,
which is the existing PR #4 behavior — no regression.

**Bridge coverage.** 801/1947 historical NBA players (41.1% raw) match
an NCAA player-season; restricted to bridgeable players (post-2008
debuts), coverage is **77.0%**. The 23% gap is dominated by
international players (Jokic, Giannis, Schroder, Capela, Sarr,
Buzelis, Risacher, ...) who never played NCAA. Documenting that an
international/G-League prospect bridge is a future PR — the existing
NBA-only fall-through handles them correctly today.

**Composite integration.** The career_arc adapter's emitted ranking
records now span both NBA-cohort players AND pure-rookie NCAA
prospects (filtered to ~580 players passing a prospect threshold:
MPG≥22 plus P5/HM BPM≥4 or USG≥22, OR BPM≥7 at any conference). The
blended dynasty_value sits in the composite alongside the existing
DARKO/CC/Vecenie/BBRef signals.

**Expected output shift (rookie class):**

  * **Cooper Flagg** — top college comps Paolo Banchero, Jayson Tatum,
    Jabari Smith, Brandon Ingram, Tobias Harris. Projected NBA career
    ≥ 10 seasons (test invariant). Blends 50/50 with NBA projection.
  * **Dylan Harper** — top comps Cade Cunningham, D'Angelo Russell,
    Collin Sexton. Pure-college projection ~12-14 seasons.
  * **VJ Edgecombe** — top comps Marcus Smart, Dejounte Murray.
  * **Jeremiah Fears, Stephon Castle** — perceived less elite by the
    pure-college engine; both bridge to NBA careers like SGA / Kemba
    via Castle and Tony Wroten / Sindarius Thornwell via Fears.
  * **Long tail:** a handful of high-BPM low-major freshmen the engine
    rates highly (Thomas Sorber, JT Toppin, Jayden Quaintance, Austin
    Rapp) appear in the rookie top-20 by college vector despite low
    consensus rank. This is the expected noise of an unanchored
    college→NBA model; future PRs can add a draft-stock prior (RSCI /
    ESPN 100) to attenuate.

**Limitations documented in `ROOKIE-SIMILARITY.md`:**

  * NCAA-only — international prospects (Wemby, Risacher, Sarr,
    Buzelis) fall back to the NBA-only similarity engine. PR target:
    add FIBA / Adidas Next Gen / G League Ignite bridges.
  * Per-36 inflation for low-MPG bench bigs — the prospect filter
    requires MPG≥22 to suppress this.
  * Censored-comp extrapolation extends still-active careers to a
    typical exit age (34 star / 30 role / 27 bench). Without it, a
    comp like Tatum gets capped at his realized 8 seasons, which
    underprojects rookies who match him.
  * Conference-strength multiplier is a flat 1.00 / 0.92 / 0.83 / 0.75
    by tier (P5/HM/MM/LM). A future PR could regress conference
    strength against actual NBA outcomes for a smoother adjustment.

**Validation.** New tests in `tests/test_rookie_similarity.py` cover:

  * NCAA corpus size ≥ 50K rows.
  * Bridge coverage ≥ 70% of post-2008-debut NBA players.
  * Cooper Flagg projection ≥ 10 NBA seasons.
  * A low-major NCAA Sr fixture projects ≤ 4 NBA seasons.
  * 50/50 blend math at n_nba_seasons=1.
  * International players (no NCAA match) produce `rookie_dv=None`
    and fall back to NBA-only dv unchanged.
  * Bridge alias hits (e.g. `Nic Claxton` ↔ `Nicolas Claxton`).
  * Batched KNN returns identical top similarities to single-target.
  * Barttorvik payload parser drops <10 MPG rows.

Full test count: **108 passing** (90 pre-PR baseline + 18 new).

**Performance.** Full pipeline (load NBA + NCAA corpora, build bridge,
project 500 NBA players + 580 NCAA prospects across both formats):
~85s on a single core. The NCAA KNN is batched into 200-target
matmuls to fit the 55K×16 candidate matrix into memory.

---

## v0.5.0 — Name resolver + dedup pass (PR #6)

**Date:** 2026-05-21

**What changed.** Added a 4-tier player-name resolver (`src/dynasty_bball/
name_resolver.py`) and wired it into the sync layer. Every incoming
source record now passes through:

1. Tier 1 — canonical key (lowercase + diacritic-fold + suffix-strip +
   punctuation-strip). Catches `Dončić`↔`Doncic`↔`DONCIC`,
   `LeBron James Jr.`↔`LeBron James`, `T.J.`↔`TJ`.
2. Tier 2 — last name + first-initial + position bucket + team abbrev.
   Catches `Nicolas Claxton`↔`Nic Claxton` (same `n`+`claxton`+`BKN`).
3. Alias map (`data/name_aliases.json`) — hand-curated edge cases like
   `Bones Hyland`↔`Nah'Shon Hyland`, `Bub Carrington`↔`Carlton
   Carrington`, `Nigel Hayes-Davis`↔`Nigel Hayes`.
4. Tier 3 — conservative fuzzy. Last name equal, position bucket
   compatible, **same team**, first-name prefix/diminutive/similarity
   guard. Never crosses team boundaries.

DARKO and Court Consensus emit full team names ("Washington Wizards");
the resolver normalizes those to 3-letter abbrevs (`WAS`) so they
join cleanly onto Sleeper's `WAS` rows. A post-sync `dedup_players_
by_canonical()` pass then walks the Player table and merges any
leftover duplicates carried in from the pre-PR database.

**Why.** Before PR #6 the top-300 had 5-7 duplicate rows per snapshot
— every "Nicolas Claxton" / "Alexandre Sarr" / "Carlton Carrington"
entry double-counted because DARKO's full-name spelling never joined
onto Sleeper's short-form row. Those duplicates also lacked the
Basketball-Reference / career-arc similarity signal (no join → no
stats join), so they showed up as orphan rows with `data-position=""`
and verbose team names. Phil flagged this directly:

> If the model cannot locate the name of the player it is looking for
> in the basketball reference database… it should be a dead giveaway
> that it is a name / unique identifier issue. … use the last name of
> the player and do a fuzzy match type of code on the first name.

**Known dupes fixed in this PR (with previous rank pairs):**

| Player                                  | Before                          | After       |
| --------------------------------------- | ------------------------------- | ----------- |
| Alex Sarr / Alexandre Sarr              | #30 (orphan) + #39 (C, WAS)     | one row     |
| Nic Claxton / Nicolas Claxton           | #38 (orphan) + #85 (C, BKN)     | one row     |
| Bub Carrington / Carlton Carrington     | #150 (orphan) + #270 (PG, WAS)  | one row     |
| Bones Hyland / Nah'Shon Hyland          | #40 (orphan) + #300 (PG, MIN)   | one row     |
| David Jones / David Jones Garcia        | #97 (orphan) + (SAS row)        | one row     |
| Nigel Hayes / Nigel Hayes-Davis         | #221 (orphan) + (PHX row)       | one row     |

**Excluded players.** Any Sleeper-tracked player that doesn't match a
Basketball-Reference row through all four tiers + alias map is dropped
from the rankings and listed on `sources.html#unmatched`. On the
shipped DB this list is empty.

**Expected output shift.** Top-300 shrinks by ~6 rows after dedup. The
remaining row for each merged player keeps its better-of identity
(position + 3-letter team) and the *summed* market value from both
rows' source contributions, which can nudge those players up a few
spots in the composite (more sources corroborating them).

**Validation.** Site banner now shows `300 players · N name variants
merged · N unmatched`. Tier counts are persisted to `data/diagnostics/
resolver_stats.json` per sync so we can track resolver behavior over
time. New unit tests in `tests/test_name_resolver.py` cover every
canonical case (29 tests, all green) and pin the no-false-merge
invariant on different surnames.

---

## v0.4.0 — Career-arc similarity engine (PR #4)

**Date:** 2026-05-21

The biggest model change since v0.1. Implements Phil's dynasty thesis:

> One player that seems to be ranked too low is Cooper Flagg. Also some
> of the older players like James Harden and Kawhi Leonard seem to be
> ranked too high. The idea with dynasty is that the leagues go on
> forever and the players production ends when that player retires from
> the NBA. The DARKO score is saying Cooper Flagg will retire at 28?
> That is clearly not right. Let's use a higher weight towards
> similarity scores using the players age and production of fantasy
> stats (and remaining stats for their career) to arrive at the player
> rankings.

### What changed

**New source: `career_arc`** (`sources/career_arc.py`, `similarity/`).

- For every NBA player-season since 1980 (cached under
  `data/historical_nba/league_<season>.json`), we compute a profile
  vector of per-36 production rates (PTS / REB / AST / STL / BLK /
  3PM / TOV), usage proxies (FGA/36, FTA/36), efficiency (TS%), and
  durability (GP/82, MPG). Z-score normalized across the corpus.
- For each current player at age A, we find the top-20 historical
  player-seasons at age A±1 in the same / adjacent position bucket
  (PG→SG→SF→PF→C; bucket derived from stat shape, not roster label).
- We aggregate those comps' **actual remaining careers** into:
  - `projected_remaining_years` = similarity-weighted median of comps'
    remaining seasons.
  - `projected_total_fantasy_points` = similarity-weighted, 5%/yr
    time-discounted sum of (comp ppg × comp games per remaining season)
    — in the league's actual scoring (so DHK and default diverge).
  - `per_year_survival_prob[1..15]` = fraction of comps still playing
    at age A+k.
  - `dynasty_value` = `projected_total_fantasy_points` rescaled
    0..100 across the current cohort.
- The adapter emits two RankingRecords per player (DHK + default) and
  writes a sidecar JSON at `data/career_arc/comparables.json` carrying
  the top-5 comps per player. The report page renders the comparables
  on each player's detail view ("Most similar historical players at
  this age: 1. LeBron James (age 19, sim=0.94)...").

**New weights:**

| Source                | Old weight | New weight | Rationale |
|-----------------------|-----------:|-----------:|-----------|
| `career_arc`          |        —   |    **1.8** | New dominant longevity signal. |
| `darko`               |       1.5  |        0.8 | Demoted to current-skill only; longevity now owned by career_arc. |
| `basketball_reference`|       1.2  |        1.0 | Stays as current-year production signal, on par with Court Consensus. |
| `court_consensus`     |       1.0  |        1.0 | Unchanged (market signal). |
| `vecenie`             |       0.5  |        0.5 | Unchanged (rookie-only filter, low weight). |

### Why

DARKO's survival model bakes a single Bayesian retirement-age estimate
into every player. The model is calibrated against the population of
active NBA careers but doesn't condition on profile — it just sees age
and recent production. As a result it assigns ~age-28 retirement to a
19-year-old superstar wing whose comp pool says he'll play to 38. For
young high-ceiling players this is a catastrophic miscalibration
because longevity dominates dynasty value.

The similarity engine sidesteps this by using **actual observed comp
careers** rather than a parametric survival fit. For Cooper Flagg at
age 19, his top comps include Carmelo Anthony (17 more seasons),
Kevin Durant (16+), Anthony Edwards (4 visible so far + still active)
— a much richer prior than DARKO's curve.

For old high-usage stars, the inverse is true: Harden at 36 finds
comps in late-career LeBron (5 more), late-career Curry (2 more),
Kobe at 36 (1 more). The weighted median is ~3 years — reality, not
DARKO's continued mid-tier projection.

### Expected output shift

**Risers** (model corroborated post-implementation):

| Player           | Age | BEFORE rank | AFTER rank | Δ   |
|------------------|----:|------------:|-----------:|----:|
| Cooper Flagg     |  19 |          26 |          3 | +23 |
| Jeremiah Fears   |  19 |         160 |         31 | +129|
| Dylan Harper     |  20 |          90 |         27 | +63 |
| VJ Edgecombe     |  20 |          43 |          9 | +34 |
| Stephon Castle   |  21 |          62 |         33 | +29 |
| Paolo Banchero   |  23 |          33 |         14 | +19 |
| Evan Mobley      |  24 |          17 |          5 | +12 |

**Fallers** (model corroborated post-implementation):

| Player           | Age | BEFORE rank | AFTER rank | Δ    |
|------------------|----:|------------:|-----------:|-----:|
| LeBron James     |  41 |          92 |        225 | -133 |
| Jimmy Butler     |  36 |          89 |        211 | -122 |
| Stephen Curry    |  38 |          52 |        169 | -117 |
| Paul George      |  36 |         132 |        230 | -98  |
| Kevin Durant     |  37 |          35 |        123 | -88  |
| James Harden     |  36 |          42 |        113 | -71  |
| Kawhi Leonard    |  34 |          16 |         78 | -62  |
| Anthony Davis    |  33 |          39 |         82 | -43  |
| Joel Embiid      |  32 |          32 |         64 | -32  |

### Example comparable lists

**Cooper Flagg (19yo, 21.0 PPG / 7.0 RPG / 4.0 APG)** — top 5 comps:
  1. Carmelo Anthony 2003-04 (age 20, sim 0.927) — 17 remaining seasons,
     21.5 fppg dhk
  2. Kevin Durant 2007-08 (age 19, sim 0.916) — 16 remaining seasons,
     27.8 fppg dhk
  3. Anthony Edwards 2020-21 (age 19, sim 0.907) — 4 remaining (active),
     24.8 fppg dhk
  4. RJ Barrett 2019-20 (age 20, sim 0.899)
  5. Paolo Banchero 2022-23 (age 20, sim 0.897)

**Victor Wembanyama (22yo, 25.0 PPG / 12.9 RPG / 3.8 APG / 3.0 BLK)** — top 5 comps:
  1. Kristaps Poržiņģis 2017-18 (age 22, sim 0.947) — 6 remaining seasons
  2. Jaren Jackson Jr. 2021-22 (age 22, sim 0.936)
  3. Anthony Davis 2015-16 (age 23, sim 0.930) — 9 remaining, 32.3 fppg dhk
  4. Kristaps Poržiņģis 2016-17 (age 21, sim 0.908)
  5. Karl-Anthony Towns 2018-19 (age 23, sim 0.907) — 6 remaining, 27.6 fppg dhk

**James Harden (36yo, declining usage)** — top 5 comps:
  1. LeBron James 2019-20 (age 35, sim 0.948) — 5 remaining (active)
  2. LeBron James 2020-21 (age 36, sim 0.927) — 4 remaining (active)
  3. Stephen Curry 2022-23 (age 35, sim 0.897) — 2 remaining (active)
  4. J.J. Barea 2018-19 (age 35, sim 0.894) — 1 remaining
  5. Kobe Bryant 2014-15 (age 36, sim 0.887) — 1 remaining

Weighted median remaining years: **3.0**. Compared with DARKO's prior
estimate of ~7+ years for a player at this DPM — the survival fit
over-extends late-career high-skill guards because they hold DPM
longer than they hold roster spots.

### Validation

- Tests pin the Flagg-top-15 invariant and the Harden-falls invariant
  against the cached corpus (`tests/test_similarity.py`).
- Long-term: once we have a backtest pipeline (PR #6+), the comp's
  observed remaining production becomes the ground truth. Run
  `career_arc` against frozen historical snapshots (predict Joel
  Embiid's 2020-21 dynasty arc using only 2015-16 data, compare to
  what actually happened) and let the track-record multiplier float
  the weight to whatever the data supports. Initial 1.8 is set by
  thesis priority, not by backtest — will be revisited.

### Forward hooks (PR #5)

The vectorize / comparables / projection modules are deliberately
agnostic to NBA vs. college input. PR #5 will add a college→NBA
bridge corpus (sports-reference college stats vectorized in the same
feature space after pace + league-strength translation) so we can
rank rookies the same way we rank pros — by who they're most
similar to among historical pre-NBA prospects, then chained to those
players' NBA careers. Stub: `similarity.vectorize.vectorize_college_season`.

---

## v0.3.0 — Basketball-Reference per-game production via nba_api (PR #3)

**Date:** 2026-05-21

Resolves the open issue from PR #2: `points_dhk` and `points_default`
rendered **identical** composite rankings because every source we had
so far (DARKO, Court Consensus, Vecenie) was scoring-agnostic — they
all emit the same `market_value` regardless of league format. This PR
lands the first source whose `market_value` is *format-dependent*, so
the two formats finally diverge.

**Basketball-Reference adapter** (`sources/basketball_reference.py`)

- Category: `model`. Default weight: 1.2. Sits between DARKO (1.5,
  forward-looking impact + longevity) and Court Consensus (1.0,
  community consensus). BBRef is backward-looking *realized* per-game
  production — hard ground truth, but stale by definition, hence the
  middle weight. Track-record multiplier will float it once the
  backtest pipeline lands.
- Source: NBA Stats API via [`nba_api`](https://github.com/swar/nba_api)
  (LeagueDashPlayerStats endpoint, `PerMode=PerGame`,
  `SeasonType=Regular Season`). One call per launcher run.
- **Cached JSON** at `data/basketball_reference/leaguedash_<season>.json`.
  The cache is checked into the repo so the daily CI build is
  deterministic and never hammers stats.nba.com (they rate-limit hard
  from CI ranges). Live refresh is gated behind
  `DYNASTY_BBALL_BBREF_LIVE=1` or cache absence.
- Default season is `2025-26` (configurable via
  `DYNASTY_BBALL_BBREF_SEASON`). Two cache files ship in the repo:
  2024-25 and 2025-26.
- Min-games floor: 10 GP. Filters out one-game cameos whose per-game
  numbers are noise.
- **Format-aware `market_value`**: for each player we compute
  `fantasy_ppg` under both `points_dhk` and `points_default`
  using `scoring.LEAGUE_SCORING` (the single source of truth for
  stat weights). The top scorer in each format rescales to
  `market_value=100`; everyone else tapers proportionally. This is
  the first signal in the model that *differs by format* — it's why
  the two composite tables now diverge.
- Per-game counters (pts/reb/ast/stl/blk/tov/3pm) propagate onto the
  `RankingRecord` so downstream views can show realized production
  alongside the DARKO impact metrics.
- v1 ships without DD% / TD% — LeagueDashPlayerStats does not expose
  them and per-game logs would 30x the API cost. Follow-up PR will
  add a PlayerGameLogs cache for that.

**Scoring formulas** (per PR scope)

Phil's Dynasty Hoop Kings league (`points_dhk`):
```
fantasy_ppg = pts*0.5 + reb*1.0 + ast*1.0 + stl*2.0 + blk*2.0
            + tov*(-1.0) + 3pm*0.5
```

Generic Sleeper points league (`points_default`):
```
fantasy_ppg = pts*1.0 + reb*1.2 + ast*1.5 + stl*3.0 + blk*3.0
            + tov*(-1.0) + 3pm*0.5
```

Both come straight from `scoring.LEAGUE_SCORING`. DD%/TD% bonuses and
technical/flagrant deductions are still defined in DHK's scoring dict
but intentionally not applied by BBRef in v1 (no data).

**Expected ranking shifts** (BBRef-only signal, no composite mix)

Stocks merchants (high stl/blk/reb, modest scoring) move UP in DHK
because `stl=2.0 / blk=2.0 / reb=1.0` while `pts=0.5`. Pure scorers
move UP in DEFAULT because `pts=1.0` dominates.

From the live 2025-26 cache:

| Player                | DHK rank | DEFAULT rank | Delta |
|-----------------------|----------|--------------|-------|
| Amen Thompson         | 31       | 38           | −7    |
| Ausar Thompson        | 110      | 119          | −9    |
| Dyson Daniels         | 52       | 64           | −12   |
| Kevin Durant          | 28       | 20           | +8    |
| Devin Booker          | 43       | 30           | +13   |
| Trae Young            | 90       | 73           | +17   |
| Klay Thompson         | 264      | 247          | +17   |
| Stephen Curry         | 35       | 25           | +10   |

(Negative delta = ranks higher in DHK.)

Top-15 in each format is also visibly different. DHK pushes Jalen
Johnson into the top 5 (high stocks); DEFAULT favors high-volume
scorers like Anthony Edwards and Jamal Murray.

**Composite impact**

BBRef enters the composite via the same value-based normalization
path DARKO / Court Consensus use. The DHK and DEFAULT composite
tables now produce DIFFERENT top-15s for the first time — verified
in a launcher run on 2026-05-21 against the cached BBRef payload
(see PR #3 description for the side-by-side table).

**Validation**

- 48 unit tests pass (`pytest tests/ -q`). New tests in
  `tests/test_basketball_reference_parser.py` (15 tests, fixture-only,
  no network) cover:
  - parser shape + min-games floor + empty-payload tolerance
  - **format divergence**: Amen Thompson ranks higher in DHK than
    DEFAULT; Devin Booker ranks higher in DEFAULT than DHK; the
    two ordered lists are not identical.
  - hand-checked `fantasy_ppg` against known DHK + DEFAULT formulas
  - adapter cache-first behavior + missing-cache fallback
- Smoke test still passes end-to-end.
- Live launcher run on 2026-05-21 confirmed BBRef sync: 1012 rows
  (506 players × 2 formats), composite scoring 506 players in each.

**Caveats**

- nba_api wraps stats.nba.com which is rate-limited per IP. CI relies
  on the checked-in JSON cache; live refresh from CI is opt-in only.
- 2025-26 cache reflects the season-to-date snapshot at the time of
  this PR — will be refreshed periodically by re-running with
  `DYNASTY_BBALL_BBREF_LIVE=1`. Cache files are gitignored under no
  rule, so they ride with the repo.
- The BBRef signal is *retrospective*. A young player who breaks out
  mid-season will be undervalued by BBRef and overvalued by DARKO;
  composite blending is intentional — PR #4 will add
  forward-projection sources (Hashtag Basketball / Basketball Monster)
  to balance.

---

## v0.2.0 — Court Consensus baseline + Sam Vecenie CSV slot (PR #2)

**Date:** 2026-05-21

Addresses Phil's PR #1 feedback that DARKO-only inflates rookies:
Knueppel (#8), Queen (#14), and Clingan (#15) sat too high because
DARKO's longevity bonus (`years_remaining * 2.5`) compounds with a
decent rookie DPM. Two new sources sand this down by adding the
consensus+expert anchor the model was missing.

**Court Consensus adapter** (`sources/court_consensus.py`)

- Category: `market`. Default weight: 1.0. Lower than DARKO (1.5)
  because Court Consensus does not carry impact or longevity
  signals — it's pure crowd-derived ELO. Both float on track-record
  multiplier once a Production loader lands.
- The site is a Vite/React SPA backed by a public Supabase project.
  Tier-1 fetch hits the same `/rest/v1/players` endpoint the SPA
  itself uses (anon JWT publicly embedded in the JS bundle; read-
  only via RLS). One round-trip per launcher run, with a polite UA.
- Tier-2 (HTML) is a stub today; tier-3 reads a local CSV at
  `data/court_consensus/court_consensus_dump.csv` if anything ever
  changes upstream — same fallback pattern as DARKO.
- Filters out CC's PICK rows (`2026 Pick 1.01`, `2026 Early 1st`,
  etc.) by both `position == "PICK"` and a name regex. Emits one
  `RankingRecord` per real NBA player to **both** `points_dhk` and
  `points_default` because CC's points-league ELO is the right
  signal for both formats.
- `market_value` is the ELO rescaled to 0–100 so it lands in the
  same band as DARKO's composite scalar, giving the value-based
  scoring branch a clean magnitude match.

**Sam Vecenie adapter** (`sources/vecenie.py`)

- Category: `expert`. Default weight: 1.3. Elevated single-analyst
  weight — Vecenie has a documented strong NBA-draft hit rate; the
  football-side analog would be Lance Zierlein.
- CSV-only adapter (Vecenie's Big Board is paywalled at The Athletic
  and cannot be scraped). User drops a CSV at
  `data/vecenie/vecenie_big_board.csv` and the launcher picks it up
  on the next refresh. Schema documented in `data/vecenie/README.md`.
  Until the CSV exists the adapter yields zero rows and the launcher
  continues without error — same pattern as the football repo's
  RAS / CFBD adapters.
- Registered in `weights.ROOKIE_SIGNAL_SOURCES`: players whose ONLY
  ranking comes from Vecenie are filtered out of the top of the
  composite (the "draft-prospect squatting" guard). Once a player
  makes their first NBA appearance, DARKO and Court Consensus pick
  them up automatically and the filter releases.
- `market_value` tapers linearly from rank: rank 1 → 100, max-rank → 0.
  Same magnitude band as Court Consensus and DARKO.

**Launcher wiring**

- `launcher_headless.sources_to_sync` now lists both new adapters.
- `_duplicate_rankings_to_format` was narrowed to operate on DARKO
  only — CC and Vecenie emit per-format records directly during
  fetch, so the helper no longer touches their rows. Once a real
  production-based adapter lands and DARKO is also per-format, the
  helper becomes a no-op.
- The `report.py` sources page picks up both new entries from
  `SOURCE_DESCRIPTIONS`.

**Expected output shift**

Observed against the live model run (2026-05-21):

| Player          | PR #1 rank  | PR #2 rank  |
|-----------------|-------------|-------------|
| Kon Knueppel    | #8 (DARKO)  | **#12**     |
| Derik Queen     | #14 (DARKO) | **#19**     |
| Donovan Clingan | #15 (DARKO) | **#24**     |
| Cooper Flagg    | top-10ish   | **#29**     |

Veterans at the top (Wemby, SGA, Jokic, Luka, Edwards, Tatum,
Giannis) stay where Phil wants them — DARKO and CC agree on the
elite tier so the composite is stable there. Court Consensus is
moving the rookies because CC's community ranks Knueppel #34,
Queen #37, and Clingan #43 — see the `rank_divergence` column for
the magnitude (Knueppel +22, Queen +18, Clingan +19).

**Validation**

- 33 unit tests pass (`pytest tests/ -q`). New tests:
  - `tests/test_court_consensus_parser.py` (9 tests, fixture-based,
    no network) — verifies PICK filtering, ELO→value rescale,
    position normalization, position-rank assignment, format
    propagation.
  - `tests/test_vecenie_csv.py` (10 tests) — verifies CSV parsing,
    rank→value taper, missing-file tolerance, both-format emit.
- Smoke test still passes end-to-end with the fake sources.
- Live launcher run on 2026-05-21 confirmed:
  - DARKO: 526 rows (unchanged)
  - Court Consensus: 606 rows (303 players × 2 formats)
  - Vecenie: 0 rows (no CSV — expected behavior)
  - Composite scores: 530 players in each format
- Post-merge: confirm `/sources.html` shows three entries, the
  rookies have visibly dropped on the published rankings page, and
  `consensus_rank` / `rank_divergence` populate on per-player views.

**Caveats**

- The hardcoded Supabase anon key may rotate. If it does the
  adapter logs a warning, falls back to CSV, and the launcher
  continues without error. A future PR can sniff the bundle to
  rediscover the key automatically.
- Vecenie's signal stays inert until someone drops the CSV. Phil:
  see `data/vecenie/README.md` for the format — a 30-row top board
  is enough to start moving things.

---

## v0.1.0 — Initial repo scaffold + DARKO foundation (PR #1)

**Date:** 2026-05-20

This PR establishes the basketball repo as a parallel to
Dynasty-Football-Model and lands the first source: DARKO.

**Scaffold**

- Python package `dynasty_bball` under `src/`, mirroring the football
  repo's `dynasty` layout.
- SQLAlchemy 2.0 schema in `src/dynasty_bball/db/models.py`. Player table
  carries `sleeper_id`, `nba_id`, `bbref_id`, `espn_id`, `yahoo_id`,
  `fantrax_id`, `position` (PG/SG/SF/PF/C), `nba_team`,
  `est_retirement_age`, `years_remaining`. Production table stores raw
  per-game counting stats so we can re-score for any league setting
  on demand.
- Default `league_format` everywhere is `points_dhk` (Phil's Dynasty
  Hoop Kings league). `scoring.LEAGUE_SCORING` is the single source
  of truth for the stat-to-points map.
- Deterministic weighting model carried over verbatim from
  Dynasty-Football-Model v0.10:
  `effective_weight = default_weight × track_record_multiplier`.
  No hand-coded per-(source, position) overrides, no years-pro decay.
- Site generator at `src/dynasty_bball/report.py`. Visual reference:
  courtconsensus.com — clean white/dark NBA-orange theme.
- Headless launcher + GitHub Actions daily refresh wired to Pages.

**DARKO adapter** (`sources/darko.py`)

- Talks to DARKO's Shiny app over its WebSocket protocol — there is no
  documented JSON API, so we replicate what the page itself does.
  Protocol details documented in the module docstring.
- Pulls both league-wide tables: `table` (current-season DPM /
  per-100 stats, ~526 players) and `surv_table` (survival /
  longevity model, ~655 players).
- Joins the two by normalized player name (handles "T.J. McConnell" vs
  "TJ McConnell" etc. via the suffix-aware normalizer in
  `dynasty_bball.names`).
- Composite "longevity-adjusted DPM" scalar lands in
  `RankingRecord.market_value`:

  ```
  score = 50 + (dpm * 5) + (years_remaining * 2.5)
          + max(0, dpm_improvement) * 5
  ```

  Tuned so:

  | Cohort                            | Approx scalar |
  |-----------------------------------|---------------|
  | Peak elite DPM (+7, 7 yrs left)   | ~102          |
  | Solid vet (+2, 6 yrs left)        | ~75           |
  | Fading vet (0, 1 yr left)         | ~52           |
  | Rising rookie (0, 11 yrs left)    | ~78           |

- DPM, O-DPM, D-DPM, DPM Improvement, years_remaining, and
  est_retirement_age are also persisted as `Evaluation` rows for
  downstream views (per-player page surfaces them).
- **Resilience:** if the Shiny scrape fails, the adapter falls back to
  `data/darko/darko_dump.csv` if one exists; otherwise it yields zero
  records and the launcher's "starter pack fallback" (currently empty)
  kicks in — the site still builds with whatever rankings made it in.

**Default weight:** 1.5 — the highest in the model. Justified because
DARKO bundles impact metric + longevity in a single source, which no
public alternative does. Track-record multiplier will adjust this once
we have a Production loader and can backtest.

**Sleeper NBA player map**

- `sync_sleeper_players()` pulls `https://api.sleeper.app/v1/players/nba`
  and upserts every player into the canonical `players` table by
  `sleeper_id`. This is the spine that joins DARKO names → Sleeper IDs
  → other future sources.
- Endpoint is ~few-MB JSON, marked `update_frequency=weekly`.

**Scoring**

- `compute_composite_scores()` writes a `composite_scores` snapshot.
  Defaults to `points_dhk`. Launcher also generates `points_default`
  (standard Sleeper NBA points) by duplicating DARKO's Rankings to the
  second format — DARKO's scalar doesn't depend on scoring weights,
  and downstream production-based adapters (next PR) will emit
  format-specific rows directly. Helper documented in
  `launcher_headless._duplicate_rankings_to_format`.

**Expected output shift**

- N/A (first PR). Top of the rankings should match top of DARKO's DPM
  table, with longevity nudging young players (Wemby, Chet, Holmgren)
  upward and older stars (LeBron, KD, Curry) slightly down.

**Validation**

- Smoke test passes with two fake sources (`tests/smoke_test.py`).
- DARKO parser test runs against a real fixture
  (`tests/fixtures/darko_sample.json`) with NO NETWORK.
- Manual sanity check at launcher time: top 10 should include Wemby,
  Jokic, SGA, Luka, Giannis, Tatum, Brunson, Curry-tier names. Logged
  in the PR description.
