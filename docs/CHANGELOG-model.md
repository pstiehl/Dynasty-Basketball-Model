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
