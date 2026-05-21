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
