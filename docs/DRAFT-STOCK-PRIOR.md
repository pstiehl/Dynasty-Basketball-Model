# Draft-stock prior (PR #8)

PR #7's college→NBA similarity engine surfaces every prospect that
passes a BPM/PPG/USG filter, but mid-major freshmen with one hot year
slipped into the rookie top-20 because the barttorvik corpus has no
notion of *"and an NBA team also believes in this guy."* The
draft-stock prior closes that gap with a one-multiplier-per-prospect
lookup against the real NBA Draft history.

## TL;DR

```python
from dynasty_bball.sources.draft_stock import load_index, compute_multiplier

index = load_index()                                  # 1,073 prospects, 2008-2025
mult, tier, prospect = compute_multiplier(
    name="Cooper Flagg",
    index=index,
    college_rec_rank=100.0,
)
# -> (1.20, "top_5", DraftProspect(name="Cooper Flagg", consensus_rank=1, ...))
```

The multiplier scales `rookie_dynasty_value` in
`career_arc.build_rookie_projections` *before* the cohort is rescaled
to 0–100. Real lottery picks float to the top; undrafted high-BPM
mid-major freshmen sink.

## Multiplier table

| Tier label | NBA draft signal | Multiplier |
|---|---|---|
| `top_5` | Pick #1–5 | **1.20×** |
| `lottery` | Pick #6–14 | **1.10×** |
| `first_round` | Pick #15–30 | 1.00× (baseline) |
| `second_round` | Pick #31–60 | 0.85× |
| `rec_rank_prior:elite` | Undrafted, consensus top-10 HS recruit (rec_rank ≥ 99.5) | 0.85× |
| `not_on_board:fourstar` | Undrafted, 4-star recruit (95.0 ≤ rec_rank < 99.5) | 0.30× |
| `noise` | Undrafted, sub-4-star or unranked | 0.30× |

The `rec_rank` fallback is intentionally conservative: a recruiting
percentile is a much weaker signal than an actual NBA team writing a
guaranteed contract, so even a true 5-star without a draft outcome
never beats the baseline `first_round` multiplier. The empirical PR #7
noise sources (Thiam, Toppin, Rapp, Quaintance) all sit in the
4-star-or-below band — exactly where the harshest 0.30× penalty bites.

## Data source choice

We use [`nba_api`'s `DraftHistory`](https://github.com/swar/nba_api)
endpoint exclusively, for three reasons:

  1. **Already in the dependency tree.** PR #4 introduced nba_api for
     historical NBA stats. No new dependency, no new auth, no Cloudflare
     fights with Basketball-Reference's draft pages.
  2. **Authoritative.** Pick numbers from `stats.nba.com` are the
     official record. No need to reconcile multiple mock-draft sources.
  3. **Backfillable.** Pulling 2008–2025 in one batch (with a 2-sec
     polite delay) takes ~40 seconds and produces 1,073 prospects
     covering every player whose college career overlaps with our
     NCAA corpus (2008–2025).

We considered and rejected:

  * **Tankathon / NBADraft.net big boards** — their HTML uses
    JavaScript-rendered components that readability-style fetchers
    strip out. Would need a real headless browser.
  * **Basketball-Reference `/draft/NBA_<year>.html`** — Cloudflare
    challenge wall blocks unattended fetches.
  * **247Sports / RSCI** — high-school recruiting data, not NBA draft
    data. Less directly informative for the *NBA outcome*. We use
    barttorvik's `rec_rank` as a lightweight stand-in.

## Name-resolver integration

The big-board cache is keyed by `name_resolver.canonical_key`, which
strips diacritics, suffixes, punctuation, and case. At index-build
time we also fold in every entry from `data/name_aliases.json` so a
lookup for `"Carlton Carrington"` (barttorvik's spelling) finds the
NBA-drafted `"Bub Carrington"` record:

```python
from dynasty_bball.sources.draft_stock import load_index

idx = load_index()
idx.lookup("Carlton Carrington").consensus_rank   # 14 (2024 draft)
idx.lookup("Bub Carrington").consensus_rank       # also 14
```

When the rookie engine resolves a player by `entry["name"]` it always
goes through the canonical key, so any future alias additions land
both in the resolver and in the draft-stock lookup automatically.

## Where the multiplier is applied

```
src/dynasty_bball/sources/career_arc.py::build_rookie_projections
  ...
  rescale_to_0_100(projections_per_format)            # initial rescale
  apply_multipliers_to_rookie_entries(entries, idx)   # PR #8 prior
  rescale_to_0_100(projections_per_format)            # final rescale
  return entries
```

The double-rescale is intentional: the first rescale puts every player
on a common 0–100 scale so the multiplier values are directly
comparable across formats; the multiplier then redistributes mass; the
final rescale re-normalizes so the cohort still spans 0–100 (otherwise
the top boost would push the leader to 120 and break downstream
consumers that assume a 0–100 range).

`apply_multipliers_to_rookie_entries` scales **both** `dynasty_value`
(displayed value) and `dynasty_value_raw` (the input the final
rescale reads). Skipping the `_raw` field would silently revert the
multiplier — that's the bug we caught during development.

## Per-player rendering

Every player page with a college projection now shows a draft-stock
badge in the rookie-headline line:

```
rookie dynasty_value 100.0 · projected NBA seasons 12.3 · draft stock top_5 (pick #1 2025, src=nba_api) ×1.20
```

The badge format degrades gracefully:

  * Drafted player → `top_5 (pick #N YYYY, src=nba_api) ×M`
  * Elite undrafted recruit → `rec_rank_prior:elite (no NBA draft entry; rec_rank prior) ×0.85`
  * Pure noise → `noise (undrafted, not on any board) ×0.30`

`/sources.html` has a "Draft-stock prior" card with the full multiplier
table and source attribution.

## Refresh cadence

`data/draft_stock/big_board.json` is committed to git and only changes
after the actual NBA draft (annually, late June). Rebuild via:

```bash
DYNASTY_BBALL_DRAFT_LIVE=1 python scripts/refresh_draft_stock.py
```

The env-var gate prevents CI from hammering `stats.nba.com`. CI
**must not** refresh the cache — it reads the committed JSON.

## Test coverage

`tests/test_draft_stock_prior.py` has 18 tests across five categories:

  1. **Pure functions** — tier-to-multiplier mapping, rec_rank
     percentile mapping.
  2. **Synthetic fixtures** — top-5 boost, lottery boost, baseline,
     noise penalty, fourstar penalty, elite floor.
  3. **Name resolver** — Carlton ↔ Bub alias resolution.
  4. **Real cache shape** — 1,073 prospects loaded, every spec-named
     real lottery pick present, every spec-named noise prospect absent.
  5. **End-to-end invariants** — against the committed NCAA + draft
     caches: Flagg stays top-3, Sorber stays top-10, the 5 real
     lottery picks stay top-15, no noise survives in the top-15, no
     noise outranks a real lottery pick.

All 108 PR #7 + 90 baseline tests still pass.

## Future improvements

  * **Forward-looking big boards.** Currently the cache only contains
    actual draft results. For pre-draft prospect evaluation (e.g.
    rookies whose draft year hasn't happened yet) we'd want a live
    mock-draft scrape. Tankathon and NBADraft.net are the candidates.
  * **Per-pick precision.** A `top_5` boost currently treats #1
     identically to #5. We could parametrize on the actual pick
     number for a smoother gradient.
  * **Tier-track-record calibration.** Backtest second-round picks
    against realized fppg — the 0.85× multiplier is a prior, not a
    measurement.
