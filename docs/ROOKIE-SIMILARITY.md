# Rookie College→NBA Similarity (PR #7)

The PR #4/#5 career-arc engine projects active NBA players by KNN over a
1980-present NBA corpus. That engine couldn't project rookies — they have
no NBA seasons to vectorize. PR #7 extends it: vectorize the rookie's
college season, find his most similar NCAA player-seasons, look up each
comp's realized NBA career through a NCAA→NBA bridge, and aggregate as
usual.

## Pipeline

```
NCAA D1 corpus (2008-2025, 55,528 player-seasons)   from barttorvik.com
        ↓
  build_college_corpus_profiles
        ↓
  vectorize_college_season(target)
        ↓
  find_college_comparables_batch (KNN, same-class, ±1.5 age)
        ↓
  for each comp: bridge_lookup(btv_pid → nba_id) → realized NBA career
        ↓
  censored extrapolation (still-active comps → typical exit age)
        ↓
  project_career  (sim-weighted, 5%/yr discount)
        ↓
  CareerProjection.rookie_dynasty_value
        ↓
  blended_dynasty_value(rookie_dv, nba_dv, n_nba_seasons)
        ↓
  career_arc emits RankingRecord (nba_id or "ncaa:<btv_pid>")
```

## College vectorization

`similarity.vectorize.vectorize_college_season(row, conference_multiplier)`
returns a `Profile` in a 16-dim college feature space (parallel to but
distinct from the 12-dim NBA space). Features:

  * Per-36 production: PTS, REB, AST, STL, BLK, 3PM, TOV-proxy, FGA, FTA
  * TS%, GP/35, MPG (durability proxies)
  * USG%, conference strength multiplier, class progress (0..3 for Fr..Sr),
    age-relative-to-class

Conference strength multiplier is applied to the production dimensions
so a 20-PPG SEC freshman beats a 20-PPG Sun Belt freshman in the same
KNN.

Tiers (`historical_ncaa.CONFERENCE_TIER`):

  * **P5 = 1.00** — ACC, B10, B12, SEC, P12, BE
  * **HM = 0.92** — A10, AAC, MWC, WCC
  * **MM = 0.83** — Sun Belt, MAC, Horizon, Patriot, CAA, ...
  * **LM = 0.75** — everything else (small conferences)

Each NCAA corpus z-scores against ITSELF — we do NOT share normalization
with the NBA corpus. College and NBA stat distributions are genuinely
different and forcing them into the same z-score space would distort
both engines.

## Position bucket from college stats

`_derive_college_bucket()` mirrors the NBA-side `derive_position_bucket`
but tuned for NCAA per-game ranges (college pace is slower, scoring
totals smaller) and uses listed height as a tiebreaker. Examples:

  * Anthony Davis (Kentucky 2011-12, 6-10, 14.2 RPG, 4.7 BPG) → C
  * Karl-Anthony Towns (Kentucky 2014-15, 6-11) → C
  * Cooper Flagg (Duke 2024-25, 6-9, 4.2 APG) → PF (height + playmaking)
  * Jayson Tatum (Duke 2016-17, 6-8) → PF
  * Stephon Castle (UConn 2023-24, 6-6, 2.7 APG) → SG
  * VJ Edgecombe (Baylor 2024-25, 6-5, 3.2 APG) → SG
  * Dylan Harper (Rutgers 2024-25, 6-6, 4.0 APG) → SG

The bucket is a soft filter — adjacent buckets are eligible with a
small similarity penalty.

## College→NBA bridge

`similarity.bridge.build_bridge(nba_rows, ncaa_rows)` walks every NBA
player and looks up their canonical_key in the NCAA corpus. Temporal
plausibility check: NCAA last season within 4 years of NBA debut.

Output schema (`data/bridge/ncaa_to_nba.json`):

```json
{
  "generated_at": "...",
  "n_nba_players_total": 2069,
  "n_nba_players_matched": 897,
  "match_rate": 0.43,
  "n_pre_corpus_nba_players": 906,
  "n_alias_hits": 6,
  "by_nba_id": {
    "<nba_id>": {
      "btv_pid": "<barttorvik_id>",
      "ncaa_name": "...",
      "school": "...",
      "ncaa_seasons": ["2024-25", ...],
      "match_tier": "canonical" | "canonical_ambiguous" | "alias"
    }, ...
  },
  "by_btv_pid": { "<btv_pid>": "<nba_id>", ... },
  "unmatched_nba_ids": [...]
}
```

### Coverage

  * **41.1% raw** (897/2069) — diluted by pre-2008 NBA stars who
    couldn't possibly bridge.
  * **77.0% of post-2008-debut NBA players** — the meaningful metric.
  * **The 23% unmatched gap** is dominated by international players
    (Jokić, Giannis, Schroder, Capela, Sarr, Buzelis, Risacher,
    Yabusele, ...) who never played NCAA.

### Alias map integration

The bridge consults `data/name_aliases.json` (curated by PR #6) so:

  * `Nic Claxton` ↔ `Nicolas Claxton` (BBRef full-name vs. preferred)
  * `Mo Bamba` ↔ `Mohamed Bamba`
  * `Bones Hyland` ↔ `Nah'Shon Hyland`
  * `Bub Carrington` ↔ `Carlton Carrington`

all bridge correctly.

## Censored extrapolation

A comp like Jayson Tatum is still-active. His realized 8 NBA seasons
are a LOWER bound on his career, not a final answer. Without
extrapolation, Flagg's projection caps at 8 years because his closest
comps are all young censored stars.

`rookie._extrapolate_censored(comp)` projects the censored comp out to
a typical exit age:

  * Star (≥18 fppg dhk OR default) → exit at age 34
  * Role (≥8 fppg) → exit at age 30
  * Bench (<8 fppg) → exit at age 27

Career-length extrapolation is `max(realized, exit_age - debut_age + 1)`.
Games are scaled proportionally. Ages_after is extended year-by-year so
per-year survival probability stays correct.

## Longevity median excludes zero-NBA comps

The spec is explicit: comps who didn't make the NBA "weight their
contribution to longevity as zero". So `project_rookie` computes
`projected_remaining_years` over the NBA-having subset only, then
attaches that median back to the full-cohort projection. Fantasy
points and per-year survival probability still aggregate over the
full comp list (a zero-NBA comp correctly contributes zero
present-value fantasy points and zero probability of being in the NBA
at age+N).

Without this split, Cooper Flagg projects 7 years instead of the
spec-required 10+, because the 5 zero-NBA comps among his top-20 drag
the weighted median down.

## Blend logic

For players who already have NBA experience, the rookie projection and
the NBA-side projection are blended:

```python
def blended_dynasty_value(*, rookie_dv, nba_dv, n_nba_seasons):
    if n_nba_seasons <= 0:  return rookie_dv          # pure rookie
    if n_nba_seasons == 1:  return 0.5*rookie + 0.5*nba   # noisy 1-yr sample
    return nba_dv                                       # PR #4 behavior
```

Rationale: the 1-NBA-season case is the most uncertain — too small for
the NBA-side similarity to lock onto, but a real productive sample.
The college comps are still informative. At 2+ seasons, the NBA side
has enough signal to drop the college prior.

## Prospect filter

The career_arc adapter doesn't emit a RankingRecord for every D1
rotation player — that would be 3,000+ noise rows. We filter to
plausible NBA prospects:

  * MPG ≥ 22 (per-36 stats are unreliable below this; a 12-MPG bench
    big looking like KAT is the failure mode the filter prevents)
  * AND (any of):
    - P5/HM conference + (BPM ≥ 4 OR PPG ≥ 14 OR USG ≥ 22)
    - Any conference + BPM ≥ 7

~150-300 players pass per year, matching the rough combine + draft
pool. The KNN itself still searches the full corpus — the filter only
gates which prospects get emitted as ranking records.

## Site rendering

Each rookie / current-rookie-blend player page renders BOTH:

  * **Career-Arc Comparables** (top 5 NBA-comparables, PR #4 logic).
  * **Rookie / College Comparables** (top 5 college-comparables, PR #7).

When the player has 0 NBA seasons, only the college block appears.
The block carries a blend note explaining which projection drove the
composite dynasty value (`rookie_only` / `blend_50_50` / `nba_only`).

The `/rankings.html` page adds:

  * An **R** badge next to player names who are 0-1-NBA-season rookies.
  * A "Rookies only" checkbox filter.

## International / non-NCAA fallback

A current draft prospect with no NCAA season (Wemby, Sarr, Risacher,
Buzelis, Coulibaly, ...) has `rookie_dv = None`. The composite uses
their NBA-side projection unchanged (PR #4 behavior). Their player
page does NOT render the college block (no comps to show), only the
NBA comps if they have ≥1 NBA season.

**Open follow-up:** add international leagues bridges (FIBA / Adidas
Next Gen / G League Ignite) so Wemby-tier prospects also get a
production-based projection before they have NBA data. PR #7 is
NCAA-only and explicitly flags this gap.

## Performance

End-to-end pipeline (NBA + NCAA + bridge + rookie projections) runs
in ~85s on a single core:

  * Load NBA corpus: 0.1s
  * Load NCAA corpus (18 season JSONs): 1.2s
  * Build bridge: 0.3s
  * Build career indices: 0.0s
  * NCAA corpus profiles + z-score: 0.3s
  * NCAA search index: 0.1s
  * Batched KNN (3,259 NCAA targets × 55K corpus, 200-target chunks): ~30s
  * Per-target projection × 2 formats: ~25s
  * NBA-side comparables (existing PR #4): ~30s

The batched KNN uses one `(chunk, D) @ (D, N)` matmul per 200 targets
to amortize the matrix-vector dispatch cost. Without batching the
pipeline runs ~10× slower.

## Test invariants

`tests/test_rookie_similarity.py` enforces:

  * NCAA corpus ≥ 50K rows (spec target).
  * Bridge coverage ≥ 70% of post-2008-debut NBA players.
  * Cooper Flagg projection ≥ 10 NBA seasons.
  * A low-major NCAA Sr fixture projects ≤ 4 NBA seasons.
  * 50/50 blend math at n_nba_seasons=1.
  * International / no-NCAA fallback returns NBA dv unchanged.
  * Censored star extrapolation extends career to ≥12 seasons.
  * Bridge alias hits (`Nic Claxton` ↔ `Nicolas Claxton`).
  * Batched KNN matches single-target output.
  * Barttorvik payload parser drops <10 MPG rows.

18 new tests, all passing. Full suite 108 passing (90 baseline + 18 new).

## Known noise + limitations

1. **High-BPM freshmen at mid-majors** can outrank consensus top picks
   in the pure-college projection. Thomas Sorber (Georgetown, 14.5 PPG,
   2.0 BPG, BPM 7.5 Fr) and Austin Rapp (Portland WCC, similar tier)
   appear in the rookie top-10 by college vector despite consensus
   placing them outside the lottery. This is the expected noise of an
   unanchored college→NBA model; future PRs can layer in a draft-stock
   prior (RSCI / ESPN 100) to attenuate.

2. **Conference strength tier is flat** — 4 buckets with hand-picked
   multipliers. A future PR could regress conference strength against
   actual NBA outcomes for a smoother adjustment.

3. **Censored extrapolation thresholds are eyeballed** — exit ages
   34/30/27 by tier. With a real survival analysis on the historical
   corpus we could fit these properly per archetype.

4. **No international bridges** — see "International fallback" above.

5. **Pre-2008 NBA stars (Kobe, KG, LeBron, Carmelo)** can't appear as
   college comps for current rookies because barttorvik's coverage
   starts at 2008. The full set of valid college comps for a current
   freshman is thus narrower than the platonic ideal.
