# NBA dynasty source landscape — research notes

Living document. New entries land as we evaluate new sources. Format
borrowed from Dynasty-Football-Model's research doc.

For each source: what it is, why it matters, accessibility (API / ToS),
expected value-add to the composite, and whether it lands in PR #1 or
later.

---

## 1. DARKO — Daily Adjusted Regressed Kalman Optimized

**Site:** https://apanalytics.shinyapps.io/DARKO/
**Author:** Kostya Medvedovsky.
**Status in PR #1:** LIVE.

The only free, public, transparent NBA player-impact metric with a
built-in longevity model. Two outputs we care about:

1. **DPM (Defensive Plus Minus style)** — current-season impact metric
   broken into O-DPM, D-DPM, Box DPM. Updates daily during the season.
   Includes a "DPM Improvement" column (vs. prior season) which is a
   strong signal for young-player risers.
2. **Survival / longevity model** — for every player, estimated
   retirement age and per-year probability of still playing for the
   next 12 years. Exactly the input a dynasty model needs.

**Access:** Shiny app, no JSON API. Scraped via the WebSocket protocol
the page itself uses. Protocol details in `sources/darko.py` module
docstring. Outage risk is mitigated by a CSV fallback at
`data/darko/darko_dump.csv`.

**Default weight:** 1.5 (highest). Justified because DARKO is the only
source giving us *both* impact AND longevity. Once we have a Production
backtest, this weight floats with realized correlation.

---

## 2. Court Consensus

**Site:** https://courtconsensus.com/
**Status in PR #1:** Reference only (we're modeling the site's visual
look). Possible adapter in PR #2 if their published rankings are
scrape-friendly.

Aggregator of NBA expert dynasty rankings. The site itself is the
inspiration for our published-site look — clean white/dark theme,
NBA-orange accents, tidy ranking tables. Their methodology page should
be inspected before deciding what they contribute uniquely (vs. just
re-aggregating sources we already have).

---

## 3. Sam Vecenie (The Athletic)

**Site:** https://theathletic.com/author/sam-vecenie/
**Status in PR #1:** Not loaded.

Top-tier NBA draft analyst. His Big Board is the de-facto consensus
for pre-NBA prospect evaluation — analogous to Lance Zierlein for NFL.
Public top-N lists periodically; full board is paywalled. PR #2/#3
candidate: transcribe publicly-cited top-30 into a starter-pack-style
adapter.

Anticipated weight: 1.4 (rookie-signal source, highest among rookie
adapters). Will be added to `ROOKIE_SIGNAL_SOURCES` in `weights.py` so
players whose ONLY ranking comes from Vecenie are filtered out of the
top of the dynasty rankings unless corroborated.

---

## 4. Hashtag Basketball — Dynasty Rankings

**Site:** https://hashtagbasketball.com/
**Status in PR #1:** Not loaded.

One of the largest free fantasy-basketball ranking sites. Publishes
per-format dynasty rankings (9-cat, 8-cat, points-league). Scraping
their public ranking pages is feasible.

Anticipated weight: 1.0 (consensus aggregator). Will exercise the
production-based scoring branch — their per-format dynasty rankings
ARE format-dependent so the same `RankingRecord` cannot be re-used
across formats.

---

## 5. Basketball-Reference

**Site:** https://www.basketball-reference.com/
**Status in PR #1:** Not loaded.

Source-of-truth for historical NBA per-season stats. Will be our
Production loader (planned for the next PR). Once we have several
seasons of realized fantasy production by player, the `backtest.py`
stub becomes real and the SourceTrackRecord-driven weight multipliers
start floating.

---

## 6. ESPN / Yahoo composite rankings

**Status in PR #1:** Not loaded.

Both publish public top-N dynasty rankings periodically; both are
heavily aggregator-style. Lower weight (0.8–1.0) when added — useful
as consensus inputs but not as differentiated evaluators.

---

## 7. Lance Stephenson Big Board

**Status in PR #1:** Not loaded.

Public draft-prospect ranker with cult following on basketball Twitter.
Possible starter-pack entry for rookie-only formats. Weight TBD pending
back-of-envelope hit-rate review.

---

## 8. B/R Top 100 (Bleacher Report)

**Status in PR #1:** Not loaded.

Public NBA Top 100 articles published periodically. Aggregator-style;
similar weight to ESPN composite.

---

## 9. 538 RAPTOR (retired, referenceable only)

**Status in PR #1:** Not loaded.

FiveThirtyEight's RAPTOR metric — frozen in 2023 after the site's
shutdown. Historical RAPTOR data is publicly available as CSV and is
ONLY useful as a backtest cross-check (compare DARKO's historical
opinions vs. RAPTOR's for the same season). Not a live ranking source.

---

## Source taxonomy (for the weighting model)

Same four categories as the football repo:

- **market** — crowdsourced trade values (FantasyCalc-equivalent for NBA
  is sparser; KTC has an NBA spinoff but is largely closed).
- **aggregator** — sites that average multiple experts (Court
  Consensus, FantasyPros NBA).
- **expert** — single-analyst rankings (Sam Vecenie, Lance Stephenson).
- **model** — algorithmic systems (DARKO, RAPTOR-historical, eventual
  Production-derived models).

Rookie-signal sources (data is pre-NBA) get added to
`weights.ROOKIE_SIGNAL_SOURCES`. Players whose ONLY rankings come from
that set are filtered out of the top of the composite to avoid the
"draft picks who never played" pollution pattern that bit the football
repo in early PRs.
