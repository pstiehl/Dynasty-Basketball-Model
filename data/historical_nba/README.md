# Historical NBA corpus cache

This directory holds cached `LeagueDashPlayerStats` payloads for every
NBA season from 1980-81 through the current season. ~45 JSON files,
one per season, totaling ~4 MB.

## Why this data lives in the repo

The career-arc similarity engine (`src/dynasty_bball/similarity/`)
needs a stable historical corpus to find comparables. Without
checked-in caches the CI build would have to make 45 calls to
stats.nba.com on every refresh — which they rate-limit hard and which
would make CI runs flaky and slow.

The cache files are deterministic outputs of
`scripts/backfill_historical_nba.py`. Live refreshes are gated behind
`DYNASTY_BBALL_HISTORICAL_LIVE=1`; the launcher's `historical_nba`
adapter only verifies the cache by default.

## Refreshing

Run from the repo root **once locally**, then commit:

```bash
python scripts/backfill_historical_nba.py             # adds missing seasons
python scripts/backfill_historical_nba.py --force     # re-fetch everything
python scripts/backfill_historical_nba.py --start 2020 --end 2024-25
```

The backfill takes ~75 seconds with the default 0.6s inter-request
delay (stats.nba.com 429s if pushed harder).

## Format

Each file is the raw `LeagueDashPlayerStats` JSON
(`PerMode=PerGame`, `SeasonType=Regular Season`) with the standard
`resultSets[0].headers` + `rowSet` shape. See
`src/dynasty_bball/sources/historical_nba.py::parse_season_payload`
for the parser.

## Why 1980 as the floor

The 3-point line arrived in 1979-80, and league-wide STL/BLK tracking
started in 1973-74. 1980-81 gives us 45+ years of consistently shaped
stat lines so the profile vector dimensions don't get distorted by
missing dimensions in older data.
