# Basketball-Reference / NBA Stats cache

This directory holds cached `LeagueDashPlayerStats` payloads from the
NBA Stats API (the data backing basketball-reference.com).

## Why we cache

The NBA Stats API rate-limits CI ranges hard. The cache:

1. Makes CI builds deterministic — every daily refresh consumes the
   same input data.
2. Keeps the daily refresh fast (no slow waits on a flaky API).
3. Avoids tripping rate limits if the workflow re-runs.

The cache is small (~200KB per season JSON) and explicitly allowed
through `.gitignore`.

## Files

```
leaguedash_<season>.json   # e.g. leaguedash_2024-25.json, leaguedash_2025-26.json
```

Each file is the raw response from
`nba_api.stats.endpoints.LeagueDashPlayerStats(...).get_dict()` —
canonical NBA Stats API JSON with `resultSets[0].headers` +
`resultSets[0].rowSet`. The adapter parses that shape via
`parse_leaguedash_payload()` in
`src/dynasty_bball/sources/basketball_reference.py`.

## Refreshing

To pull fresh data and update the cache:

```bash
DYNASTY_BBALL_BBREF_LIVE=1 \
  python -c "from dynasty_bball.sync import sync_source; sync_source('basketball_reference')"
```

Or for an ad-hoc dump:

```bash
python -c "
import json
from nba_api.stats.endpoints import leaguedashplayerstats
ep = leaguedashplayerstats.LeagueDashPlayerStats(
    season='2025-26', per_mode_detailed='PerGame',
    season_type_all_star='Regular Season',
)
with open('data/basketball_reference/leaguedash_2025-26.json', 'w') as f:
    json.dump(ep.get_dict(), f)
"
```

Commit the refreshed JSON the same way you'd commit any data file.

## Configuration

Environment variables consumed by the adapter:

- `DYNASTY_BBALL_BBREF_SEASON` — which season to pull (default
  `2025-26`).
- `DYNASTY_BBALL_BBREF_LIVE` — set to `1` to force a live fetch even
  when a cache file exists.
- `DYNASTY_BBALL_BBREF_CACHE_DIR` — override the cache directory
  (default `data/basketball_reference`).

## Sanity check

```bash
python -c "
import json
with open('data/basketball_reference/leaguedash_2025-26.json') as f:
    d = json.load(f)
rs = d['resultSets'][0]
print('rows:', len(rs['rowSet']))
print('cols:', len(rs['headers']))
"
```

Expect ~500-600 rows and ~67 columns.
