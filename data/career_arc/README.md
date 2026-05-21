# Career-Arc sidecar

Holds `comparables.json` — the per-player top-5 comparables list +
per-format dynasty value summary written by the `career_arc` adapter
during sync.

Re-generated automatically on every launcher run; safe to delete (it
just gets rebuilt). Committed to the repo so the static site builds
correctly off `dynasty.db` even on a fresh checkout.

## Format

```json
{
  "generated_at": "ISO 8601 UTC",
  "current_season": "2025-26",
  "by_nba_id": {
    "<nba_id>": {
      "top_comparables": [
        {
          "name": "Kevin Durant",
          "season": "2007-08",
          "age": 19.0,
          "similarity": 0.916,
          "remaining_seasons": 16,
          "remaining_games": 1100,
          "remaining_fp_dhk": 27.79,
          "remaining_fp_default": 41.20,
          "bucket_match": true,
          "censored": true
        },
        ...
      ],
      "n_comparables": 20,
      "by_format": {
        "points_dhk":    {"dynasty_value": 100.0, "projected_remaining_years": 10.0,
                          "projected_total_fantasy_points": 11987.0,
                          "per_year_survival_prob": [...]},
        "points_default": {...}
      }
    },
    ...
  }
}
```

`censored: true` means the comp's career is still active at the
corpus end (currently 2024-25). Their "remaining seasons" is a lower
bound, not a final answer.
