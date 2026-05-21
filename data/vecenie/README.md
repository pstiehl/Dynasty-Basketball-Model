# Sam Vecenie's Big Board — local CSV drop

The `vecenie` source (`src/dynasty_bball/sources/vecenie.py`) reads a CSV
at **`data/vecenie/vecenie_big_board.csv`**. The Big Board itself is
paywalled at The Athletic and cannot be scraped — this is a manual drop.

When the file is missing the source yields zero rows and the launcher
continues without error.

## Expected schema

CSV with headers (case-insensitive). The parser accepts these alias
sets — any one column name per row is sufficient:

| Field         | Required | Aliases accepted                                       |
|---------------|----------|--------------------------------------------------------|
| `rank`        | yes      | `rank`, `overall_rank`, `big_board_rank`              |
| `player_name` | yes      | `player_name`, `name`, `player`, `full_name`          |
| `position`    | no       | `position`, `pos`, `primary_position` (PG/SG/SF/PF/C) |
| `tier`        | no       | `tier` (integer 1–5)                                  |
| `notes`       | no       | `notes`, `comment`, `comments`                        |
| `draft_year`  | no       | `draft_year`, `year`, `class` (e.g. 2025, 2026)       |

Multi-position strings (e.g. `PG/SG`) are accepted — only the primary
slot is kept. Empty values are tolerated.

## Example

```csv
rank,player_name,position,tier,draft_year,notes
1,Cooper Flagg,SF,1,2025,Generational two-way wing.
2,Dylan Harper,PG,1,2025,Elite combo guard.
3,Ace Bailey,SF,1,2025,Lottery talent; shooting.
4,VJ Edgecombe,SG,2,2025,Athletic upside swing.
5,Tre Johnson,SG,2,2025,Pure scorer.
```

The launcher prints the row count it ingested on each refresh:

```
Sam Vecenie: 30 rows
```

## How the records flow through the model

- The CSV is parsed and each row produces a `RankingRecord` in **both**
  `points_dhk` and `points_default` formats (Vecenie's signal is
  format-agnostic).
- `market_value` is derived linearly from rank: rank `1` → `100.0`,
  the highest rank in the file → `0.0`. That lands the signal on the
  same 0–100 scale Court Consensus uses, so the scoring layer's
  value-based normalization picks it up cleanly.
- Vecenie is on `weights.ROOKIE_SIGNAL_SOURCES`: players who appear
  **only** on the Big Board (and nowhere else in the model) are
  filtered out of the top of the dynasty composite to avoid
  draft-prospect squatting. Once a player makes a regular-season
  appearance, DARKO and Court Consensus pick them up automatically and
  the filter releases.

## Weighting

`default_weight = 1.3`. Sam Vecenie has a documented strong hit rate
on NBA draft prospect translation. Track-record multiplier will
adjust this once we have a Production loader and can backtest.

## Provenance / attribution

If/when you publish, attribute the underlying rankings to Sam Vecenie
(The Athletic). The model only consumes the rank order — none of the
analysis text is redistributed.
