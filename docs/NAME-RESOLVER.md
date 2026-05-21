# Name resolver — technical writeup

> Source of truth for player identity across sources. See PR #6 / v0.5.0.

The dynasty composite blends ranking signals from at least five
independent sources (DARKO, Court Consensus, Sam Vecenie, Basketball-
Reference, Career-Arc similarity). Each source spells player names
differently — sometimes wildly:

| Source                 | Long-form                | Short-form                 |
| ---------------------- | ------------------------ | -------------------------- |
| DARKO                  | `Nicolas Claxton`        | —                          |
| Sleeper                | —                        | `Nic Claxton`              |
| Basketball-Reference   | —                        | `Nic Claxton`              |
| Court Consensus        | `Carlton Carrington`     | (Sleeper has `Bub`)        |
| Sleeper / BBRef        | —                        | `Bub Carrington`           |
| DARKO                  | `Nah'Shon Hyland`        | (BBRef has `Bones Hyland`) |
| Multiple               | `Jusuf Nurkić`           | `Jusuf Nurkic`             |

Naive exact-string matching produces orphan rows in the top-300 — one
row per spelling — and orphan rows lose their Basketball-Reference /
career-arc similarity signal because the join never fires. PR #6 fixes
that with a 4-tier resolver:

```mermaid
flowchart TD
    A[Incoming RankingRecord] --> B{sleeper_id /<br/>nba_id / bbref_id?}
    B -->|yes| Z[Use existing Player]
    B -->|no| C[canonical_key normalization]
    C --> D{Tier 1:<br/>exact canonical hash}
    D -->|hit| Z
    D -->|miss| E{Tier 2:<br/>last + first[0] +<br/>pos bucket + team}
    E -->|hit| Z
    E -->|miss| F{Alias map<br/>data/name_aliases.json}
    F -->|hit| Z
    F -->|miss| G{Tier 3:<br/>fuzzy first name<br/>same team REQUIRED}
    G -->|hit| Z
    G -->|miss| H[Create new Player +<br/>add to unmatched list]
```

## Tier 1 — Canonical key

`canonical_key()` produces a string that's stable across diacritic,
suffix, and punctuation variation:

```python
canonical_key("Nikola Jokić")        # → "nikola jokic"
canonical_key("LeBron James Jr.")    # → "lebron james"
canonical_key("T.J. McConnell")      # → "tj mcconnell"
canonical_key("Kelly Oubre, Jr.")    # → "kelly oubre"
```

It's idempotent, so a record that's already canonicalized stays put.
This catches ~95% of cross-source dupes — every diacritic variant,
every suffix variant, every initials-with-periods variant.

## Tier 2 — last + first-initial + position-bucket + team

When Tier 1 misses, we widen: same last name, same first-name
initial, compatible position bucket (G/F/C), **and** same team
abbreviation. Examples:

- `Nicolas Claxton` (no pos, BKN) ↔ `Nic Claxton` (C, BKN) — last
  `claxton`, first[0] `n`, team `BKN`. ✓
- `Alexandre Sarr` (no pos, WAS) ↔ `Alex Sarr` (C, WAS) — last `sarr`,
  first[0] `a`, team `WAS`. ✓
- `Bub Carrington` (PG, WAS) ↔ `Carlton Carrington` (no pos, WAS) —
  last matches but first[0] is `b` vs `c`. ✗ → falls through to alias.

Position-bucket compatibility tolerates source-to-source disagreement
on hybrid players: `PG` and `SG` both bucket to `G`; `PF` and `C`
are adjacent and treated as compatible.

## Alias map — `data/name_aliases.json`

Hand-curated edge cases that no algorithm can safely guess. Format:

```json
{
  "entries": [
    {
      "canonical": "Bub Carrington",
      "aliases": ["Carlton Carrington"]
    },
    {
      "canonical": "Bones Hyland",
      "aliases": ["Nah'Shon Hyland", "Nahshon Hyland"]
    }
  ]
}
```

The resolver consults this map *before* Tier 3 fuzzy, so well-known
mappings never have to risk the looser fuzzy logic.

### Maintenance

When a new player shows up on the `sources.html#unmatched` list, the
fix is usually one alias entry. Walk:

1. Look up the player on Basketball-Reference and Sleeper.
2. Pick the canonical form (BBRef name wins).
3. Add an entry with the canonical and every spelling you find.
4. Re-run the launcher — the player should now match via `alias`.

## Tier 3 — Conservative fuzzy

The strictest fuzzy path. Required conditions:

1. Last names equal post-canonical-key.
2. Position buckets compatible.
3. **Same team abbreviation** — never crosses team boundaries.
4. First names satisfy one of: shared 2-char prefix, known
   diminutive (`alex` ↔ `alexandre`, `nic` ↔ `nicolas`), or token-set
   similarity ≥ 0.80.

If Tier 3 misses, the record creates a new Player row OR (if it lacks
a Basketball-Reference signal entirely) is excluded from rankings and
recorded in `data/diagnostics/unmatched_players.json`.

## False-merge prevention

> Better to leave a player unmatched than to incorrectly merge two
> different players. — PR #6 directive.

Guards baked in:

- Tier 3 **always** requires same team. Two players named "Anthony
  Davis" on different teams never collapse via fuzzy.
- Suffix tiebreak on Tier 1: if both sides carry an explicit suffix
  and they differ (Jr. vs Sr.), reject the canonical match.
- Tier 2 requires first-name *initial* match, not just last+team. Two
  Carringtons on `WAS` with first initials `b` vs `c` aren't merged.

Tested by `tests/test_name_resolver.py`:

- `test_tier3_requires_same_team_for_fuzzy`
- `test_tier3_no_false_merge_different_player`
- `test_no_false_merge_with_jr_suffix`
- `test_tier3_garbage_first_name_with_known_team`

## Dedup pass

The resolver only catches dupes that *will be created in this sync*.
For dupes already in the DB (e.g. the PR-#5-era orphan rows), the
post-sync `dedup_players_by_canonical()` pass runs in three stages:

1. Group every Player row by `canonical_key(full_name)` and merge
   members with the same key. The keeper is the row with the most
   identity (3-letter team + position + external IDs); Rankings and
   Evaluations get re-pointed onto it.
2. For remaining "orphan-shaped" rows (no `sleeper_id` / `bbref_id` /
   `nba_id`), run the full NameResolver against the pool of
   identified players. Tier 2 / alias / Tier 3 matches trigger the
   same merge.
3. Normalize team strings ("Washington Wizards" → "WAS") and refresh
   `normalized_name` on every remaining row.

The pass is idempotent — running it repeatedly is a no-op once the
DB is clean.

## Observability

Every sync writes:

- `data/diagnostics/resolver_stats.json` — tier counts + timestamp.
- `data/diagnostics/unmatched_players.json` — Sleeper-only rows that
  fell through every tier. Should be empty or near-empty.

Both sidecars are read by `report.py` and surfaced on the rendered
site:

- **rankings.html** — header banner: `300 players · 6 name variants
  merged · 0 unmatched`.
- **sources.html#unmatched** — full list of excluded players.

If you ever see `>10` players on the unmatched list, the resolver
isn't aggressive enough for the current source mix. Loosen Tier 3
(lower the similarity threshold, broaden the diminutive map) or add
alias entries.
