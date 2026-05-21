# Dynasty Basketball Model

> Open-source dynasty fantasy basketball composite ranking model. Blends
> multiple analytics sources (DARKO, Court Consensus, Hashtag Basketball,
> Basketball-Reference, public expert lists) into a single deterministic
> dynasty score per player. Mirrors the architecture of
> [Dynasty-Football-Model](https://github.com/pstiehl/Dynasty-Football-Model).

## What you get

* A unified score per player in **Dynasty Hoop Kings** scoring (Phil's
  Sleeper NBA league) AND a generic Sleeper NBA points scoring as a
  comparison format.
* The full Top-300 dynasty rankings as a static site, refreshed daily by
  GitHub Actions and published to GitHub Pages.
* "Rate My League" — paste a Sleeper NBA league id and see every team's
  total roster value, top assets, and weaknesses against the model.
* Manager skill ranking from draft and trade history (z-score blend of
  draft delta and trade delta against current model values).
* Per-source backtest hooks so weights can be adjusted by realized
  Spearman correlation with NBA fantasy production, not by hand.

## Architecture

```
                    ┌─────────────────────────────────┐
   data sources ──► │ adapters (sources/<slug>.py)    │
   (DARKO, …)       │ — each yields RankingRecord[]   │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │ sync.py              │
                          │ — resolve to Player  │
                          │ — write Ranking rows │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │ scoring.py           │
                          │ — deterministic v0.10│
                          │   weighting model    │
                          │ — composite_scores   │
                          └──────────┬───────────┘
                                     │
            ┌────────────────────────┼─────────────────────────┐
            ▼                        ▼                         ▼
   report.py                 league.py                 manager.py
   (HTML site)               (Sleeper team eval)       (manager skill)
```

The scoring layer is **deterministic per source**:

```
effective_weight = default_weight × track_record_multiplier
```

No hand-coded position modifiers, no years-pro decay. The only allowed
per-player variation comes from a position-specific backtested
correlation — and that's data-driven. See
[`docs/CHANGELOG-model.md`](docs/CHANGELOG-model.md) for the rationale,
copied directly from the football repo's v0.10 redesign.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

python -m dynasty_bball.cli init-db
python -m dynasty_bball.cli sync-players       # ~700+ NBA players from Sleeper
python -m dynasty_bball.cli sync darko         # ~500+ rankings from DARKO
python -m dynasty_bball.cli score              # composite scores in both formats
python -m dynasty_bball.cli top --n 25         # show top-25
```

Or run the full end-to-end pipeline (same thing GitHub Actions runs):

```bash
python -m dynasty_bball.launcher_headless
open dynasty_site/index.html
```

## Sources (PR #1)

| Source            | Category | Weight | Update | Status |
|-------------------|----------|--------|--------|--------|
| DARKO             | model    | 1.5    | daily  | live   |
| Sleeper (NBA)     | reference| —      | weekly | live (player map only) |

More sources land in subsequent PRs — see `docs/RESEARCH-sources.md`.

## Phil's league

[Dynasty Hoop Kings](https://sleeper.com/leagues/1349496244468199424)
is pre-fetched on every run and baked into the site at
`dynasty_site/leagues/sleeper_nba-1349496244468199424.json`. Scoring is
the league's actual settings:

```
pts=0.5  reb=1.0  ast=1.0  stl=2.0  blk=2.0  tpm=0.5
dd=1.0   td=2.0   to=-1.0  tf=-2.0  ff=-2.0
bonus_pt_40p=2.0  bonus_pt_50p=2.0
```

Roster: 10 UTIL + 8 BN + 1 IR + 2 taxi, 12 teams, no positional starters.

## License

MIT.
