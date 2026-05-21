"""Backtest stub — placeholder until NBA production data is loaded.

In Dynasty-Football-Model this module computes Spearman correlation
between a source's historical rankings and realized NFL fantasy
production. The basketball repo will do the same once we have a
Production loader (planned for a follow-on PR pulling Basketball
Reference season totals).

For PR #1 we keep the public function so callers don't break, but
return None when there's insufficient data. The CLI command surfaces
this cleanly.
"""
from __future__ import annotations
from typing import Optional


def backtest_source(
    source_slug: str,
    cohort_years: list[int],
    window_years: int = 3,
    position: Optional[str] = None,
) -> Optional[dict]:
    """Stub — returns None until Production data is populated.

    Real implementation will:
      * Pull historical Rankings for ``source_slug`` from ``cohort_years``
      * Pair them with realized fantasy production over ``window_years``
      * Compute Spearman correlation, Pearson R², MAE, hit-rate-top-12,
        hit-rate-top-24
      * Write a SourceTrackRecord row keyed on (source, position,
        cohort_year). The scoring layer then picks up the multiplier
        automatically.
    """
    return None
