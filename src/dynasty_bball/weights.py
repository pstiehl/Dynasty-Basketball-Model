"""Weighting policy for composite scoring.

Mirror of Dynasty-Football-Model's v0.10 redesign (deterministic per-source
weighting). Weights are driven only by backtested correlation with realized
NBA fantasy production; they do **not** vary per-player.

    effective_weight = default_weight * track_record_multiplier

The one allowed source of per-player variation is the position-specific
track record: if a backtest produced a SourceTrackRecord row for
``(source, position="C")``, that row's correlation is used for centers;
non-center players fall back to the overall (``position=None``) row.

What's intentionally absent (carried from football):

  * Hand-coded per-(source, position) overrides.
  * Years-pro decay for rookie-signal sources.

Both of those cause the same source to display different weight values for
different players in the breakdown JSON, which reads as inconsistent.

What remains:

  * ``ROOKIE_SIGNAL_SOURCES`` — set of source slugs whose data is
    fundamentally pre-NBA. Used by ``scoring.py`` to filter out
    retired/no-longer-rostered players whose only rankings come from
    these sources.
  * ``corr_to_multiplier()`` — the |Spearman ρ| → multiplier ladder.
  * ``select_track_record_multiplier()`` — picks the position-specific
    track record row if present, falls back to overall.

Reference: ``docs/CHANGELOG-model.md`` § v0.1.0.
"""
from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# Rookie-signal sources — used by scoring.py to filter out retired players
# whose ONLY rankings come from pre-NBA data. ("No consensus" pattern.)
#
# NBA-relevant rookie-signal sources we'll add in later PRs:
#   * nba_draft_capital  — first/second-round pick
#   * sam_vecenie        — NBA Athletic prospect ranker
#   * bleacher_top_100   — public draft prospect lists
#   * lance_stephenson_big_board (placeholder)
#
# In PR #1 the set is empty (DARKO covers ROOKIES via the active dataset
# the moment they play their first game), but the wiring is here so future
# adapters fall in cleanly.
# ---------------------------------------------------------------------------

ROOKIE_SIGNAL_SOURCES: set[str] = set()


# ---------------------------------------------------------------------------
# Position-specific track-record selector.
# ---------------------------------------------------------------------------

def select_track_record_multiplier(
    track_records_by_pos: dict[Optional[str], float],
    position: Optional[str],
) -> float:
    """Pick the position-specific multiplier when available, else fallback.

    ``track_records_by_pos`` is the per-source mapping
    ``{position_or_None: multiplier}``. Lookup order:

      1. exact position match
      2. position-None (overall) entry
      3. neutral (1.0)
    """
    if position:
        m = track_records_by_pos.get(position.upper())
        if m is not None:
            return m
    overall = track_records_by_pos.get(None)
    if overall is not None:
        return overall
    return 1.0


def corr_to_multiplier(corr: Optional[float]) -> float:
    """Convert |spearman_corr| into a multiplier.

    Same tuning as Dynasty-Football-Model:

      |ρ| ≥ 0.35  → 1.6
      |ρ| ≥ 0.25  → 1.3
      |ρ| ≥ 0.15  → 1.0
      |ρ| <  0.15 → 0.5
      None        → 1.0 (neutral)
    """
    if corr is None:
        return 1.0
    a = abs(corr)
    if a >= 0.35:
        return 1.6
    if a >= 0.25:
        return 1.3
    if a >= 0.15:
        return 1.0
    return 0.5
