"""Career-arc projection — aggregate comps into the dynasty value.

Inputs
------
For each current player we already have:

  * The player's profile (vectorized current-season production).
  * A list of top-K ``Comparable`` records, each carrying:
      - similarity score (0..1, cosine)
      - remaining_seasons (how long the comp played after that age)
      - remaining_games (total GP after that age)
      - remaining_fantasy_ppg_{dhk,default}
      - ages_after (which ages they still played)
      - censored (was their corpus career truncated at the data
        window's end — true for current pros; we treat their
        observed remaining as a lower bound, not a final answer)

Outputs
-------
A ``CareerProjection`` per player + format:

  * ``projected_remaining_years``        — weighted median across comps
  * ``projected_total_fantasy_points``   — Σ (comp ppg × comp games),
                                            similarity-weighted average
  * ``dynasty_value_raw``                — present-value weighted sum
                                            with time discount per year
  * ``dynasty_value``                    — dynasty_value_raw rescaled
                                            0..100 across the input cohort
  * ``per_year_survival_prob[1..15]``    — fraction of comps still playing
                                            at age+offset, weighted by sim
  * ``top_comparables``                  — top 5 for the UI

Method
------
Weights for aggregation are similarities normalized to sum to 1 across
the K comps. The time-discount factor (default 5% per year) is applied
to the per-comp expected remaining points:

    pv_points(comp) = ppg × games × Σ_{y=1..remaining_years} discount^y / remaining_years

That smears the comp's remaining production over their remaining years
and discounts each year. Then we similarity-weight across comps.

This is intentionally simple. A more sophisticated model would
project a YEAR-BY-YEAR arc (age curves, mean reversion). For PR #4
we want the headline thesis — "Cooper Flagg looks like a stack of
20-year careers; Harden looks like a stack of 3-year careers" — and
that requires nothing more than these aggregations.

Why a median for years, but a mean for points?
The remaining-years distribution is bimodal (some careers end fast,
some last 15+). The median is robust against the tail outliers. Points
per year, conversely, are normally distributed around the player's
true skill level — the mean (similarity-weighted) is the right
estimator. The weighted-median helper is in this module.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from .comparables import Comparable


# 5% per-year time discount. The same magnitude Dynasty-Football-Model
# uses for its draft-pick valuations.
DEFAULT_DISCOUNT_RATE = 0.05

# How far out to compute per-year survival probability.
SURVIVAL_HORIZON_YEARS = 15

# Floor for projected remaining years — even if every comp retired by
# age+0, we still credit the current player with one more season of
# their existing production (they're under contract, after all).
MIN_PROJECTED_REMAINING_YEARS = 1.0


@dataclass
class CareerProjection:
    """The dynasty-value output for a single (player, league_format)."""
    player_nba_id: str
    player_name: str
    player_age: float
    league_format: str
    projected_remaining_years: float
    projected_total_fantasy_points: float
    dynasty_value_raw: float
    dynasty_value: float                       # 0..100, rescaled across cohort
    per_year_survival_prob: list[float]        # length SURVIVAL_HORIZON_YEARS
    top_comparables: list[Comparable] = field(default_factory=list)
    n_comps: int = 0
    # The single most-similar comp — handy for the headline UI blurb.
    top_comp_name: Optional[str] = None
    top_comp_similarity: Optional[float] = None


# ---------------------------------------------------------------------------
# Weighted-statistics helpers
# ---------------------------------------------------------------------------

def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted median. Treats values as a sorted CDF."""
    if not values:
        return 0.0
    pairs = sorted(zip(values, weights), key=lambda t: t[0])
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return float(pairs[len(pairs) // 2][0])
    cum = 0.0
    half = total_w / 2.0
    for v, w in pairs:
        cum += w
        if cum >= half:
            return float(v)
    return float(pairs[-1][0])


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


# ---------------------------------------------------------------------------
# Per-player projection
# ---------------------------------------------------------------------------

def project_career(
    *,
    nba_id: str,
    name: str,
    age: float,
    comparables: list[Comparable],
    league_format: str = "points_dhk",
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> CareerProjection:
    """Build a CareerProjection from comparables. Pre-rescaling.

    The ``dynasty_value`` field is set to ``dynasty_value_raw`` here
    and later rescaled 0..100 across the cohort by
    ``project_all_current_players``.
    """
    if not comparables:
        return CareerProjection(
            player_nba_id=nba_id,
            player_name=name,
            player_age=age,
            league_format=league_format,
            projected_remaining_years=MIN_PROJECTED_REMAINING_YEARS,
            projected_total_fantasy_points=0.0,
            dynasty_value_raw=0.0,
            dynasty_value=0.0,
            per_year_survival_prob=[0.0] * SURVIVAL_HORIZON_YEARS,
            top_comparables=[],
            n_comps=0,
        )

    # Similarity weights, normalized to sum to 1. Negative sims (shouldn't
    # happen post-clip) treated as zero.
    sims = np.array([max(0.0, c.similarity) for c in comparables], dtype=np.float64)
    if sims.sum() <= 0:
        sims = np.ones_like(sims) / len(sims)
    else:
        sims = sims / sims.sum()

    ppg_key = "remaining_fantasy_ppg_dhk" if league_format == "points_dhk" else "remaining_fantasy_ppg_default"

    remaining_years = [float(c.remaining_seasons) for c in comparables]
    proj_years = max(MIN_PROJECTED_REMAINING_YEARS, _weighted_median(remaining_years, sims))

    # Per-comp present-value remaining fantasy points.
    pv_points: list[float] = []
    for c in comparables:
        ppg = getattr(c, ppg_key)
        if ppg is None or c.remaining_seasons <= 0:
            pv_points.append(0.0)
            continue
        # Assume the comp's remaining games are spread evenly across
        # their remaining seasons. Discount each season at (1+r)^-y.
        games_per_season = c.remaining_games / c.remaining_seasons
        pv = 0.0
        for y in range(1, c.remaining_seasons + 1):
            pv += ppg * games_per_season / ((1.0 + discount_rate) ** y)
        pv_points.append(pv)

    proj_total_pts = _weighted_mean(pv_points, sims)

    # Per-year survival probability — what fraction of weighted comps
    # still played at age + offset?
    survival: list[float] = []
    for offset in range(1, SURVIVAL_HORIZON_YEARS + 1):
        target_age = age + offset
        weighted_still = 0.0
        for c, w in zip(comparables, sims):
            if any(a >= target_age - 0.5 for a in c.ages_after):
                weighted_still += w
        survival.append(weighted_still)

    top_5 = comparables[:5]
    top1 = comparables[0] if comparables else None

    return CareerProjection(
        player_nba_id=nba_id,
        player_name=name,
        player_age=age,
        league_format=league_format,
        projected_remaining_years=proj_years,
        projected_total_fantasy_points=proj_total_pts,
        dynasty_value_raw=proj_total_pts,
        dynasty_value=0.0,  # filled in by cohort-rescale below
        per_year_survival_prob=survival,
        top_comparables=top_5,
        n_comps=len(comparables),
        top_comp_name=top1.comp_name if top1 else None,
        top_comp_similarity=top1.similarity if top1 else None,
    )


# ---------------------------------------------------------------------------
# Cohort-wide rescaling
# ---------------------------------------------------------------------------

def rescale_to_0_100(projections: list[CareerProjection]) -> list[CareerProjection]:
    """Rescale ``dynasty_value_raw`` → ``dynasty_value`` 0..100 across cohort.

    Top player = 100.0. Players with raw=0 stay at 0. Mutates in place
    and also returns the list.
    """
    if not projections:
        return projections
    top = max(p.dynasty_value_raw for p in projections)
    if top <= 0:
        for p in projections:
            p.dynasty_value = 0.0
        return projections
    for p in projections:
        p.dynasty_value = round(100.0 * max(0.0, p.dynasty_value_raw) / top, 3)
    return projections


# ---------------------------------------------------------------------------
# Convenience: project a full cohort of current players in one call.
# ---------------------------------------------------------------------------

def project_all_current_players(
    targets: list[tuple],
    corpus,
    career_index,
    league_format: str = "points_dhk",
    k: int = 20,
    age_window: float = 1.0,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> list[CareerProjection]:
    """Run KNN + projection across many current players.

    ``targets`` is a list of tuples ``(nba_id, name, age, profile)``
    where ``profile`` is a Profile (un-normalized; we normalize against
    the corpus here).

    Returns a list of CareerProjections, rescaled to 0..100 across the
    full cohort.
    """
    from .comparables import find_comparables

    out: list[CareerProjection] = []
    for nba_id, name, age, profile in targets:
        comps = find_comparables(
            target_profile=profile,
            corpus=corpus,
            career_index=career_index,
            k=k,
            age_window=age_window,
            exclude_nba_id=nba_id,
        )
        proj = project_career(
            nba_id=nba_id,
            name=name,
            age=age,
            comparables=comps,
            league_format=league_format,
            discount_rate=discount_rate,
        )
        out.append(proj)
    rescale_to_0_100(out)
    return out
