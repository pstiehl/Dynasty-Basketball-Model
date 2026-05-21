"""Composite scoring — blends source rankings into a single dynasty score.

This is a near-direct port of Dynasty-Football-Model's scoring.py, with
basketball-specific scoring weights for the production-based branch.

Approach (same v0.10 deterministic model):
  1. For each source, pull its most-recent ranking per player at a given
     league_format.
  2. Convert each source's rank/value to a normalized 0..100 score.
  3. Weight each source by:
         effective_weight = default_weight × track_record_multiplier
  4. Composite = weighted average of per-source scores.
  5. Compute a "consensus rank" using only market/aggregator sources.
  6. Compute rank_divergence = consensus_rank - model_rank.
  7. Write CompositeScore rows.

Phil's Dynasty Hoop Kings scoring (default ``points_dhk``):
    pts=0.5  reb=1.0  ast=1.0  stl=2.0  blk=2.0  tpm=0.5
    dd=1.0   td=2.0   to=-1.0  tf=-2.0  ff=-2.0
    bonus_pt_40p=2.0  bonus_pt_50p=2.0

Generic Sleeper NBA points scoring (``points_default``):
    pts=1.0  reb=1.2  ast=1.5  stl=3.0  blk=3.0  tpm=0.5  to=-1.0

These per-stat weights are only used when scoring projected production
(currently a placeholder branch — Production rows aren't populated in
PR #1). DPM-style sources like DARKO bake their stat opinions into the
market_value scalar already, so the same scalar is used unchanged in
both league formats today. Subsequent PRs that add raw projections
(Hashtag Basketball, Basketball Monster) will start exercising the
per-format weighting branch.
"""
from __future__ import annotations
import json
from datetime import datetime
from collections import defaultdict
from typing import Optional
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .db.session import get_session
from .db.models import Source, Player, Ranking, CompositeScore, SourceTrackRecord
from .weights import (
    select_track_record_multiplier,
    corr_to_multiplier,
    ROOKIE_SIGNAL_SOURCES,
)


# How many players to consider "in range" for normalization. Above this rank,
# score floors to 0.
DEFAULT_NORMALIZATION_DEPTH = 300

# Source categories that represent "consensus" / "the market".
CONSENSUS_CATEGORIES = {"market", "aggregator"}


# League-format scoring tables. Each maps a stat label to its point weight.
# Used by the (currently placeholder) per-format production-based scoring
# branch. Documented here for transparency — these are exactly Phil's
# league settings.
LEAGUE_SCORING: dict[str, dict[str, float]] = {
    "points_dhk": {
        "pts": 0.5, "reb": 1.0, "ast": 1.0, "stl": 2.0, "blk": 2.0,
        "tpm": 0.5, "dd": 1.0, "td": 2.0,
        "to": -1.0, "tf": -2.0, "ff": -2.0,
        "bonus_pt_40p": 2.0, "bonus_pt_50p": 2.0,
    },
    "points_default": {
        "pts": 1.0, "reb": 1.2, "ast": 1.5, "stl": 3.0, "blk": 3.0,
        "tpm": 0.5, "to": -1.0,
    },
}


def per_game_fantasy_points(
    stats: dict, scoring: dict[str, float]
) -> float:
    """Apply a scoring dict to a per-game stat line. Missing values = 0."""
    return sum(
        (stats.get(k) or 0.0) * w for k, w in scoring.items() if not k.startswith("bonus_")
    )


# ---------------------------------------------------------------------------
# Core scoring pipeline
# ---------------------------------------------------------------------------

def _latest_rankings_by_source(
    session: Session, league_format: str
) -> dict[int, dict[int, Ranking]]:
    """Returns {source_id: {player_id: latest_ranking}}.

    "Latest" = max captured_at per (source, player, league_format).
    """
    subq = (
        select(
            Ranking.source_id,
            Ranking.player_id,
            func.max(Ranking.captured_at).label("max_cap"),
        )
        .where(Ranking.league_format == league_format)
        .group_by(Ranking.source_id, Ranking.player_id)
        .subquery()
    )

    rows = session.execute(
        select(Ranking)
        .join(
            subq,
            (Ranking.source_id == subq.c.source_id)
            & (Ranking.player_id == subq.c.player_id)
            & (Ranking.captured_at == subq.c.max_cap),
        )
        .where(Ranking.league_format == league_format)
    ).scalars().all()

    out: dict[int, dict[int, Ranking]] = defaultdict(dict)
    for r in rows:
        out[r.source_id][r.player_id] = r
    return out


def _track_record_multipliers(
    session: Session,
) -> dict[int, dict[Optional[str], float]]:
    """Convert backtest results into per-(source, position) weight multipliers."""
    rows = session.execute(
        select(SourceTrackRecord)
        .where(SourceTrackRecord.cohort_year.is_(None))
        .order_by(SourceTrackRecord.calculated_at.desc())
    ).scalars().all()

    out: dict[int, dict[Optional[str], float]] = defaultdict(dict)
    for r in rows:
        pos = (r.position.upper() if r.position else None)
        if pos in out[r.source_id]:
            continue
        out[r.source_id][pos] = corr_to_multiplier(r.spearman_corr)
    return out


def _rank_to_score(rank: int | None, depth: int) -> float | None:
    if rank is None:
        return None
    if rank <= 0:
        return None
    if rank > depth:
        return 0.0
    return 100.0 * (1.0 - (rank - 1) / depth)


def _value_to_score(value: float | None, max_value: float) -> float | None:
    if value is None or max_value <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * value / max_value))


def compute_composite_scores(
    league_format: str = "points_dhk",
    depth: int = DEFAULT_NORMALIZATION_DEPTH,
    model_version: str = "0.1.0",
    score_year: int | None = None,
) -> int:
    """Run the scoring pipeline. Returns number of CompositeScore rows written."""
    with get_session() as session:
        per_source = _latest_rankings_by_source(session, league_format)
        if not per_source:
            return 0

        sources = {
            s.id: s for s in session.execute(select(Source)).scalars().all()
        }
        multipliers_by_pos = _track_record_multipliers(session)

        consensus_source_ids = {
            sid for sid, s in sources.items() if s.category in CONSENSUS_CATEGORIES
        }

        source_max_value: dict[int, float] = {}
        for sid, plr_rankings in per_source.items():
            vals = [r.market_value for r in plr_rankings.values() if r.market_value is not None]
            if vals:
                source_max_value[sid] = max(vals)

        all_pids = set()
        for plr_rankings in per_source.values():
            all_pids.update(plr_rankings.keys())
        players_by_id: dict[int, Player] = {
            p.id: p
            for p in session.execute(
                select(Player).where(Player.id.in_(all_pids))
            ).scalars().all()
        }

        effective_score_year = score_year or datetime.utcnow().year  # noqa: F841 — reserved for prod scoring

        def _weight_for(sid: int, pos: Optional[str]) -> float:
            src = sources[sid]
            tr_mult = select_track_record_multiplier(
                multipliers_by_pos.get(sid, {}), pos
            )
            return src.default_weight * tr_mult

        contribs: dict[int, list[tuple[str, str, float, float, int | None]]] = defaultdict(list)
        consensus_ranks: dict[int, list[int]] = defaultdict(list)

        for sid, plr_rankings in per_source.items():
            src = sources.get(sid)
            if src is None:
                continue
            for pid, ranking in plr_rankings.items():
                player = players_by_id.get(pid)
                pos = player.position if player else None
                weight = _weight_for(sid, pos)
                score = None
                if ranking.market_value is not None and sid in source_max_value:
                    score = _value_to_score(ranking.market_value, source_max_value[sid])
                if score is None:
                    score = _rank_to_score(ranking.overall_rank, depth)
                if score is None:
                    continue
                contribs[pid].append((src.slug, src.category, score, weight, ranking.overall_rank))
                if sid in consensus_source_ids and ranking.overall_rank is not None:
                    consensus_ranks[pid].append(ranking.overall_rank)

        avg_consensus_rank = {
            pid: int(round(sum(ranks) / len(ranks)))
            for pid, ranks in consensus_ranks.items() if ranks
        }

        generated_at = datetime.utcnow()
        results = []
        for pid, items in contribs.items():
            # Corroboration filter (same as football): skip players whose
            # ONLY rankings come from pre-NBA / rookie-signal sources.
            slugs_present = {slug for slug, _, _, _, _ in items}
            if slugs_present and ROOKIE_SIGNAL_SOURCES and slugs_present.issubset(ROOKIE_SIGNAL_SOURCES):
                continue

            total_w = sum(w for _, _, _, w, _ in items)
            if total_w <= 0:
                continue
            score = sum(s * w for _, _, s, w, _ in items) / total_w

            breakdown = {
                slug: {
                    "score": round(s, 2),
                    "weight": round(w, 3),
                    "raw_rank": rank,
                    "category": cat,
                }
                for slug, cat, s, w, rank in items
            }
            results.append((pid, score, breakdown))

        results.sort(key=lambda x: x[1], reverse=True)
        position_counters: dict[str, int] = defaultdict(int)

        count = 0
        for overall_rank, (pid, score, breakdown) in enumerate(results, start=1):
            pos = players_by_id.get(pid).position if pid in players_by_id else None
            pos_rank = None
            if pos:
                position_counters[pos] += 1
                pos_rank = position_counters[pos]
            consensus_r = avg_consensus_rank.get(pid)
            divergence = (consensus_r - overall_rank) if consensus_r is not None else None

            session.add(CompositeScore(
                player_id=pid,
                league_format=league_format,
                score=score,
                overall_rank=overall_rank,
                position_rank=pos_rank or 0,
                tier=_tier_from_rank(overall_rank),
                consensus_rank=consensus_r,
                rank_divergence=divergence,
                breakdown_json=json.dumps(breakdown),
                model_version=model_version,
                generated_at=generated_at,
            ))
            count += 1
        return count


def _tier_from_rank(rank: int) -> int:
    if rank <= 6:    return 1
    if rank <= 12:   return 2
    if rank <= 24:   return 3
    if rank <= 36:   return 4
    if rank <= 60:   return 5
    if rank <= 100:  return 6
    if rank <= 150:  return 7
    if rank <= 200:  return 8
    return 9
