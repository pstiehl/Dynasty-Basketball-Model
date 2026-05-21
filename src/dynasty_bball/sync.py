"""Sync logic: run a source adapter and persist its records.

Player resolution order: sleeper_id → nba_id → bbref_id → (full_name + position).
If no match, a minimal Player row is auto-created so we never drop data.

The DARKO adapter also writes Evaluation rows for the underlying DPM
components and player-level longevity fields (est_retirement_age,
years_remaining) onto the Player row.
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import select

from .db.session import get_session
from .db.models import Source, Player, Ranking, Evaluation
from .sources import REGISTRY
from .sources.base import BaseSource, RankingRecord
from .sources.sleeper_players import SleeperPlayers
from .names import normalize as _normalize_name


# Numeric fields on RankingRecord that should be persisted as Evaluation rows
# alongside the Ranking. Each is (metric_name, attr_name).
DARKO_METRIC_FIELDS = [
    ("dpm", "dpm"),
    ("dpm_improvement", "dpm_improvement"),
    ("o_dpm", "o_dpm"),
    ("d_dpm", "d_dpm"),
    ("years_remaining", "years_remaining"),
    ("est_retirement_age", "est_retirement_age"),
]


def _ensure_source_row(session, adapter: BaseSource) -> Source:
    row = session.execute(select(Source).where(Source.slug == adapter.slug)).scalar_one_or_none()
    if row is None:
        row = Source(
            slug=adapter.slug,
            name=adapter.name,
            category=adapter.category,
            url=adapter.homepage,
            update_frequency=adapter.update_frequency,
            tos_compliant=adapter.tos_compliant,
            default_weight=adapter.default_weight,
            notes=adapter.notes,
        )
        session.add(row)
        session.flush()
    else:
        # Refresh fields that can legitimately change between releases.
        # Weight in particular MUST refresh — PR #4 drops DARKO from
        # 1.5 → 0.8 and that change has to land in the DB on every
        # sync, not just on fresh installs.
        row.name = adapter.name
        row.category = adapter.category
        row.url = adapter.homepage
        row.update_frequency = adapter.update_frequency
        row.tos_compliant = adapter.tos_compliant
        row.default_weight = adapter.default_weight
        row.notes = adapter.notes
    return row


def _resolve_player(session, rec: RankingRecord) -> Player | None:
    if rec.sleeper_id:
        p = session.execute(select(Player).where(Player.sleeper_id == rec.sleeper_id)).scalar_one_or_none()
        if p:
            return _enrich_player(p, rec)
    if rec.nba_id:
        p = session.execute(select(Player).where(Player.nba_id == rec.nba_id)).scalar_one_or_none()
        if p:
            return _enrich_player(p, rec)
    if rec.bbref_id:
        p = session.execute(select(Player).where(Player.bbref_id == rec.bbref_id)).scalar_one_or_none()
        if p:
            return _enrich_player(p, rec)
    if rec.full_name:
        # Exact match first — cheapest.
        q = select(Player).where(Player.full_name == rec.full_name)
        if rec.position:
            q = q.where(Player.position == rec.position)
        p = session.execute(q).scalars().first()
        if p:
            return _enrich_player(p, rec)

        # Normalized-name match for suffix/punctuation mismatches.
        norm = _normalize_name(rec.full_name)
        if norm:
            q = select(Player).where(Player.normalized_name == norm)
            if rec.position:
                q = q.where(Player.position == rec.position)
            p = session.execute(q).scalars().first()
            if p:
                return _enrich_player(p, rec)
            # Last resort: name-only normalized match (no position constraint).
            p = session.execute(
                select(Player).where(Player.normalized_name == norm)
            ).scalars().first()
            if p:
                return _enrich_player(p, rec)

    p = Player(
        sleeper_id=rec.sleeper_id,
        nba_id=rec.nba_id,
        bbref_id=rec.bbref_id,
        full_name=rec.full_name or "(unknown)",
        normalized_name=_normalize_name(rec.full_name),
        position=rec.position,
        nba_team=rec.nba_team,
        age=rec.age,
        years_exp=rec.years_exp,
        draft_year=rec.draft_year,
        draft_round=rec.draft_round,
        draft_pick_overall=rec.draft_pick_overall,
        draft_team=rec.draft_team,
        college=rec.college,
        est_retirement_age=rec.est_retirement_age,
        years_remaining=rec.years_remaining,
    )
    session.add(p)
    session.flush()
    return p


def _enrich_player(p: Player, rec: RankingRecord) -> Player:
    """Fill in missing fields on an existing Player from a RankingRecord."""
    if rec.draft_year and not p.draft_year:
        p.draft_year = rec.draft_year
    if rec.draft_round and not p.draft_round:
        p.draft_round = rec.draft_round
    if rec.draft_pick_overall and not p.draft_pick_overall:
        p.draft_pick_overall = rec.draft_pick_overall
    if rec.draft_team and not p.draft_team:
        p.draft_team = rec.draft_team
    if rec.college and not p.college:
        p.college = rec.college
    if rec.nba_team and not p.nba_team:
        p.nba_team = rec.nba_team
    if rec.position and not p.position:
        p.position = rec.position
    if rec.age and not p.age:
        p.age = rec.age
    if rec.years_exp and not p.years_exp:
        p.years_exp = rec.years_exp
    # Longevity fields refresh on each sync (they actually change).
    if rec.est_retirement_age is not None:
        p.est_retirement_age = rec.est_retirement_age
    if rec.years_remaining is not None:
        p.years_remaining = rec.years_remaining
    if rec.nba_id and not p.nba_id:
        p.nba_id = rec.nba_id
    if rec.bbref_id and not p.bbref_id:
        p.bbref_id = rec.bbref_id
    return p


def sync_source(slug: str) -> int:
    """Run a source by slug. Returns the number of ranking rows written."""
    if slug not in REGISTRY:
        raise KeyError(f"Unknown source slug: {slug}")
    AdapterCls = REGISTRY[slug]
    adapter = AdapterCls()
    count = 0
    try:
        with get_session() as session:
            source_row = _ensure_source_row(session, adapter)
            for rec in adapter.fetch():
                player = _resolve_player(session, rec)
                if player is None:
                    continue
                session.add(Ranking(
                    source_id=source_row.id,
                    player_id=player.id,
                    overall_rank=rec.overall_rank,
                    position_rank=rec.position_rank,
                    market_value=rec.market_value,
                    tier=rec.tier,
                    trend_30d=rec.trend_30d,
                    league_format=rec.league_format,
                    is_dynasty=rec.is_dynasty,
                    is_rookie_only=rec.is_rookie_only,
                    captured_at=rec.captured_at,
                ))
                # DARKO-style impact metrics — write as Evaluations.
                for metric_name, attr in DARKO_METRIC_FIELDS:
                    val = getattr(rec, attr, None)
                    if val is None:
                        continue
                    session.add(Evaluation(
                        source_id=source_row.id,
                        player_id=player.id,
                        metric=metric_name,
                        value=float(val),
                        captured_at=rec.captured_at,
                    ))
                count += 1
            source_row.last_synced_at = datetime.utcnow()
            source_row.last_sync_status = "ok"
            source_row.last_sync_error = None
    except Exception as e:
        with get_session() as session:
            row = session.execute(select(Source).where(Source.slug == slug)).scalar_one_or_none()
            if row:
                row.last_sync_status = "error"
                row.last_sync_error = str(e)[:1000]
        raise
    finally:
        adapter.close()
    return count


def sync_sleeper_players() -> int:
    """Pull the Sleeper NBA player dict and upsert into the players table.

    Run this BEFORE other sources to populate the canonical ID map.
    """
    adapter = SleeperPlayers()
    try:
        players_dict = adapter.fetch_players_dict()
    finally:
        adapter.close()

    count = 0
    with get_session() as session:
        for sleeper_id, p in players_dict.items():
            full_name = p.get("full_name") or " ".join(
                x for x in (p.get("first_name"), p.get("last_name")) if x
            )
            if not full_name:
                continue
            existing = session.execute(
                select(Player).where(Player.sleeper_id == sleeper_id)
            ).scalar_one_or_none()

            def _str(key):
                v = p.get(key)
                return str(v) if v is not None else None

            # NBA player payload may also carry "espn_id", "yahoo_id".
            if existing:
                existing.full_name = full_name or existing.full_name
                existing.normalized_name = _normalize_name(existing.full_name)
                existing.first_name = p.get("first_name") or existing.first_name
                existing.last_name = p.get("last_name") or existing.last_name
                existing.position = p.get("position") or existing.position
                existing.nba_team = p.get("team") or existing.nba_team
                existing.espn_id = _str("espn_id") or existing.espn_id
                existing.yahoo_id = _str("yahoo_id") or existing.yahoo_id
                existing.years_exp = p.get("years_exp") if p.get("years_exp") is not None else existing.years_exp
                existing.college = p.get("college") or existing.college
                if p.get("age") is not None:
                    existing.age = float(p["age"])
                existing.is_active = p.get("active", existing.is_active)
            else:
                session.add(Player(
                    sleeper_id=sleeper_id,
                    full_name=full_name,
                    normalized_name=_normalize_name(full_name),
                    first_name=p.get("first_name"),
                    last_name=p.get("last_name"),
                    position=p.get("position"),
                    nba_team=p.get("team"),
                    espn_id=_str("espn_id"),
                    yahoo_id=_str("yahoo_id"),
                    years_exp=p.get("years_exp"),
                    age=float(p["age"]) if p.get("age") is not None else None,
                    college=p.get("college"),
                    is_active=p.get("active", True),
                ))
            count += 1
    return count
