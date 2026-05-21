"""Sync logic: run a source adapter and persist its records.

Player resolution order:
  1. sleeper_id (canonical, if the record carries it)
  2. nba_id (BBRef / nba_api)
  3. bbref_id
  4. NameResolver tier cascade (Tier 1 / Tier 2 / alias / Tier 3)

If no match, a minimal Player row is auto-created so we never drop data.
The NameResolver collapses cross-source spelling variants ("Nicolas
↔ Nic", "Alexandre ↔ Alex", diacritics) before that fallback fires,
which means the orphan "DARKO-only" rows that previously haunted the
top-300 join cleanly onto their Sleeper-backed siblings.

The DARKO adapter also writes Evaluation rows for the underlying DPM
components and player-level longevity fields (est_retirement_age,
years_remaining) onto the Player row.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from sqlalchemy import select

from .db.session import get_session
from .db.models import Source, Player, Ranking, Evaluation
from .sources import REGISTRY
from .sources.base import BaseSource, RankingRecord
from .sources.sleeper_players import SleeperPlayers
from .names import normalize as _normalize_name
from .name_resolver import (
    NameResolver,
    ResolverCandidate,
    ResolverQuery,
    ResolverStats,
    load_alias_map,
    normalize_team,
)

log = logging.getLogger(__name__)

# Sources whose ranking rows MUST come with a Basketball-Reference signal
# to count toward the top-300. Records whose underlying Player has no
# bbref linkage after the full sync will be dropped from rankings.
# (See Phil's directive: "if there are any players who cannot be found
# on basketball-reference after the fuzzy match… do not include them
# in the model.")
_DIAGNOSTICS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "diagnostics"
_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


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


def _build_resolver(session) -> NameResolver:
    """Build a NameResolver from the current Player pool."""
    cands = []
    for p in session.execute(select(Player)).scalars().all():
        cands.append(
            ResolverCandidate.from_fields(
                player_id=p.id,
                full_name=p.full_name or "",
                position=p.position,
                nba_team=p.nba_team,
            )
        )
    return NameResolver(cands, alias_map=load_alias_map())


def _resolve_player(
    session,
    rec: RankingRecord,
    resolver: NameResolver | None = None,
    stats: ResolverStats | None = None,
) -> Player | None:
    # 1–3: cheap ID-based lookups still come first.
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

    # 4: NameResolver tier cascade (built once per sync).
    if resolver is not None and rec.full_name:
        hit, tier = resolver.resolve(
            ResolverQuery(
                full_name=rec.full_name,
                position=rec.position,
                nba_team=rec.nba_team,
            )
        )
        if hit is not None:
            p = session.get(Player, hit.player_id)
            if p is not None:
                if stats is not None:
                    setattr(stats, tier, getattr(stats, tier) + 1)
                return _enrich_player(p, rec)

    # No resolver wired in OR the resolver couldn't find a match. Fall
    # back to the historical name-based lookup so we never silently drop
    # data — then create a fresh Player row as a last resort.
    if rec.full_name:
        q = select(Player).where(Player.full_name == rec.full_name)
        if rec.position:
            q = q.where(Player.position == rec.position)
        p = session.execute(q).scalars().first()
        if p:
            return _enrich_player(p, rec)

        norm = _normalize_name(rec.full_name)
        if norm:
            p = session.execute(
                select(Player).where(Player.normalized_name == norm)
            ).scalars().first()
            if p:
                return _enrich_player(p, rec)

    # Create a new player row — normalize the team to an abbrev on the
    # way in so future sources can join cleanly.
    p = Player(
        sleeper_id=rec.sleeper_id,
        nba_id=rec.nba_id,
        bbref_id=rec.bbref_id,
        full_name=rec.full_name or "(unknown)",
        normalized_name=_normalize_name(rec.full_name),
        position=rec.position,
        nba_team=normalize_team(rec.nba_team) or rec.nba_team,
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
    if resolver is not None and rec.full_name:
        resolver.add(
            ResolverCandidate.from_fields(
                p.id, p.full_name, p.position, p.nba_team
            )
        )
    if stats is not None:
        stats.unresolved += 1
        stats.unmatched.append(
            {
                "sleeper_name": rec.full_name,
                "team": rec.nba_team,
                "position": rec.position,
                "sleeper_id": rec.sleeper_id,
                "player_id": p.id,
                "reason": "no_bbref_match_after_tier3",
            }
        )
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
    # Team enrichment: always prefer a 3-letter abbrev over a full team
    # name. If the existing row carries the verbose form (e.g. an old
    # DARKO orphan row), replace it.
    if rec.nba_team:
        new_team = normalize_team(rec.nba_team) or rec.nba_team
        existing_team = p.nba_team
        if not existing_team:
            p.nba_team = new_team
        elif normalize_team(existing_team) is None and normalize_team(new_team):
            # existing is verbose, new is an abbrev — upgrade.
            p.nba_team = new_team
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


def sync_source(slug: str, stats: ResolverStats | None = None) -> int:
    """Run a source by slug. Returns the number of ranking rows written.

    If ``stats`` is provided, resolver tier counts and unmatched entries
    are accumulated into it so the headless launcher can write a single
    diagnostics sidecar at the end of the sync step.
    """
    if slug not in REGISTRY:
        raise KeyError(f"Unknown source slug: {slug}")
    AdapterCls = REGISTRY[slug]
    adapter = AdapterCls()
    count = 0
    try:
        with get_session() as session:
            source_row = _ensure_source_row(session, adapter)
            resolver = _build_resolver(session)
            for rec in adapter.fetch():
                if stats is not None:
                    stats.total += 1
                player = _resolve_player(session, rec, resolver=resolver, stats=stats)
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


def write_resolver_diagnostics(stats: ResolverStats) -> Path:
    """Persist resolver stats + the unmatched players list to disk.

    Files:
      data/diagnostics/resolver_stats.json   — numbers (with timestamp)
      data/diagnostics/unmatched_players.json — the players that fell
                                                through every tier
    """
    _DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = _DIAGNOSTICS_DIR / "resolver_stats.json"
    unmatched_path = _DIAGNOSTICS_DIR / "unmatched_players.json"
    payload = stats.as_dict()
    payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    unmatched_path.write_text(
        json.dumps(stats.unmatched, indent=2),
        encoding="utf-8",
    )
    return stats_path


def _merge_player_rows(session, keep: Player, dupes: list[Player]) -> int:
    """Union enrich + re-point Rankings/Evaluations onto the keeper, then
    delete the dupes. Returns the number of rows removed."""
    if not dupes:
        return 0
    for d in dupes:
        if not keep.position and d.position:
            keep.position = d.position
        if not normalize_team(keep.nba_team) and d.nba_team:
            keep.nba_team = normalize_team(d.nba_team) or d.nba_team
        if d.age is not None and (keep.age is None or d.age < keep.age):
            keep.age = d.age
        if not keep.nba_id and d.nba_id:
            keep.nba_id = d.nba_id
        if not keep.bbref_id and d.bbref_id:
            keep.bbref_id = d.bbref_id
        if not keep.sleeper_id and d.sleeper_id:
            keep.sleeper_id = d.sleeper_id
        if not keep.years_remaining and d.years_remaining is not None:
            keep.years_remaining = d.years_remaining
        if not keep.est_retirement_age and d.est_retirement_age is not None:
            keep.est_retirement_age = d.est_retirement_age
        session.execute(
            Ranking.__table__.update()
            .where(Ranking.player_id == d.id)
            .values(player_id=keep.id)
        )
        session.execute(
            Evaluation.__table__.update()
            .where(Evaluation.player_id == d.id)
            .values(player_id=keep.id)
        )
        session.delete(d)
    session.flush()
    return len(dupes)


def dedup_players_by_canonical(stats: ResolverStats | None = None) -> int:
    """Post-sync safety net: collapse any leftover dupe Player rows.

    The resolver prevents new dupes from being created — but the DB
    inherited from PR #5 already contains the orphan rows that triggered
    this PR (#2033–#2038 etc.). This pass runs in three stages:

      1. Group by canonical key (Tier 1) and merge.
      2. Use the full NameResolver on the remaining orphan-shaped rows
         (no bbref_id / no nba_id) to find a Tier-2 / alias / Tier-3
         match against a fully-identified Player. Merge if found.
      3. Normalize team strings ("Washington Wizards" → "WAS") and
         refresh ``normalized_name`` on every remaining row.

    Returns the total number of duplicate rows removed.
    """
    from collections import defaultdict
    from .name_resolver import canonical_key, NameResolver, ResolverCandidate, ResolverQuery

    removed = 0
    with get_session() as session:
        # Stage 1: canonical-key collapse.
        groups: dict[str, list[Player]] = defaultdict(list)
        for p in session.execute(select(Player)).scalars().all():
            key = canonical_key(p.full_name)
            if not key:
                continue
            groups[key].append(p)

        for key, members in groups.items():
            if len(members) < 2:
                continue

            def _id_score(pl: Player) -> tuple:
                team_abbr = normalize_team(pl.nba_team)
                return (
                    int(bool(team_abbr)),
                    int(bool(pl.position)),
                    int(bool(pl.bbref_id or pl.nba_id)),
                    -pl.id,
                )

            members.sort(key=_id_score, reverse=True)
            keep, dupes = members[0], members[1:]
            removed += _merge_player_rows(session, keep, dupes)

        # Stage 2: orphan-shaped rows that lost the Tier-1 race — try
        # the resolver against the *good* Player pool. An orphan-shaped
        # row has no sleeper_id, no bbref_id, no nba_id, no position
        # (typical DARKO-only intake from before the resolver landed).
        all_players = session.execute(select(Player)).scalars().all()
        good_cands = [
            ResolverCandidate.from_fields(
                p.id, p.full_name or "", p.position, p.nba_team
            )
            for p in all_players
            if (p.sleeper_id or p.bbref_id or p.nba_id) and (p.position or normalize_team(p.nba_team))
        ]
        resolver = NameResolver(good_cands)

        for p in all_players:
            if p.sleeper_id or p.bbref_id or p.nba_id:
                continue
            if not p.full_name:
                continue
            hit, tier = resolver.resolve(
                ResolverQuery(
                    full_name=p.full_name,
                    position=p.position,
                    nba_team=p.nba_team,
                )
            )
            if hit is None or hit.player_id == p.id:
                continue
            keep = session.get(Player, hit.player_id)
            if keep is None:
                continue
            removed += _merge_player_rows(session, keep, [p])

        # Stage 3: normalize teams + normalized_name on every row left.
        for p in session.execute(select(Player)).scalars().all():
            if p.nba_team:
                abbrev = normalize_team(p.nba_team)
                if abbrev and abbrev != p.nba_team:
                    p.nba_team = abbrev
            if p.full_name:
                p.normalized_name = _normalize_name(p.full_name)

    if removed:
        log.info("dedup_players_by_canonical removed %d rows", removed)
    return removed


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
