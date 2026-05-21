"""Headless launcher — used by GitHub Actions (no browser to open).

Steps (mirror the football repo):

  1/6 init-db
  2/6 sync-players (Sleeper NBA)
  3/6 sync-all (DARKO; future PRs add more)
  4/6 score in both league formats (points_dhk, points_default)
  5/6 build site
  6/6 prefetch leagues from leagues.json
"""
from __future__ import annotations
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("Dynasty Basketball Model — headless refresh (CI)")
    print("=" * 60)

    # Step 1: init DB
    print("\n[1/6] Initializing database...")
    try:
        from dynasty_bball.db.session import init_db
        init_db()
        print("  OK")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # Step 2: Sleeper NBA players
    print("\n[2/6] Loading NBA player metadata from Sleeper...")
    try:
        from dynasty_bball.sync import sync_sleeper_players
        n = sync_sleeper_players()
        print(f"  OK ({n:,} NBA players from Sleeper)")
    except Exception as e:
        print(f"  WARN: {e}")
        print("  Continuing without Sleeper player map (resolution will rely on name only).")

    # Step 3: Sync data sources
    print("\n[3/6] Syncing data sources...")
    from dynasty_bball.sync import sync_source

    synced_any = False
    sources_to_sync = [
        ("darko", "DARKO"),
        ("court_consensus", "Court Consensus"),
        ("vecenie", "Sam Vecenie"),
        ("basketball_reference", "Basketball-Reference"),
    ]
    for slug, label in sources_to_sync:
        try:
            n = sync_source(slug)
            print(f"  {label}: {n:,} rows")
            if n > 0:
                synced_any = True
        except Exception as e:
            print(f"  {label}: FAILED ({e})")

    # Starter pack (empty in PR #1 — kept so the path exists)
    try:
        from dynasty_bball.starter_pack import import_starter_pack
        n = import_starter_pack()
        print(f"  Starter pack: {n} rows")
        if n > 0:
            synced_any = True
    except Exception as e:
        print(f"  Starter pack: FAILED ({e})")

    if not synced_any:
        print("\nERROR: No sources synced successfully. Cannot build site.")
        sys.exit(1)

    # Step 4: Score in both formats
    print("\n[4/6] Computing composite scores...")
    try:
        from dynasty_bball.scoring import compute_composite_scores
        # DARKO only emits to points_dhk by default; duplicate the rankings
        # to points_default so we can score that format too. (In future PRs,
        # adapters that produce production-based rankings can emit format-
        # specific values directly.)
        _duplicate_rankings_to_format("points_dhk", "points_default")
        for fmt in ["points_dhk", "points_default"]:
            n = compute_composite_scores(league_format=fmt)
            print(f"  {fmt}: {n:,} players scored")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # Step 5: Build site (Phil's format)
    print("\n[5/6] Building site...")
    try:
        from dynasty_bball.report import generate_site
        out = generate_site(output_dir="dynasty_site", league_format="points_dhk", limit=300)
        print(f"  OK -> {out}")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # Step 6: Pre-fetch any leagues in leagues.json
    print("\n[6/6] Pre-fetching listed leagues...")
    try:
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "scripts"))
        import prefetch_leagues
        summary = prefetch_leagues.prefetch_all()
        ok_count = len(summary.get("leagues", []))
        err_count = len(summary.get("errors", []))
        print(f"  Pre-fetched {ok_count} leagues, {err_count} errors")
        for L in summary.get("leagues", []):
            print(f"    {L['slug']:>40}  teams={L['n_teams']:>2}  managers={L['n_managers']:>2}  ({L['name']})")
        for err in summary.get("errors", []):
            print(f"    [error] {err['entry']}: {err['error']}")
    except Exception as e:
        print(f"  WARN: pre-fetch step failed: {e}")
        print("  (Site still builds without pre-fetched leagues.)")

    print("\nDone.")


def _duplicate_rankings_to_format(
    src_fmt: str,
    dst_fmt: str,
    source_slugs: tuple[str, ...] = ("darko",),
) -> int:
    """Clone Rankings from src_fmt to dst_fmt for format-agnostic sources.

    DARKO's market_value is a longevity-adjusted impact scalar that does
    not depend on scoring format, so we duplicate its ``points_dhk``
    rows to ``points_default`` to keep that format's composite alive.
    Court Consensus and Vecenie emit per-format records directly during
    fetch and are NOT duplicated here (passing source_slugs scopes the
    operation). Once a real production-based adapter lands and DARKO
    too becomes per-format, this helper becomes a no-op.
    """
    from datetime import datetime
    from sqlalchemy import select, func
    from dynasty_bball.db.session import get_session
    from dynasty_bball.db.models import Ranking, Source

    n = 0
    with get_session() as session:
        src_ids = [
            s.id
            for s in session.execute(
                select(Source).where(Source.slug.in_(source_slugs))
            ).scalars().all()
        ]
        if not src_ids:
            return 0
        latest = session.execute(
            select(func.max(Ranking.captured_at))
            .where(Ranking.league_format == src_fmt)
            .where(Ranking.source_id.in_(src_ids))
        ).scalar_one_or_none()
        if latest is None:
            return 0
        rows = session.execute(
            select(Ranking)
            .where(Ranking.league_format == src_fmt)
            .where(Ranking.source_id.in_(src_ids))
            .where(Ranking.captured_at == latest)
        ).scalars().all()

        # Drop any prior duplicates from these sources at the same
        # captured_at to keep the operation idempotent.
        existing = session.execute(
            select(Ranking)
            .where(Ranking.league_format == dst_fmt)
            .where(Ranking.source_id.in_(src_ids))
            .where(Ranking.captured_at == latest)
        ).scalars().all()
        for e in existing:
            session.delete(e)
        session.flush()

        for r in rows:
            session.add(Ranking(
                source_id=r.source_id,
                player_id=r.player_id,
                overall_rank=r.overall_rank,
                position_rank=r.position_rank,
                market_value=r.market_value,
                tier=r.tier,
                trend_30d=r.trend_30d,
                league_format=dst_fmt,
                is_dynasty=r.is_dynasty,
                is_rookie_only=r.is_rookie_only,
                captured_at=r.captured_at,
            ))
            n += 1
    return n


if __name__ == "__main__":
    main()
