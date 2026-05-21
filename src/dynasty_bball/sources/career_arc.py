"""Career-arc similarity engine — the dominant longevity signal.

This is the source that operationalizes Phil's PR #4 thesis:

  > "DARKO is saying Cooper Flagg will retire at 28? That is clearly
  >  not right. Let's use a higher weight towards similarity scores
  >  using the players age and production of fantasy stats (and
  >  remaining stats for their career) to arrive at the player
  >  rankings."

How it works
------------
1. Load the current-season production rows we already have cached
   (``data/basketball_reference/leaguedash_<season>.json`` — the same
   payload the basketball_reference adapter reads).
2. Load the historical NBA corpus (1980-present) cached under
   ``data/historical_nba/league_<season>.json``.
3. Vectorize every player-season in both into the same profile space,
   z-score against the historical corpus.
4. For each current player at age A: find the top-20 historical
   player-seasons at age A±1 with same/adjacent position bucket and
   highest cosine similarity.
5. Project each current player's remaining career as a similarity-
   weighted aggregate of those comps' actual remaining careers.
6. Emit one RankingRecord per (player, league_format) with
   ``market_value = dynasty_value`` (0..100, top player = 100).

Per-format because each league_format weights stats differently —
the comps' remaining-career fantasy_ppg is format-dependent (a
high-steal/block historical career projects more value in DHK
than in default).

Comparables payload
-------------------
Top-5 comparables are written into the RankingRecord ``notes``
field — a simple JSON-in-string carrier — so the site/report layer
can render them on player pages without doing the comparable
computation a second time.

Default weight: 1.8. This is intentionally the highest weight in the
composite. The thesis of the model is that for dynasty, what matters
is *the rest of a player's career*, and the similarity engine is the
only source we have that produces a forward-looking longevity-aware
fantasy value. DARKO's survival curves were the previous default;
they're broken (Flagg at 28). The DARKO weight drops from 1.5 → 0.8
in tandem (see weights.py and CHANGELOG-model.md v0.4.0).

Rookie / college extension (PR #5)
----------------------------------
``similarity.vectorize.vectorize_college_season`` is a stub. The
career_arc adapter has a parameterized ``_targets_for_cohort`` so
a college cohort can be plugged in. Leaving the integration to PR #5.
"""
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .base import BaseSource, RankingRecord
from .basketball_reference import (
    parse_leaguedash_payload as _parse_current_payload,
    _load_cache as _load_bbref_cache,
    DEFAULT_CACHE_DIR as BBREF_CACHE_DIR,
    DEFAULT_SEASON as BBREF_DEFAULT_SEASON,
    MIN_GAMES_DEFAULT,
)
from .historical_nba import (
    load_corpus,
    DEFAULT_CACHE_DIR as HISTORICAL_CACHE_DIR,
    DEFAULT_START_YEAR,
    DEFAULT_END_SEASON as HISTORICAL_END_SEASON,
    season_end_year,
    HistoricalPlayerSeason,
)
from .historical_ncaa import (
    load_corpus as load_ncaa_corpus,
    DEFAULT_CACHE_DIR as NCAA_CACHE_DIR,
    DEFAULT_START_YEAR as NCAA_START_YEAR,
    DEFAULT_END_YEAR as NCAA_END_YEAR,
)
from ..similarity import (
    build_corpus_profiles,
    build_career_index,
    find_comparables,
    project_career,
    rescale_to_0_100,
    prepare_corpus_for_search,
    build_college_corpus_profiles,
    prepare_ncaa_search_index,
    project_rookie,
    blended_dynasty_value,
    build_bridge,
    save_bridge,
)
from ..similarity.vectorize import (
    build_profile_from_stats,
    derive_position_bucket,
)
from ..similarity.bridge import DEFAULT_BRIDGE_PATH


log = logging.getLogger(__name__)


# How many top-K comps to retain. Phil's spec asks for top-20 in the
# aggregation, top-5 surfaced on the player page.
COMPARABLES_K = 20
TOP_COMPS_FOR_UI = 5
AGE_WINDOW = 1.0


@dataclass
class _CohortTarget:
    """One current player ready for KNN lookup."""
    nba_id: str
    name: str
    age: float
    team: Optional[str]
    profile: object  # vectorize.Profile
    # Carry the per-game stat line so the site can show "current production"
    # next to the projection.
    pts: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float
    tpm: float
    gp: int
    minutes: float


def _build_current_targets(
    cache_dir: Path,
    season: str,
    min_games: int = MIN_GAMES_DEFAULT,
) -> list[_CohortTarget]:
    """Read current-season production via the basketball_reference cache.

    We deliberately reuse the bbref cache instead of pulling fresh —
    the bbref adapter is already responsible for that data freshness,
    and double-fetching wastes API calls.
    """
    payload = _load_bbref_cache(cache_dir, season)
    if payload is None:
        log.warning(
            "career_arc: no basketball_reference cache for season %s in %s — "
            "cannot build current cohort.",
            season, cache_dir,
        )
        return []
    productions = _parse_current_payload(payload, min_games=min_games)
    season_end = season_end_year(season)
    targets: list[_CohortTarget] = []
    for prod in productions:
        if prod.age is None or prod.gp is None or prod.minutes is None:
            continue
        profile = build_profile_from_stats(
            nba_id=prod.nba_id,
            name=prod.name,
            season=season,
            season_end_year=season_end,
            age=prod.age,
            team=prod.team,
            gp=prod.gp,
            mpg=prod.minutes,
            pts=prod.pts,
            reb=prod.reb,
            ast=prod.ast,
            stl=prod.stl,
            blk=prod.blk,
            tov=prod.tov,
            tpm=prod.tpm,
            # bbref _PlayerProduction doesn't carry FGA/FTA — we
            # estimate them from PTS to keep the TS% feature non-zero.
            # ~46% of bucket on FG, ~25% on FT (league averages).
            # This is a coarse approximation; for production use the
            # basketball_reference adapter would need to also carry
            # FGA/FTA (TODO follow-up). It only affects the TS%
            # feature for current players' KNN search, not the
            # historical corpus's TS% values.
            fga=max(0.0, prod.pts / 1.05) if prod.pts else 0.0,
            fta=max(0.0, prod.pts * 0.25) if prod.pts else 0.0,
        )
        targets.append(_CohortTarget(
            nba_id=prod.nba_id,
            name=prod.name,
            age=prod.age,
            team=prod.team,
            profile=profile,
            pts=prod.pts, reb=prod.reb, ast=prod.ast, stl=prod.stl,
            blk=prod.blk, tov=prod.tov, tpm=prod.tpm,
            gp=prod.gp, minutes=prod.minutes,
        ))
    return targets


def _comparable_to_dict(c) -> dict:
    """Compact JSON-able representation of a Comparable for the report."""
    return {
        "name": c.comp_name,
        "season": c.comp_season,
        "age": round(c.comp_age_when_compared, 1),
        "similarity": round(c.similarity, 3),
        "remaining_seasons": c.remaining_seasons,
        "remaining_games": c.remaining_games,
        "remaining_fp_dhk": (
            round(c.remaining_fantasy_ppg_dhk, 2)
            if c.remaining_fantasy_ppg_dhk is not None else None
        ),
        "remaining_fp_default": (
            round(c.remaining_fantasy_ppg_default, 2)
            if c.remaining_fantasy_ppg_default is not None else None
        ),
        "bucket_match": c.bucket_match,
        "censored": c.censored,
    }


def build_rookie_projections(
    *,
    ncaa_corpus_rows: list,
    nba_rows: list,
    nba_career_index,
    bridge_by_pid: dict,
    target_end_year: int = NCAA_END_YEAR,
) -> dict:
    """Project every CURRENT NCAA player (latest season available).

    Returns:
        {
          "projections_by_btv_pid": {
            btv_pid: {
              "name": str, "school": str, "class_year": str,
              "age": float, "position_bucket": str,
              "points_dhk": CareerProjection,
              "points_default": CareerProjection,
              "comparables": [Comparable, ...],
            }
          },
          "n_targets": int,
          "n_with_nba_comps": int,
        }

    Only the most recent season of each NCAA player is used as the
    "target" (subsequent seasons would just be different snapshots of
    the same prospect). For the rookie chain we want every player
    whose latest NCAA season is target_end_year -- i.e. current
    draft prospects + everyone playing their final season of college
    eligibility.
    """
    ncaa_corpus = build_college_corpus_profiles(ncaa_corpus_rows)
    ncaa_index = prepare_ncaa_search_index(ncaa_corpus, ncaa_corpus_rows)

    # Group NCAA rows by btv_pid and keep only the most recent season
    # for each player. The "target" pool is players whose most recent
    # season is at target_end_year -- i.e. current college players.
    by_pid: dict[str, tuple] = {}
    for idx, r in enumerate(ncaa_corpus_rows):
        slot = by_pid.get(r.sr_player_id)
        if slot is None or r.season_end_year > slot[0].season_end_year:
            by_pid[r.sr_player_id] = (r, idx)

    targets_all = [(pid, r, idx) for pid, (r, idx) in by_pid.items()
                   if r.season_end_year == target_end_year]
    # PROSPECT FILTER: limit emitted ranking records to actual draft
    # prospects, not every D1 rotation player. Without an RSCI/ESPN-100
    # list we use a stats-based heuristic that approximates the draft
    # pool: top conferences (P5/HM) AND any of (BPM>=4, PPG>=14, USG>=22),
    # OR (any conference + BPM>=7).
    # ~150-300 players pass per year, matching the rough size of the
    # combine pool. The KNN itself still searches the full corpus --
    # this filter only gates which prospects produce RankingRecords.
    from .historical_ncaa import CONFERENCE_TIER
    def _is_prospect(r):
        bpm = r.bpm if r.bpm is not None else -99
        tier = CONFERENCE_TIER.get(r.conference, "LM")
        # MPG floor of 22: a player with <22 MPG hasn't shown he can
        # sustain NBA minutes. Per-36 stats lie when the sample is
        # 12 MPG of garbage-time bench play. The same threshold the
        # Combine + Big Boards implicitly use to screen prospects.
        if r.mpg is not None and r.mpg < 22.0:
            return False
        if bpm >= 7.0:
            return True
        if tier in ("P5", "HM") and (bpm >= 4.0 or r.pts_pg >= 14.0 or r.usg_pct >= 22.0):
            return True
        return False
    targets = [(pid, r, idx) for pid, r, idx in targets_all if _is_prospect(r)]
    log.info(
        "career_arc: rookie projection for %d current NCAA prospects "
        "(filtered from %d total D1 rotation players via BPM/PPG/USG threshold)",
        len(targets), len(targets_all),
    )

    # Batched KNN: one matmul for all 3000 targets instead of 3000
    # separate matvecs. ~50x speedup.
    from ..similarity.rookie import find_college_comparables_batch, _extrapolate_censored
    from ..similarity.projection import project_career as _proj_career

    batch_targets = [
        (pid, ncaa_corpus.profiles[idx], r.class_year)
        for pid, r, idx in targets
    ]
    comps_by_pid = find_college_comparables_batch(
        targets=batch_targets,
        ncaa_corpus=ncaa_corpus,
        ncaa_index=ncaa_index,
        bridge_by_pid=bridge_by_pid,
        nba_career_index=nba_career_index,
    )

    projections_by_pid: dict[str, dict] = {}
    n_with_nba_comps = 0
    for pid, r, idx in targets:
        prof = ncaa_corpus.profiles[idx]
        age = r.age_at_season if r.age_at_season is not None else 19.5
        entry = {
            "name": r.name,
            "school": r.school,
            "conference": r.conference,
            "class_year": r.class_year,
            "age": age,
            "position_bucket": prof.position_bucket,
            # PR #8: recruiting percentile (barttorvik rec_rank) is
            # consumed by the draft-stock prior as a fallback signal
            # for prospects not yet on any NBA draft board.
            "rec_rank": getattr(r, "rec_rank", None),
        }
        raw_comps = comps_by_pid.get(pid, [])
        # Extrapolate censored comps once -- format-independent.
        comps = [_extrapolate_censored(c) for c in raw_comps]
        nba_having = [c for c in comps if c.remaining_seasons > 0]
        if any(c.remaining_seasons > 0 for c in comps):
            n_with_nba_comps += 1
        for fmt in ("points_dhk", "points_default"):
            if nba_having:
                proj_years_only = _proj_career(
                    nba_id=pid, name=r.name, age=age,
                    comparables=nba_having, league_format=fmt,
                )
                proj = _proj_career(
                    nba_id=pid, name=r.name, age=age,
                    comparables=comps, league_format=fmt,
                )
                proj.projected_remaining_years = proj_years_only.projected_remaining_years
            else:
                proj = _proj_career(
                    nba_id=pid, name=r.name, age=age,
                    comparables=comps, league_format=fmt,
                )
            entry[fmt] = proj
        entry["comparables"] = comps
        projections_by_pid[pid] = entry

    # Rescale dynasty value within the rookie cohort, per-format. We
    # do NOT rescale to the NBA cohort here -- the NBA composite is
    # applied at the surface layer where we know which players are
    # pure rookies vs. blends.
    for fmt in ("points_dhk", "points_default"):
        proj_list = [e[fmt] for e in projections_by_pid.values()]
        rescale_to_0_100(proj_list)

    # PR #8: Apply the draft-stock prior. Boost real lottery picks,
    # penalize undrafted high-BPM mid-major freshmen. The multiplier
    # works on the 0-100 rescaled values, then we re-rescale so the
    # cohort still spans 0-100 after the prior compresses noise to
    # the bottom and bumps real picks to the top.
    from .draft_stock import (
        load_index_or_empty,
        apply_multipliers_to_rookie_entries,
    )
    big_board_index = load_index_or_empty()
    if big_board_index.n_prospects > 0:
        ds_stats = apply_multipliers_to_rookie_entries(
            projections_by_pid, big_board_index,
        )
        log.info(
            "career_arc: draft-stock prior applied to %d/%d rookies; tier counts: %s",
            ds_stats["n_adjusted"], len(projections_by_pid), ds_stats["tier_counts"],
        )
        # Re-rescale so the cohort still spans 0-100 after the multiplier.
        for fmt in ("points_dhk", "points_default"):
            proj_list = [e[fmt] for e in projections_by_pid.values()]
            rescale_to_0_100(proj_list)
    else:
        log.warning(
            "career_arc: draft-stock big board empty; skipping prior. "
            "Run scripts/refresh_draft_stock.py to build the cache."
        )

    return {
        "projections_by_btv_pid": projections_by_pid,
        "n_targets": len(targets),
        "n_with_nba_comps": n_with_nba_comps,
    }


def build_projections(
    current_cache_dir: Path = BBREF_CACHE_DIR,
    current_season: str = BBREF_DEFAULT_SEASON,
    historical_cache_dir: Path = HISTORICAL_CACHE_DIR,
    historical_start_year: int = DEFAULT_START_YEAR,
    historical_end_season: str = HISTORICAL_END_SEASON,
    min_games: int = MIN_GAMES_DEFAULT,
    ncaa_cache_dir: Path = NCAA_CACHE_DIR,
    ncaa_start_year: int = NCAA_START_YEAR,
    ncaa_end_year: int = NCAA_END_YEAR,
) -> dict:
    """Run the full similarity → projection pipeline.

    Returns a dict::

        {
            "league_format": "points_dhk" | "points_default",
            "projections": list[CareerProjection],
            "targets_by_id": {nba_id: _CohortTarget},
            "n_historical_seasons": int,
            "n_current_players": int,
        }

    But shaped for both formats — top-level keys are "points_dhk" and
    "points_default".
    """
    # 1. Historical corpus.
    rows = load_corpus(
        cache_dir=historical_cache_dir,
        start_year=historical_start_year,
        end_season=historical_end_season,
    )
    if not rows:
        log.warning(
            "career_arc: historical corpus is empty (no caches under %s). "
            "Run scripts/backfill_historical_nba.py once locally and commit "
            "the data/historical_nba/ directory.",
            historical_cache_dir,
        )
        return {
            "points_dhk": {"projections": [], "targets_by_id": {}},
            "points_default": {"projections": [], "targets_by_id": {}},
            "n_historical_seasons": 0,
            "n_current_players": 0,
        }

    corpus = build_corpus_profiles(rows)
    career_index = build_career_index(rows)
    # Precompute the search index ONCE — reused across both formats
    # and every current player. Cuts cohort projection from several
    # minutes to a couple of seconds.
    search_index = prepare_corpus_for_search(corpus, career_index)

    # 2. Current cohort.
    targets = _build_current_targets(
        cache_dir=current_cache_dir,
        season=current_season,
        min_games=min_games,
    )
    targets_by_id = {t.nba_id: t for t in targets}

    # 3. KNN + projection per league_format.
    # Compute comps ONCE per current player (they're format-independent
    # — the comp list is purely about production-profile similarity);
    # then run the per-format projection step on the cached comps.
    comps_by_target: dict = {}
    for t in targets:
        comps_by_target[t.nba_id] = find_comparables(
            target_profile=t.profile,
            corpus=corpus,
            career_index=career_index,
            k=COMPARABLES_K,
            age_window=AGE_WINDOW,
            exclude_nba_id=t.nba_id,
            search_index=search_index,
        )

    # ------------------------------------------------------------------
    # PR #7: NCAA corpus + bridge + rookie projections.
    # ------------------------------------------------------------------
    ncaa_rows = load_ncaa_corpus(
        cache_dir=ncaa_cache_dir,
        start_year=ncaa_start_year,
        end_year=ncaa_end_year,
    )
    if ncaa_rows:
        # Build the bridge from BOTH historical and current-season NBA
        # players. The historical corpus drives the realized-career
        # rollups, but the bridge also needs to know about CURRENT
        # players (Cooper Flagg, drafted 2025-26 but absent from the
        # historical 1980-2024 corpus) so we can attach their college
        # comps. We synthesize lightweight HistoricalPlayerSeason
        # objects for current targets and feed them through the bridge.
        from .historical_nba import HistoricalPlayerSeason
        current_synth_rows = []
        for t in targets:
            current_synth_rows.append(HistoricalPlayerSeason(
                nba_id=t.nba_id, name=t.name, season=current_season,
                season_end_year=season_end_year(current_season),
                age=t.age, team=t.team, gp=t.gp, minutes=t.minutes,
                pts=t.pts, reb=t.reb, ast=t.ast, stl=t.stl, blk=t.blk,
                tov=t.tov, tpm=t.tpm,
                fga=0.0, fta=0.0, fgm=0.0, ftm=0.0, fg_pct=0.0, ft_pct=0.0,
            ))
        bridge_rows = rows + current_synth_rows
        bridge = build_bridge(bridge_rows, ncaa_rows)
        try:
            save_bridge(bridge)
        except Exception as e:
            log.warning("career_arc: failed to save bridge: %s", e)
        rookie_results = build_rookie_projections(
            ncaa_corpus_rows=ncaa_rows,
            nba_rows=rows,
            nba_career_index=career_index,
            bridge_by_pid=bridge["by_btv_pid"],
            target_end_year=ncaa_end_year,
        )
        log.info(
            "career_arc: NCAA corpus %d rows; bridge matched %d/%d NBA players "
            "(post-2008 coverage %.1f%%); %d rookie projections built",
            len(ncaa_rows),
            bridge["n_nba_players_matched"],
            bridge["n_nba_players_total"],
            100.0 * bridge["n_nba_players_matched"] / max(1, bridge["n_nba_players_total"] - bridge["n_pre_corpus_nba_players"]),
            rookie_results["n_targets"],
        )
    else:
        bridge = None
        rookie_results = {"projections_by_btv_pid": {}, "n_targets": 0, "n_with_nba_comps": 0}
        log.warning(
            "career_arc: NCAA corpus is empty (no caches under %s). "
            "Run scripts/backfill_historical_ncaa.py once locally and commit "
            "the data/historical_ncaa/ directory. Rookie chain disabled.",
            ncaa_cache_dir,
        )

    results: dict = {
        "n_historical_seasons": len(rows),
        "n_current_players": len(targets),
        "n_ncaa_seasons": len(ncaa_rows),
        "n_rookie_projections": rookie_results["n_targets"],
        "targets_by_id": targets_by_id,
        "comps_by_target": comps_by_target,
        "rookie_results": rookie_results,
        "bridge": bridge,
    }
    for fmt in ("points_dhk", "points_default"):
        projections = []
        for t in targets:
            comps = comps_by_target.get(t.nba_id, [])
            proj = project_career(
                nba_id=t.nba_id,
                name=t.name,
                age=t.age,
                comparables=comps,
                league_format=fmt,
            )
            projections.append(proj)
        rescale_to_0_100(projections)
        results[fmt] = {"projections": projections, "targets_by_id": targets_by_id}
    return results


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class CareerArc(BaseSource):
    slug = "career_arc"
    name = "Career-Arc Similarity Engine"
    category = "model"
    update_frequency = "daily"
    tos_compliant = True
    # Dominant weight — see CHANGELOG-model.md v0.4.0 for the rationale.
    default_weight = 1.8
    homepage = "https://www.nba.com/stats/"
    notes = (
        "KNN over a 1980-present historical NBA corpus. For each current "
        "player, finds the top-20 historical player-seasons at the same "
        "age (±1) with matching production profile, then projects "
        "remaining-career fantasy points by aggregating those comps' "
        "actual remaining careers. This is the new dominant longevity "
        "signal in the composite (default_weight=1.8). Replaces DARKO's "
        "survival model as the primary longevity input."
    )

    BBREF_CACHE_DIR = BBREF_CACHE_DIR
    BBREF_SEASON = BBREF_DEFAULT_SEASON
    HISTORICAL_CACHE_DIR = HISTORICAL_CACHE_DIR
    HISTORICAL_START_YEAR = DEFAULT_START_YEAR
    HISTORICAL_END_SEASON = HISTORICAL_END_SEASON
    MIN_GAMES = MIN_GAMES_DEFAULT

    def __init__(
        self,
        client=None,
        bbref_cache_dir: Optional[Path | str] = None,
        bbref_season: Optional[str] = None,
        historical_cache_dir: Optional[Path | str] = None,
        historical_start_year: Optional[int] = None,
        historical_end_season: Optional[str] = None,
        min_games: Optional[int] = None,
    ):
        super().__init__(client=client)
        if bbref_cache_dir is not None:
            self.BBREF_CACHE_DIR = Path(bbref_cache_dir)
        elif os.environ.get("DYNASTY_BBALL_BBREF_CACHE_DIR"):
            self.BBREF_CACHE_DIR = Path(os.environ["DYNASTY_BBALL_BBREF_CACHE_DIR"])
        if bbref_season is not None:
            self.BBREF_SEASON = bbref_season
        elif os.environ.get("DYNASTY_BBALL_BBREF_SEASON"):
            self.BBREF_SEASON = os.environ["DYNASTY_BBALL_BBREF_SEASON"]
        if historical_cache_dir is not None:
            self.HISTORICAL_CACHE_DIR = Path(historical_cache_dir)
        elif os.environ.get("DYNASTY_BBALL_HISTORICAL_CACHE_DIR"):
            self.HISTORICAL_CACHE_DIR = Path(os.environ["DYNASTY_BBALL_HISTORICAL_CACHE_DIR"])
        if historical_start_year is not None:
            self.HISTORICAL_START_YEAR = historical_start_year
        if historical_end_season is not None:
            self.HISTORICAL_END_SEASON = historical_end_season
        if min_games is not None:
            self.MIN_GAMES = min_games

    def fetch(self) -> Iterator[RankingRecord]:
        results = build_projections(
            current_cache_dir=self.BBREF_CACHE_DIR,
            current_season=self.BBREF_SEASON,
            historical_cache_dir=self.HISTORICAL_CACHE_DIR,
            historical_start_year=self.HISTORICAL_START_YEAR,
            historical_end_season=self.HISTORICAL_END_SEASON,
            min_games=self.MIN_GAMES,
        )
        if results["n_historical_seasons"] == 0:
            return iter([])

        captured_at = datetime.utcnow()
        targets_by_id = results.get("targets_by_id", {})
        comps_by_target = results.get("comps_by_target", {})
        rookie_results = results.get("rookie_results") or {}
        rookie_by_pid = rookie_results.get("projections_by_btv_pid", {}) or {}
        bridge = results.get("bridge") or {}
        bridge_by_nba_id = (bridge or {}).get("by_nba_id", {})
        out: list[RankingRecord] = []

        # Per-NBA-player NBA-season count (used by the blend logic).
        nba_season_count: dict[str, int] = {}
        for t in targets_by_id.values():
            seasons = (
                results.get("_career_seasons_per_player")
                or {}
            )
        # Easier: walk the career_index seasons directly.
        # build_projections didn't surface it, so re-derive from rows.
        from .historical_nba import load_corpus as _load_nba
        try:
            _all_nba = _load_nba(
                cache_dir=self.HISTORICAL_CACHE_DIR,
                start_year=self.HISTORICAL_START_YEAR,
                end_season=self.HISTORICAL_END_SEASON,
            )
            for r in _all_nba:
                nba_season_count[r.nba_id] = nba_season_count.get(r.nba_id, 0) + 1
        except Exception:
            pass
        # Also count the current season for every current-cohort player
        # (the historical corpus ends one season before current).
        for t in targets_by_id.values():
            nba_season_count[t.nba_id] = nba_season_count.get(t.nba_id, 0) + 1

        # Compute, per NBA player, their rookie-side projection (only
        # meaningful when the bridge has a match -> a btv_pid). Then
        # blend with the NBA-side projection per the season-count rule:
        #   0 NBA seasons -> rookie only (covered separately by pure-
        #                    rookie records keyed by btv_pid below)
        #   1 NBA season  -> 0.5 * rookie + 0.5 * nba
        #   2+ NBA seasons-> nba only
        rookie_projection_by_nba_id: dict[str, dict] = {}
        for nba_id, info in bridge_by_nba_id.items():
            btv_pid = info.get("btv_pid")
            if not btv_pid:
                continue
            rookie_entry = rookie_by_pid.get(btv_pid)
            if not rookie_entry:
                continue
            rookie_projection_by_nba_id[nba_id] = rookie_entry

        blended_by_id_and_fmt: dict[tuple, float] = {}
        for fmt in ("points_dhk", "points_default"):
            block = results.get(fmt) or {}
            for proj in block.get("projections", []):
                nba_id = proj.player_nba_id
                n_seasons = nba_season_count.get(nba_id, 1)
                rk_entry = rookie_projection_by_nba_id.get(nba_id)
                rk_dv = (
                    rk_entry[fmt].dynasty_value
                    if rk_entry is not None else None
                )
                blended = blended_dynasty_value(
                    rookie_dv=rk_dv,
                    nba_dv=proj.dynasty_value,
                    n_nba_seasons=n_seasons,
                )
                if blended is None:
                    blended = proj.dynasty_value
                blended_by_id_and_fmt[(nba_id, fmt)] = blended
                # Mutate so dynasty_value reflects the blended value.
                proj.dynasty_value = float(blended)

        # ----------------------------------------------------------
        # PURE ROOKIES — in the rookie cohort but NOT in the bbref
        # current-season cohort. These need their own RankingRecord
        # output. We DON'T have an nba_id for them yet, so we mint a
        # synthetic id ("ncaa:<btv_pid>") that downstream merging by
        # canonical name will collapse onto the existing Player row
        # (e.g. Sleeper has Cooper Flagg pre-listed and that row is
        # what the resolver attaches our record to).
        # ----------------------------------------------------------
        bridged_pids = {info["btv_pid"] for info in bridge_by_nba_id.values() if info.get("btv_pid")}
        nba_by_target_id = set(targets_by_id.keys())

        # Re-scale across the combined cohort so blended values stay
        # comparable to pure-rookie values. We re-rank by dynasty_value
        # post-blend before emission.
        pure_rookie_records: list[tuple[str, dict, float, float]] = []  # (pid, entry, dv_dhk, dv_default)
        for pid, entry in rookie_by_pid.items():
            if pid in bridged_pids:
                # Already has NBA seasons -> blend handled above.
                continue
            pure_rookie_records.append(
                (pid, entry,
                 float(entry["points_dhk"].dynasty_value),
                 float(entry["points_default"].dynasty_value))
            )

        # Write the comparables sidecar JSON so the site renderer can
        # surface top-5 comps on each player page without re-running
        # the KNN. Keyed by nba_id, format-agnostic (comps depend only
        # on the player's profile, not on league scoring).
        sidecar_path = Path("data/career_arc/comparables.json")
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_payload: dict = {
            "generated_at": captured_at.isoformat(),
            "current_season": self.BBREF_SEASON,
            "by_nba_id": {},
        }
        # We also need per-format dynasty value + projected remaining
        # years so the player page can display the headline numbers.
        dynasty_by_id: dict = {}
        for fmt in ("points_dhk", "points_default"):
            block = results.get(fmt) or {}
            for proj in block.get("projections", []):
                dynasty_by_id.setdefault(proj.player_nba_id, {})[fmt] = {
                    "dynasty_value": round(proj.dynasty_value, 2),
                    "projected_remaining_years": round(proj.projected_remaining_years, 1),
                    "projected_total_fantasy_points": round(proj.projected_total_fantasy_points, 0),
                    "per_year_survival_prob": [round(x, 3) for x in proj.per_year_survival_prob],
                }
        for nba_id, comps in comps_by_target.items():
            entry = {
                "top_comparables": [_comparable_to_dict(c) for c in comps[:TOP_COMPS_FOR_UI]],
                "n_comparables": len(comps),
                "by_format": dynasty_by_id.get(nba_id, {}),
            }
            # Attach rookie comparables (for players who also have a
            # bridged college season -- shown alongside NBA comps).
            rk_entry = rookie_projection_by_nba_id.get(nba_id)
            if rk_entry is not None:
                entry["top_college_comparables"] = [
                    _comparable_to_dict(c) for c in rk_entry["comparables"][:TOP_COMPS_FOR_UI]
                ]
                entry["rookie_by_format"] = {
                    fmt: {
                        "rookie_dynasty_value": round(rk_entry[fmt].dynasty_value, 2),
                        "projected_remaining_years": round(rk_entry[fmt].projected_remaining_years, 1),
                        "projected_total_fantasy_points": round(rk_entry[fmt].projected_total_fantasy_points, 0),
                    }
                    for fmt in ("points_dhk", "points_default")
                }
                entry["n_nba_seasons"] = nba_season_count.get(nba_id, 0)
                entry["blend_strategy"] = (
                    "blend_50_50" if nba_season_count.get(nba_id, 0) == 1
                    else ("nba_only" if nba_season_count.get(nba_id, 0) >= 2 else "rookie_only")
                )
                # PR #8: surface the draft-stock prior on the player page.
                entry["draft_stock"] = {
                    "tier": rk_entry.get("draft_stock_tier"),
                    "multiplier": rk_entry.get("draft_stock_multiplier"),
                    "source": rk_entry.get("draft_stock_source"),
                    "pick": rk_entry.get("draft_stock_pick"),
                    "draft_year": rk_entry.get("draft_stock_draft_year"),
                }
            sidecar_payload["by_nba_id"][nba_id] = entry

        # Pure rookies sidecar block.
        pure_rookies_payload: dict = {}
        for pid, entry, dv_dhk, dv_def in pure_rookie_records:
            pure_rookies_payload[pid] = {
                "name": entry["name"],
                "school": entry["school"],
                "conference": entry["conference"],
                "class_year": entry["class_year"],
                "age": round(entry["age"], 1),
                "position_bucket": entry["position_bucket"],
                "top_college_comparables": [
                    _comparable_to_dict(c) for c in entry["comparables"][:TOP_COMPS_FOR_UI]
                ],
                "by_format": {
                    fmt: {
                        "rookie_dynasty_value": round(entry[fmt].dynasty_value, 2),
                        "projected_remaining_years": round(entry[fmt].projected_remaining_years, 1),
                        "projected_total_fantasy_points": round(entry[fmt].projected_total_fantasy_points, 0),
                    }
                    for fmt in ("points_dhk", "points_default")
                },
                # PR #8: surface the draft-stock prior on the player page.
                "draft_stock": {
                    "tier": entry.get("draft_stock_tier"),
                    "multiplier": entry.get("draft_stock_multiplier"),
                    "source": entry.get("draft_stock_source"),
                    "pick": entry.get("draft_stock_pick"),
                    "draft_year": entry.get("draft_stock_draft_year"),
                },
            }
        sidecar_payload["pure_rookies_by_btv_pid"] = pure_rookies_payload
        sidecar_payload["bridge_summary"] = {
            "n_nba_players_total": (bridge or {}).get("n_nba_players_total", 0),
            "n_nba_players_matched": (bridge or {}).get("n_nba_players_matched", 0),
            "n_pre_corpus_nba_players": (bridge or {}).get("n_pre_corpus_nba_players", 0),
            "match_rate": (bridge or {}).get("match_rate", 0),
            "n_alias_hits": (bridge or {}).get("n_alias_hits", 0),
        }
        sidecar_payload["n_ncaa_seasons"] = results.get("n_ncaa_seasons", 0)
        sidecar_payload["n_rookie_projections"] = results.get("n_rookie_projections", 0)

        try:
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_payload, f, separators=(",", ":"))
        except Exception as e:
            log.warning("career_arc: failed to write comparables sidecar: %s", e)

        # Emit ranking records.
        for fmt in ("points_dhk", "points_default"):
            block = results.get(fmt) or {}
            projections = block.get("projections", []) or []
            # Combine NBA projections + pure rookies in one ranking list.
            # Pure rookies need to be slotted in by dynasty_value.
            combined: list[tuple[float, str, str, Optional[str], float]] = []
            for proj in projections:
                t = targets_by_id.get(proj.player_nba_id)
                combined.append((
                    proj.dynasty_value,
                    proj.player_nba_id,
                    proj.player_name,
                    t.team if t else None,
                    proj.player_age,
                ))
            for pid, entry, dv_dhk, dv_def in pure_rookie_records:
                dv = entry[fmt].dynasty_value
                combined.append((
                    dv,
                    f"ncaa:{pid}",
                    entry["name"],
                    entry["school"],
                    entry["age"],
                ))
            combined.sort(key=lambda x: x[0], reverse=True)
            for rank, (dv, pid_or_id, name, team, age) in enumerate(combined, start=1):
                out.append(RankingRecord(
                    source_slug=self.slug,
                    nba_id=pid_or_id if not pid_or_id.startswith("ncaa:") else None,
                    full_name=name,
                    nba_team=team,
                    age=age,
                    overall_rank=rank,
                    market_value=dv,
                    league_format=fmt,
                    is_dynasty=True,
                    captured_at=captured_at,
                ))

        for r in out:
            yield r


# ---------------------------------------------------------------------------
# Diagnostic exports — used by tests and CLI for inspection.
# ---------------------------------------------------------------------------

def comparables_for_player(
    nba_id: str,
    *,
    league_format: str = "points_dhk",
    bbref_cache_dir: Path = BBREF_CACHE_DIR,
    bbref_season: str = BBREF_DEFAULT_SEASON,
    historical_cache_dir: Path = HISTORICAL_CACHE_DIR,
    historical_start_year: int = DEFAULT_START_YEAR,
    historical_end_season: str = HISTORICAL_END_SEASON,
    k: int = COMPARABLES_K,
    age_window: float = AGE_WINDOW,
) -> dict:
    """Return the full comp + projection report for a single nba_id.

    Mostly for tests, debugging, and the "show me Flagg's comps" CLI.
    """
    rows = load_corpus(
        cache_dir=historical_cache_dir,
        start_year=historical_start_year,
        end_season=historical_end_season,
    )
    corpus = build_corpus_profiles(rows)
    career_index = build_career_index(rows)
    targets = _build_current_targets(
        cache_dir=bbref_cache_dir,
        season=bbref_season,
    )
    target = next((t for t in targets if t.nba_id == nba_id), None)
    if target is None:
        return {"error": f"player {nba_id} not found in current cohort"}
    comps = find_comparables(
        target_profile=target.profile,
        corpus=corpus,
        career_index=career_index,
        k=k,
        age_window=age_window,
        exclude_nba_id=target.nba_id,
    )
    proj = project_career(
        nba_id=target.nba_id,
        name=target.name,
        age=target.age,
        comparables=comps,
        league_format=league_format,
    )
    return {
        "player": {"nba_id": target.nba_id, "name": target.name, "age": target.age},
        "projection": {
            "projected_remaining_years": proj.projected_remaining_years,
            "projected_total_fantasy_points": proj.projected_total_fantasy_points,
            "dynasty_value_raw": proj.dynasty_value_raw,
            "per_year_survival_prob": proj.per_year_survival_prob,
        },
        "top_comparables": [_comparable_to_dict(c) for c in comps[:TOP_COMPS_FOR_UI]],
        "all_comparables": [_comparable_to_dict(c) for c in comps],
    }
