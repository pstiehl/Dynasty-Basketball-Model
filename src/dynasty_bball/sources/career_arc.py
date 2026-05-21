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
from ..similarity import (
    build_corpus_profiles,
    build_career_index,
    find_comparables,
    project_career,
    rescale_to_0_100,
    prepare_corpus_for_search,
)
from ..similarity.vectorize import (
    build_profile_from_stats,
    derive_position_bucket,
)


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


def build_projections(
    current_cache_dir: Path = BBREF_CACHE_DIR,
    current_season: str = BBREF_DEFAULT_SEASON,
    historical_cache_dir: Path = HISTORICAL_CACHE_DIR,
    historical_start_year: int = DEFAULT_START_YEAR,
    historical_end_season: str = HISTORICAL_END_SEASON,
    min_games: int = MIN_GAMES_DEFAULT,
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

    results: dict = {
        "n_historical_seasons": len(rows),
        "n_current_players": len(targets),
        "targets_by_id": targets_by_id,
        "comps_by_target": comps_by_target,
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
        out: list[RankingRecord] = []

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
            sidecar_payload["by_nba_id"][nba_id] = {
                "top_comparables": [_comparable_to_dict(c) for c in comps[:TOP_COMPS_FOR_UI]],
                "n_comparables": len(comps),
                "by_format": dynasty_by_id.get(nba_id, {}),
            }
        try:
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_payload, f, separators=(",", ":"))
        except Exception as e:
            log.warning("career_arc: failed to write comparables sidecar: %s", e)

        for fmt in ("points_dhk", "points_default"):
            block = results.get(fmt) or {}
            projections = block.get("projections", []) or []
            # Sort by dynasty_value desc → overall_rank.
            projections_sorted = sorted(
                projections,
                key=lambda p: p.dynasty_value,
                reverse=True,
            )
            for rank, proj in enumerate(projections_sorted, start=1):
                t = targets_by_id.get(proj.player_nba_id)
                out.append(RankingRecord(
                    source_slug=self.slug,
                    nba_id=proj.player_nba_id,
                    full_name=proj.player_name,
                    nba_team=t.team if t else None,
                    age=proj.player_age,
                    overall_rank=rank,
                    market_value=proj.dynasty_value,
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
