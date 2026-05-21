"""Basketball-Reference / nba_api production adapter.

This is the model's first **production-based** source — every other
source so far (DARKO, Court Consensus, Vecenie) is opinion or impact.
This one is *what actually happened*: realized per-game NBA box-score
production over the most recent regular season, pulled from the NBA
Stats API (the same backend that powers basketball-reference.com).

Why this PR matters
-------------------
Before this adapter, ``points_dhk`` and ``points_default`` produced
**identical** composite rankings because every signal we had (DARKO,
Court Consensus, Vecenie) was scoring-agnostic — none of them changed
shape under different scoring weights. As a result Phil's DHK league
saw the same model as a generic Sleeper points league.

Basketball-Reference fixes that. We compute a real fantasy points-per-
game under each league's actual scoring weights, so a player like Amen
Thompson (high steals, blocks, rebounds, low scoring) ranks visibly
higher under ``points_dhk`` (stl=2.0 / blk=2.0) than under
``points_default`` (stl=3.0 / blk=3.0 but pts=1.0 — the higher pts
weight lifts pure scorers more aggressively in default).

Access
------
nba_api (https://github.com/swar/nba_api) wraps stats.nba.com.
LeagueDashPlayerStats with PerMode=PerGame gives us one row per player
for a full season:

  * PTS, REB, AST, STL, BLK, TOV, FG3M
  * GP, MIN
  * Plus the player's PERSON_ID, TEAM_ABBREVIATION, AGE

We cache the response JSON under ``data/basketball_reference/`` so the
daily CI run does not hammer stats.nba.com (they rate-limit hard). The
cache is keyed by season string; a launcher env var can force a
refresh.

Cache layout::

  data/basketball_reference/leaguedash_<season>.json   # e.g. leaguedash_2024-25.json

Live fetch is gated behind ``DYNASTY_BBALL_BBREF_LIVE=1`` *or* the
cache file being missing. The CI workflow ships the cached JSON in the
repo so the daily build always produces real divergent rankings.

Double-double / triple-double percentages are intentionally omitted in
v1. LeagueDashPlayerStats does not expose them; getting them requires
per-game logs which would 30x the API call count. A follow-up PR can
add a BBRef ``PlayerGameLogs`` cache for that.

Scoring formula
---------------
The adapter does *not* compute fantasy points itself — that would
double up on ``scoring.LEAGUE_SCORING``. Instead it emits per-game
counters on the RankingRecord and a precomputed ``market_value``
per format using ``scoring.per_game_fantasy_points``. The composite
scoring layer picks up ``market_value`` via the existing value-based
normalization branch (same path DARKO uses).

We emit two records per player:

  * ``league_format=points_dhk``     — DHK scoring
  * ``league_format=points_default`` — generic Sleeper points scoring

Default weight: 1.2. Below DARKO (1.5) because BBRef is backward-
looking realized production, not a forward-looking projection. Above
Court Consensus (1.0) because it's hard ground truth, not opinion.
Once a real backtest lands the track-record multiplier will float
this number naturally.
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


log = logging.getLogger(__name__)


# Default season to pull. NBA seasons are formatted "YYYY-YY" by
# stats.nba.com (e.g. 2024-25 means Oct 2024 → Jun 2025). The CI
# workflow can override via env var when the league flips over.
DEFAULT_SEASON = os.environ.get("DYNASTY_BBALL_BBREF_SEASON", "2025-26")

# Where we keep cached LeagueDashPlayerStats payloads.
DEFAULT_CACHE_DIR = Path("data/basketball_reference")

# Minimum games played to be included. Filters out one-game cameos
# whose per-game numbers are noise.
MIN_GAMES_DEFAULT = 10


# nba_api column names we depend on. Keeping these as constants means
# the parser stays trivially testable against a fixture JSON.
COL_PERSON_ID = "PLAYER_ID"
COL_PLAYER_NAME = "PLAYER_NAME"
COL_TEAM_ABBR = "TEAM_ABBREVIATION"
COL_AGE = "AGE"
COL_GP = "GP"
COL_MIN = "MIN"
COL_PTS = "PTS"
COL_REB = "REB"
COL_AST = "AST"
COL_STL = "STL"
COL_BLK = "BLK"
COL_TOV = "TOV"
COL_FG3M = "FG3M"


@dataclass
class _PlayerProduction:
    """Internal per-player container — what we extract from one row."""
    nba_id: str
    name: str
    team: Optional[str]
    age: Optional[float]
    gp: Optional[int]
    minutes: Optional[float]
    pts: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float
    tpm: float


# ---------------------------------------------------------------------------
# Pure parsing — accepts a LeagueDashPlayerStats-shape dict and produces a
# list of _PlayerProduction. Fixture-testable, no network.
# ---------------------------------------------------------------------------

def _row_to_production(
    headers: list[str], row: list, min_games: int = MIN_GAMES_DEFAULT
) -> Optional[_PlayerProduction]:
    """Convert one nba_api result-set row into a _PlayerProduction.

    Returns None if the row is invalid or below the games-played floor.
    """
    try:
        idx = {h: i for i, h in enumerate(headers)}
        name = row[idx[COL_PLAYER_NAME]]
        if not name:
            return None
        gp = _to_int(row[idx.get(COL_GP, -1)] if COL_GP in idx else None)
        if gp is None or gp < min_games:
            return None
        return _PlayerProduction(
            nba_id=str(row[idx[COL_PERSON_ID]]),
            name=str(name).strip(),
            team=row[idx[COL_TEAM_ABBR]] if COL_TEAM_ABBR in idx else None,
            age=_to_float(row[idx[COL_AGE]]) if COL_AGE in idx else None,
            gp=gp,
            minutes=_to_float(row[idx[COL_MIN]]) if COL_MIN in idx else None,
            pts=_to_float(row[idx[COL_PTS]]) or 0.0,
            reb=_to_float(row[idx[COL_REB]]) or 0.0,
            ast=_to_float(row[idx[COL_AST]]) or 0.0,
            stl=_to_float(row[idx[COL_STL]]) or 0.0,
            blk=_to_float(row[idx[COL_BLK]]) or 0.0,
            tov=_to_float(row[idx[COL_TOV]]) or 0.0,
            tpm=_to_float(row[idx[COL_FG3M]]) or 0.0,
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def parse_leaguedash_payload(
    payload: dict, min_games: int = MIN_GAMES_DEFAULT
) -> list[_PlayerProduction]:
    """Parse a LeagueDashPlayerStats JSON payload into _PlayerProductions.

    Expects the nba_api ``get_normalized_dict()`` or
    ``get_dict()`` shape::

        {"resultSets": [{"name": "LeagueDashPlayerStats",
                          "headers": [...],
                          "rowSet": [[...], [...]]}]}

    Returns [] on shape mismatch.
    """
    if not isinstance(payload, dict):
        return []
    rss = payload.get("resultSets") or payload.get("resultSet")
    if not rss:
        return []
    if isinstance(rss, dict):
        rss = [rss]
    target = None
    for r in rss:
        if isinstance(r, dict) and (r.get("name") in ("LeagueDashPlayerStats", "OverallPlayerDashboard") or "headers" in r):
            target = r
            break
    if target is None:
        return []
    headers = target.get("headers") or []
    row_set = target.get("rowSet") or []
    out: list[_PlayerProduction] = []
    for row in row_set:
        rec = _row_to_production(headers, row, min_games=min_games)
        if rec is not None:
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Fantasy-points math — uses scoring.LEAGUE_SCORING so the source of
# truth for stat weights stays in one place.
# ---------------------------------------------------------------------------

def fantasy_ppg(prod: _PlayerProduction, league_format: str) -> float:
    """Compute per-game fantasy points for a player under a league format.

    Uses ``scoring.per_game_fantasy_points`` — the same helper the
    scoring layer would use against a Production row, so the two paths
    cannot drift. Stat keys mapped to LEAGUE_SCORING vocabulary:

        pts → points
        reb → rebounds
        ast → assists
        stl → steals
        blk → blocks
        tov → turnovers (to)
        tpm → three-pointers made
    """
    from ..scoring import LEAGUE_SCORING, per_game_fantasy_points

    scoring = LEAGUE_SCORING.get(league_format, {})
    stats = {
        "pts": prod.pts,
        "reb": prod.reb,
        "ast": prod.ast,
        "stl": prod.stl,
        "blk": prod.blk,
        "to":  prod.tov,
        "tpm": prod.tpm,
    }
    return per_game_fantasy_points(stats, scoring)


def build_records(
    productions: list[_PlayerProduction],
    captured_at: Optional[datetime] = None,
    league_format: str = "points_dhk",
    season: str = DEFAULT_SEASON,
) -> list[RankingRecord]:
    """Build RankingRecords for a single league_format.

    market_value = fantasy_ppg rescaled to 0..100 with the top player
    at 100. overall_rank is by descending fantasy_ppg.
    """
    captured_at = captured_at or datetime.utcnow()
    if not productions:
        return []

    scored: list[tuple[float, _PlayerProduction]] = [
        (fantasy_ppg(p, league_format), p) for p in productions
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[0][0] if scored else 0.0
    if top <= 0:
        top = 1.0  # avoid divide-by-zero on degenerate fixtures

    out: list[RankingRecord] = []
    for rank, (fp, p) in enumerate(scored, start=1):
        mv = round(100.0 * max(0.0, fp) / top, 3)
        out.append(RankingRecord(
            source_slug="basketball_reference",
            nba_id=p.nba_id,
            full_name=p.name,
            nba_team=p.team,
            age=p.age,
            overall_rank=rank,
            market_value=mv,
            league_format=league_format,
            is_dynasty=True,
            captured_at=captured_at,
            per_game_points=p.pts,
            per_game_rebounds=p.reb,
            per_game_assists=p.ast,
            per_game_steals=p.stl,
            per_game_blocks=p.blk,
            per_game_threes=p.tpm,
            per_game_turnovers=p.tov,
        ))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v) -> Optional[int]:
    f = _to_float(v)
    if f is None:
        return None
    try:
        return int(f)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Live fetch / cache
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, season: str) -> Path:
    return cache_dir / f"leaguedash_{season}.json"


def _load_cache(cache_dir: Path, season: str) -> Optional[dict]:
    path = _cache_path(cache_dir, season)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("basketball_reference: cache %s unreadable: %s", path, e)
        return None


def _save_cache(cache_dir: Path, season: str, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, season)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        log.warning("basketball_reference: failed to write cache %s: %s", path, e)


def _live_fetch(season: str) -> Optional[dict]:
    """Pull a fresh LeagueDashPlayerStats payload via nba_api.

    Returns None if the import fails or the call raises. nba_api ships
    with httpx-equivalent retries baked in but stats.nba.com is still
    flaky from CI ranges, hence the broad try/except.
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
    except Exception as e:
        log.warning("basketball_reference: nba_api unavailable: %s", e)
        return None
    try:
        # PerMode=PerGame so we get per-game averages directly.
        # SeasonType=Regular Season (PlayoffStatsRollup is separate).
        ep = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
        )
        # get_dict() returns canonical NBA Stats API JSON with resultSets +
        # headers + rowSet — that's the shape parse_leaguedash_payload
        # expects. get_normalized_dict() flattens to {name: [dict, ...]}
        # which loses the headers list.
        return ep.get_dict()
    except Exception as e:
        log.warning("basketball_reference: live fetch failed for %s: %s", season, e)
        return None


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class BasketballReference(BaseSource):
    slug = "basketball_reference"
    name = "Basketball-Reference / NBA Stats per-game production"
    category = "model"  # ground-truth realized stats, used as a "model" input
    update_frequency = "daily"
    tos_compliant = True
    # Weight reduced from 1.2 → 1.0 in v0.4.0 (PR #4). Career-arc
    # similarity engine (weight 1.8) now dominates the long-horizon
    # signal; BBRef stays as the current-year production signal at
    # parity with Court Consensus. See docs/CHANGELOG-model.md v0.4.0.
    default_weight = 1.0
    homepage = "https://www.basketball-reference.com/"
    notes = (
        "Per-game NBA box-score production pulled via nba_api "
        "(LeagueDashPlayerStats). Cached as JSON under "
        "data/basketball_reference/leaguedash_<season>.json. Drives the "
        "first scoring-format-aware signal in the model — fantasy_ppg "
        "is computed per league_format using scoring.LEAGUE_SCORING so "
        "points_dhk and points_default produce different rankings."
    )

    CACHE_DIR = DEFAULT_CACHE_DIR
    SEASON = DEFAULT_SEASON
    MIN_GAMES = MIN_GAMES_DEFAULT

    def __init__(
        self,
        client=None,
        season: Optional[str] = None,
        cache_dir: Optional[Path | str] = None,
        min_games: Optional[int] = None,
    ):
        super().__init__(client=client)
        if season is not None:
            self.SEASON = season
        elif os.environ.get("DYNASTY_BBALL_BBREF_SEASON"):
            self.SEASON = os.environ["DYNASTY_BBALL_BBREF_SEASON"]

        if cache_dir is not None:
            self.CACHE_DIR = Path(cache_dir)
        elif os.environ.get("DYNASTY_BBALL_BBREF_CACHE_DIR"):
            self.CACHE_DIR = Path(os.environ["DYNASTY_BBALL_BBREF_CACHE_DIR"])

        if min_games is not None:
            self.MIN_GAMES = min_games

    def _load_payload(self) -> Optional[dict]:
        """Cache-first, then live (only when DYNASTY_BBALL_BBREF_LIVE=1 or cache missing).

        We prefer cache because stats.nba.com rate-limits hard and the
        CI build needs to be deterministic. The cached JSON is checked
        into the repo so daily refreshes always have data.
        """
        live_force = os.environ.get("DYNASTY_BBALL_BBREF_LIVE") == "1"
        cached = _load_cache(self.CACHE_DIR, self.SEASON)
        if cached is not None and not live_force:
            return cached
        payload = _live_fetch(self.SEASON)
        if payload is not None:
            _save_cache(self.CACHE_DIR, self.SEASON, payload)
            return payload
        # Fall back to whatever cache we have, even if live was requested.
        return cached

    def fetch(self) -> Iterator[RankingRecord]:
        payload = self._load_payload()
        if not payload:
            log.warning(
                "basketball_reference: no payload (no cache at %s, live failed). "
                "Adapter yields nothing.",
                _cache_path(self.CACHE_DIR, self.SEASON),
            )
            return iter([])

        productions = parse_leaguedash_payload(payload, min_games=self.MIN_GAMES)
        if not productions:
            log.warning("basketball_reference: parsed 0 productions from payload.")
            return iter([])

        captured_at = datetime.utcnow()

        # Emit once per league format. This is the key behavior: the
        # market_value is *format-dependent* (unlike DARKO/CC/Vecenie
        # which use the same value across formats).
        records = build_records(
            productions, captured_at=captured_at, league_format="points_dhk",
            season=self.SEASON,
        )
        records += build_records(
            productions, captured_at=captured_at, league_format="points_default",
            season=self.SEASON,
        )
        for r in records:
            yield r
