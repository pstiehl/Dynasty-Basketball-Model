"""Historical NBA corpus — every player-season since 1980.

Purpose
-------
This source does NOT contribute rankings directly. It exists to feed
``dynasty_bball.similarity`` — the career-arc projection engine. The
similarity engine needs ground-truth historical careers to find
comparables for current players (e.g. "Cooper Flagg at age 18 looks
like Tracy McGrady at age 19; McGrady played 13 more seasons and
averaged X fantasy points per game across them — Flagg's dynasty
value should reflect that").

What it fetches
---------------
For every NBA season from 1980-81 through the current season, we pull
``LeagueDashPlayerStats`` with ``PerMode=PerGame``,
``SeasonType=Regular Season``. Each row carries per-player season
averages: PTS, REB, AST, STL, BLK, TOV, FG3M, FGA, FTA, FG%, FT%, GP,
MIN, AGE. ~45 seasons × ~450 players = ~20K player-seasons.

Why 1980 as the cutoff? The 3-point line arrived in 1979-80, and STL
/ BLK were first tracked league-wide in 1973-74; 1980 gives us 45+
years of *consistently shaped* stat lines so our profile vectors
don't get distorted by missing dimensions.

Caching
-------
Each season is cached under
``data/historical_nba/league_<season>.json``. The cache is committed
to the repo so CI is deterministic — live pulls are gated behind
``DYNASTY_BBALL_HISTORICAL_LIVE=1`` (and also happen automatically
when a cache file is missing). 45 live calls is slow (~3-5 min with
the builtin nba_api retry/backoff) so the expected mode is: run the
backfill once locally, commit, never touch stats.nba.com from CI.

Output
------
This module does NOT emit ``RankingRecord``s into the composite
pipeline. It exposes:

  * ``backfill_seasons(...)`` — populate the cache from stats.nba.com.
  * ``load_corpus(...)`` — read every cached season into a list of
    ``HistoricalPlayerSeason`` records, ready to feed
    ``similarity.vectorize``.
  * ``HistoricalNBA`` — a ``BaseSource`` subclass for plumbing parity
    with the rest of the source registry; ``fetch()`` yields nothing.
    Kept so the launcher can call ``sync_source("historical_nba")``
    cheaply and have the cache verified.

Rate limiting
-------------
nba_api ships with a builtin retry backoff but stats.nba.com still
throttles aggressively when called rapid-fire. We sleep
``HISTORICAL_REQUEST_DELAY_S`` (default 0.6s) between season calls.
"""
from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .base import BaseSource, RankingRecord


log = logging.getLogger(__name__)


# 1980-81 was the first season with both 3PT and STL/BLK tracked league-wide.
DEFAULT_START_YEAR = 1980
# Current season — kept in sync with basketball_reference.DEFAULT_SEASON.
DEFAULT_END_SEASON = os.environ.get("DYNASTY_BBALL_HISTORICAL_END", "2024-25")

DEFAULT_CACHE_DIR = Path("data/historical_nba")

# Be polite to stats.nba.com. They will 429 fast otherwise.
HISTORICAL_REQUEST_DELAY_S = float(os.environ.get("DYNASTY_BBALL_HISTORICAL_DELAY", "0.6"))

# Skip player-seasons with fewer than this many GP — too noisy for
# similarity vectors (per-game numbers blow up).
MIN_GAMES_HISTORICAL = 15


# Column names mirror basketball_reference.py for consistency.
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
COL_FGA = "FGA"
COL_FTA = "FTA"
COL_FGM = "FGM"
COL_FTM = "FTM"
COL_FG_PCT = "FG_PCT"
COL_FT_PCT = "FT_PCT"


@dataclass
class HistoricalPlayerSeason:
    """One player's box-score line for a single season.

    Lightweight container — no NBA-API specific fields. The
    similarity engine consumes lists of these.
    """
    nba_id: str
    name: str
    season: str           # e.g. "2003-04"
    season_end_year: int  # 2004 for "2003-04"
    age: float
    team: Optional[str]
    gp: int
    minutes: float
    pts: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float
    tpm: float
    fga: float
    fta: float
    fgm: float
    ftm: float
    fg_pct: float
    ft_pct: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HistoricalPlayerSeason":
        return cls(**d)


# ---------------------------------------------------------------------------
# Season string helpers
# ---------------------------------------------------------------------------

def season_string(start_year: int) -> str:
    """1980 → "1980-81"."""
    end = (start_year + 1) % 100
    return f"{start_year}-{end:02d}"


def season_end_year(season_str: str) -> int:
    """"1999-00" → 2000, "2024-25" → 2025."""
    start = int(season_str.split("-")[0])
    end_two = int(season_str.split("-")[1])
    # Handle Y2K wraparound: "1999-00" → 2000, "2024-25" → 2025
    if end_two < (start % 100):
        return (start // 100 + 1) * 100 + end_two
    return (start // 100) * 100 + end_two


# ---------------------------------------------------------------------------
# Parsing
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


def parse_season_payload(
    payload: dict,
    season: str,
    min_games: int = MIN_GAMES_HISTORICAL,
) -> list[HistoricalPlayerSeason]:
    """Convert a LeagueDashPlayerStats payload into player-season rows.

    Same payload shape as ``basketball_reference.parse_leaguedash_payload``
    but we keep all the fields we'll need for vectorization (FGA, FTA,
    FG%, FT%).
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
        if isinstance(r, dict) and (
            r.get("name") in ("LeagueDashPlayerStats", "OverallPlayerDashboard")
            or "headers" in r
        ):
            target = r
            break
    if target is None:
        return []
    headers = target.get("headers") or []
    rows = target.get("rowSet") or []
    idx = {h: i for i, h in enumerate(headers)}
    out: list[HistoricalPlayerSeason] = []
    end_year = season_end_year(season)
    for row in rows:
        try:
            name = row[idx[COL_PLAYER_NAME]]
            if not name:
                continue
            gp = _to_int(row[idx[COL_GP]]) if COL_GP in idx else None
            if gp is None or gp < min_games:
                continue
            age = _to_float(row[idx[COL_AGE]]) if COL_AGE in idx else None
            if age is None:
                continue
            mins = _to_float(row[idx[COL_MIN]]) or 0.0
            # Minutes per game floor — sub-10 mpg players are bench fodder
            # whose per-36 numbers explode and pollute the similarity space.
            if mins < 10.0:
                continue
            out.append(HistoricalPlayerSeason(
                nba_id=str(row[idx[COL_PERSON_ID]]),
                name=str(name).strip(),
                season=season,
                season_end_year=end_year,
                age=age,
                team=row[idx[COL_TEAM_ABBR]] if COL_TEAM_ABBR in idx else None,
                gp=gp,
                minutes=mins,
                pts=_to_float(row[idx[COL_PTS]]) or 0.0,
                reb=_to_float(row[idx[COL_REB]]) or 0.0,
                ast=_to_float(row[idx[COL_AST]]) or 0.0,
                stl=_to_float(row[idx[COL_STL]]) or 0.0,
                blk=_to_float(row[idx[COL_BLK]]) or 0.0,
                tov=_to_float(row[idx[COL_TOV]]) or 0.0,
                tpm=_to_float(row[idx[COL_FG3M]]) or 0.0,
                fga=_to_float(row[idx[COL_FGA]]) or 0.0,
                fta=_to_float(row[idx[COL_FTA]]) or 0.0,
                fgm=_to_float(row[idx[COL_FGM]]) or 0.0,
                ftm=_to_float(row[idx[COL_FTM]]) or 0.0,
                fg_pct=_to_float(row[idx[COL_FG_PCT]]) or 0.0,
                ft_pct=_to_float(row[idx[COL_FT_PCT]]) or 0.0,
            ))
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, season: str) -> Path:
    return cache_dir / f"league_{season}.json"


def _load_season_cache(cache_dir: Path, season: str) -> Optional[dict]:
    path = _cache_path(cache_dir, season)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("historical_nba: cache %s unreadable: %s", path, e)
        return None


def _save_season_cache(cache_dir: Path, season: str, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, season)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
    except Exception as e:
        log.warning("historical_nba: failed to write cache %s: %s", path, e)


def _live_fetch_season(season: str) -> Optional[dict]:
    """Pull one season's LeagueDashPlayerStats via nba_api.

    Returns None on failure. nba_api has builtin retry; we still wrap
    in try/except because stats.nba.com 5xx's regularly.
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
    except Exception as e:
        log.warning("historical_nba: nba_api unavailable: %s", e)
        return None
    try:
        ep = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
        )
        return ep.get_dict()
    except Exception as e:
        log.warning("historical_nba: live fetch failed for %s: %s", season, e)
        return None


# ---------------------------------------------------------------------------
# Backfill — one-shot, run locally then commit the cache.
# ---------------------------------------------------------------------------

def backfill_seasons(
    start_year: int = DEFAULT_START_YEAR,
    end_season: str = DEFAULT_END_SEASON,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force: bool = False,
    delay_s: float = HISTORICAL_REQUEST_DELAY_S,
) -> dict:
    """Populate the historical cache from stats.nba.com.

    Skips seasons whose cache file already exists unless ``force=True``.
    Returns a summary dict: ``{"fetched": N, "skipped": M, "failed": [seasons]}``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    end_year = int(end_season.split("-")[0])
    seasons = [season_string(y) for y in range(start_year, end_year + 1)]
    fetched, skipped, failed = 0, 0, []
    for s in seasons:
        path = _cache_path(cache_dir, s)
        if path.exists() and not force:
            skipped += 1
            continue
        log.info("historical_nba: fetching %s ...", s)
        payload = _live_fetch_season(s)
        if not payload:
            failed.append(s)
            continue
        _save_season_cache(cache_dir, s, payload)
        fetched += 1
        time.sleep(delay_s)
    return {"fetched": fetched, "skipped": skipped, "failed": failed, "seasons_total": len(seasons)}


# ---------------------------------------------------------------------------
# Corpus loader — what the similarity engine actually consumes.
# ---------------------------------------------------------------------------

def load_corpus(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    start_year: int = DEFAULT_START_YEAR,
    end_season: str = DEFAULT_END_SEASON,
    min_games: int = MIN_GAMES_HISTORICAL,
) -> list[HistoricalPlayerSeason]:
    """Walk every cached season file and return one big list of player-seasons.

    Silently skips missing season files (the corpus stays useful even
    if a few seasons are missing).
    """
    cache_dir = Path(cache_dir)
    end_year = int(end_season.split("-")[0])
    out: list[HistoricalPlayerSeason] = []
    for y in range(start_year, end_year + 1):
        s = season_string(y)
        payload = _load_season_cache(cache_dir, s)
        if payload is None:
            continue
        out.extend(parse_season_payload(payload, s, min_games=min_games))
    return out


def corpus_seasons_present(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> list[str]:
    """List the seasons we have on disk. For diagnostic output."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    seasons = []
    for p in sorted(cache_dir.glob("league_*.json")):
        name = p.stem.replace("league_", "")
        seasons.append(name)
    return seasons


# ---------------------------------------------------------------------------
# BaseSource shim so this module shows up in the source registry
# alongside DARKO / BBRef / etc. fetch() yields nothing — the data is
# loaded directly by the similarity engine.
# ---------------------------------------------------------------------------

class HistoricalNBA(BaseSource):
    slug = "historical_nba"
    name = "Historical NBA corpus (1980-present, LeagueDashPlayerStats)"
    category = "model"
    update_frequency = "yearly"
    tos_compliant = True
    # Default weight 0.0 because this source doesn't emit Ranking rows;
    # it backs the similarity engine. Kept in the registry for visibility.
    default_weight = 0.0
    homepage = "https://www.nba.com/stats/"
    notes = (
        "Every NBA player-season since 1980, pulled via "
        "nba_api.LeagueDashPlayerStats and cached as JSON under "
        "data/historical_nba/. Feeds the career-arc similarity engine "
        "(dynasty_bball.similarity). Does not emit Ranking rows."
    )

    CACHE_DIR = DEFAULT_CACHE_DIR
    START_YEAR = DEFAULT_START_YEAR
    END_SEASON = DEFAULT_END_SEASON

    def __init__(
        self,
        client=None,
        cache_dir: Optional[Path | str] = None,
        start_year: Optional[int] = None,
        end_season: Optional[str] = None,
    ):
        super().__init__(client=client)
        if cache_dir is not None:
            self.CACHE_DIR = Path(cache_dir)
        elif os.environ.get("DYNASTY_BBALL_HISTORICAL_CACHE_DIR"):
            self.CACHE_DIR = Path(os.environ["DYNASTY_BBALL_HISTORICAL_CACHE_DIR"])
        if start_year is not None:
            self.START_YEAR = start_year
        if end_season is not None:
            self.END_SEASON = end_season

    def fetch(self) -> Iterator[RankingRecord]:
        """No Ranking rows — this source feeds the similarity engine.

        If live backfill is requested via env var, we run it here so
        the launcher can warm the cache by invoking ``sync_source``.
        Otherwise we just verify the cache exists and log a summary.
        """
        live = os.environ.get("DYNASTY_BBALL_HISTORICAL_LIVE") == "1"
        if live:
            summary = backfill_seasons(
                start_year=self.START_YEAR,
                end_season=self.END_SEASON,
                cache_dir=self.CACHE_DIR,
            )
            log.info("historical_nba: backfill summary %s", summary)
        seasons = corpus_seasons_present(self.CACHE_DIR)
        log.info("historical_nba: %d seasons cached at %s", len(seasons), self.CACHE_DIR)
        return iter([])
