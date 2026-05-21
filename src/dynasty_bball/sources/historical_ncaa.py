"""Historical NCAA corpus — every NCAA Division I player-season since 2008.

Purpose
-------
This source does NOT contribute rankings directly. Like historical_nba.py,
it exists to feed ``dynasty_bball.similarity``. PR #7's rookie chain
needs ground-truth NCAA careers so the engine can answer:

    "Cooper Flagg's 2024-25 Duke freshman season looks like Carmelo
     Anthony's 2002-03 Syracuse freshman season; Anthony went on to
     a 19-year NBA career — Flagg's rookie dynasty value should
     reflect that."

Data source
-----------
We pull from **barttorvik.com** (NCAA D1 advanced stats — public JSON
endpoint, no anti-bot defenses, no scraping required). One round-trip
per season fetches every D1 player's per-game and advanced production
in a single ~2 MB JSON dump.

This was a deliberate substitution. sports-reference.com/cbb (originally
spec'd in the PR brief) returns 403 from CI / cloud IPs due to
Cloudflare-style fingerprinting. Barttorvik:

  * Same coverage (every D1 player back to 2008)
  * Better fields (advanced metrics: USG, TS%, BPM-equivalent are
    already computed)
  * One JSON call per season instead of N HTML scrapes per player
  * Public, friendly to bots — Bart explicitly publishes this endpoint
  * 18 years of data (2008-2025) → ~60K-85K player-seasons after
    filtering. That spans every NBA rookie since 2009.

Pre-2008 NCAA stats are not available from this endpoint (the API
returns a fallback payload for years before its dataset starts). The
NBA corpus goes back to 1980 but the rookie chain meaningfully
requires both endpoints — so the bridge naturally limits to 2008+
NBA-bound college players. The handful of pre-2008 NBA stars who
won't have a college season in the corpus (Kobe, KG, LeBron, the
2000s draft class) get NBA-only similarity, which is the existing
PR #4 behavior — no regression.

Caching
-------
Each season's payload is cached under
``data/historical_ncaa/season_<year>.json``. Live refreshes are gated
behind ``DYNASTY_BBALL_NCAA_LIVE=1`` (or absent cache file). The
cache IS committed to the repo so CI never hits external sites.

Column map (barttorvik getadvstats.php)
---------------------------------------
The endpoint returns a JSON array of arrays — no field names. We
decoded the columns by inspection against known players. The mapping
lives in ``BTV_COL`` below. The most important fields for our
vectorization:

  * [0] player, [1] team, [2] conf, [3] GP, [4] min%, [6] usg
  * [8] TS%, [11] AST%, [12] TO%, [22] BLK%, [23] STL%
  * [25] class (Fr/So/Jr/Sr), [26] height, [29] adjOE, [32] btv_pid
  * [50] BPM, [54] MPG
  * [57] OREB/G, [58] DREB/G, [59] TRB/G, [60] AST/G, [61] STL/G,
    [62] BLK/G, [63] PTS/G, [64] position role

We also derive FGA/G, 3PA/G, FTA/G from the totals in cols [13]-[21].

Conference strength
-------------------
Conferences are bucketed into four tiers for the strength multiplier
in vectorize:

  * P5 — ACC, B10, B12, SEC, P12, BE (Power Five + Big East)
  * HM — A10, AAC, MWC, WCC (high-major mid)
  * MM — Sun Belt, MAC, Horizon, Ivy, etc. (mid-major)
  * LM — small conferences (low-major)

The bucket map is in ``CONFERENCE_TIER``. The multiplier is applied
in ``similarity.vectorize.vectorize_college_season``.
"""
from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional
from urllib import request as _urlrequest

from .base import BaseSource, RankingRecord


log = logging.getLogger(__name__)


# Barttorvik data starts here. 2008+ covers every NBA rookie since 2009-10.
DEFAULT_START_YEAR = 2008
# Current NCAA "season year" — barttorvik uses end-year (2024-25 → 2025).
DEFAULT_END_YEAR = int(os.environ.get("DYNASTY_BBALL_NCAA_END_YEAR", "2025"))

DEFAULT_CACHE_DIR = Path("data/historical_ncaa")

# 3-second rate limit was specced for sports-reference. Barttorvik is
# much friendlier — 1s is fine. Still polite.
NCAA_REQUEST_DELAY_S = float(os.environ.get("DYNASTY_BBALL_NCAA_DELAY", "1.0"))

BTV_URL = "https://barttorvik.com/getadvstats.php?year={year}&t=All&conlimit=All&top=0&revquery="
BTV_USER_AGENT = "dynasty-basketball-model/0.6 (NCAA corpus for fantasy similarity research; +https://github.com/sandpaw-ai/Dynasty-Basketball-Model)"

# Filters for corpus inclusion — match the NBA corpus's MIN_GAMES and
# MPG floor so per-36 explosions don't pollute the similarity space.
MIN_GAMES_NCAA = 10
MIN_MPG_NCAA = 10.0


# ---------------------------------------------------------------------------
# Conference strength buckets
# ---------------------------------------------------------------------------

CONFERENCE_TIER: dict[str, str] = {
    # Power Five + Big East (now considered "high-major" peer of P5)
    "ACC": "P5", "B10": "P5", "B12": "P5", "SEC": "P5",
    "P12": "P5", "BE": "P5", "P10": "P5",
    # High-major mid
    "A10": "HM", "AAC": "HM", "MWC": "HM", "WCC": "HM",
    "Amer": "HM",
    # Mid-major
    "MVC": "MM", "CUSA": "MM", "Sum": "MM", "WAC": "MM",
    "OVC": "MM", "Horz": "MM", "MAC": "MM", "SB": "MM",
    "BSth": "MM", "BSky": "MM", "Pat": "MM", "CAA": "MM",
    "Slnd": "MM", "Sou": "MM", "ASun": "MM",
    # Low-major (everyone else)
}
TIER_MULTIPLIER: dict[str, float] = {
    "P5": 1.00,
    "HM": 0.92,
    "MM": 0.83,
    "LM": 0.75,
}


def conference_tier(conf: Optional[str]) -> str:
    """Map a conference code (or None) to a tier label."""
    if not conf:
        return "LM"
    return CONFERENCE_TIER.get(conf, "LM")


def conference_strength_multiplier(conf: Optional[str]) -> float:
    return TIER_MULTIPLIER[conference_tier(conf)]


# ---------------------------------------------------------------------------
# Barttorvik column indices (see module docstring).
# ---------------------------------------------------------------------------

class BTV_COL:
    PLAYER = 0
    TEAM = 1
    CONF = 2
    GP = 3
    MIN_PCT = 4
    ORTG = 5
    USG = 6
    EFG = 7
    TS = 8
    ORB_PCT = 9
    DRB_PCT = 10
    AST_PCT = 11
    TO_PCT = 12
    FTM = 13       # totals
    FTA = 14       # totals
    FT_PCT = 15
    TWOP_M = 16    # totals
    TWOP_A = 17    # totals
    TWOP_PCT = 18
    THREEP_M = 19  # totals
    THREEP_A = 20  # totals
    THREEP_PCT = 21
    BLK_PCT = 22
    STL_PCT = 23
    FTR = 24
    CLASS = 25
    HEIGHT = 26
    NUM = 27
    PORPAG = 28
    ADJOE = 29
    PFR = 30
    YEAR = 31
    PID = 32
    HOMETOWN = 33
    REC_RANK = 34
    BPM = 50
    MPG = 54
    OREB_PG = 57
    DREB_PG = 58
    TRB_PG = 59
    AST_PG = 60
    STL_PG = 61
    BLK_PG = 62
    PTS_PG = 63
    POSITION_ROLE = 64
    BIRTHDATE = 66


# ---------------------------------------------------------------------------
# Dataclass — one NCAA player-season.
# ---------------------------------------------------------------------------

@dataclass
class HistoricalNCAASeason:
    """One NCAA D1 player's box-score line for a single season.

    Field naming mirrors HistoricalPlayerSeason where possible so the
    similarity engine can vectorize them in a near-identical code path.
    """
    sr_player_id: str           # btv pid (unique across years for the same player)
    name: str
    season: str                 # e.g. "2024-25"
    season_end_year: int        # 2025 for "2024-25"
    school: str                 # team
    conference: str             # raw conf code (ACC, B10, etc.)
    class_year: Optional[str]   # Fr/So/Jr/Sr/Gr/None
    age_at_season: Optional[float]  # derived from birthdate when present
    height: Optional[str]
    position_role: Optional[str]  # btv role string ("Combo G", "Stretch 4", ...)
    gp: int
    mpg: float
    pts_pg: float
    reb_pg: float
    oreb_pg: float
    dreb_pg: float
    ast_pg: float
    stl_pg: float
    blk_pg: float
    fgm_pg: float
    fga_pg: float
    tpa_pg: float
    tpm_pg: float
    fta_pg: float
    ftm_pg: float
    fg_pct: float
    ft_pct: float
    ts_pct: float
    efg_pct: float
    usg_pct: float
    ast_pct: float
    to_pct: float
    blk_pct: float
    stl_pct: float
    bpm: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HistoricalNCAASeason":
        return cls(**d)


# ---------------------------------------------------------------------------
# Season string helpers
# ---------------------------------------------------------------------------

def season_string_from_end_year(end_year: int) -> str:
    """2025 → "2024-25"."""
    start = end_year - 1
    end_two = end_year % 100
    return f"{start}-{end_two:02d}"


# ---------------------------------------------------------------------------
# Parsing — convert one barttorvik payload to HistoricalNCAASeason rows.
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


def _derive_age(birthdate: Optional[str], season_end_year: int) -> Optional[float]:
    """Derive a player's age at the START of the season from birthdate.

    Season conventionally starts in November (end_year - 1, month 11).
    Returns None if birthdate is malformed.
    """
    if not birthdate or not isinstance(birthdate, str):
        return None
    try:
        from datetime import date
        y, m, d = birthdate.split("-")
        bd = date(int(y), int(m), int(d))
        season_start = date(season_end_year - 1, 11, 1)
        delta_days = (season_start - bd).days
        return round(delta_days / 365.25, 1)
    except Exception:
        return None


def parse_barttorvik_payload(
    payload: list,
    season_end_year: int,
    min_games: int = MIN_GAMES_NCAA,
    min_mpg: float = MIN_MPG_NCAA,
) -> list[HistoricalNCAASeason]:
    """Convert a barttorvik getadvstats.php list payload into season rows.

    Filters: GP ≥ min_games, MPG ≥ min_mpg. Drops rows with missing
    name/team/pid.
    """
    if not isinstance(payload, list):
        return []
    season_str = season_string_from_end_year(season_end_year)
    out: list[HistoricalNCAASeason] = []
    for row in payload:
        try:
            name = row[BTV_COL.PLAYER]
            team = row[BTV_COL.TEAM]
            pid = row[BTV_COL.PID]
            if not name or not team or pid is None:
                continue
            gp = _to_int(row[BTV_COL.GP])
            if gp is None or gp < min_games:
                continue
            mpg = _to_float(row[BTV_COL.MPG]) or 0.0
            if mpg < min_mpg:
                continue
            # Year sanity — barttorvik returns the latest dataset year
            # for out-of-range queries, so make sure the row's own year
            # matches what we asked for.
            row_year = _to_int(row[BTV_COL.YEAR])
            if row_year is not None and row_year != season_end_year:
                continue

            ftm_total = _to_float(row[BTV_COL.FTM]) or 0.0
            fta_total = _to_float(row[BTV_COL.FTA]) or 0.0
            twom_total = _to_float(row[BTV_COL.TWOP_M]) or 0.0
            twoa_total = _to_float(row[BTV_COL.TWOP_A]) or 0.0
            tpm_total = _to_float(row[BTV_COL.THREEP_M]) or 0.0
            tpa_total = _to_float(row[BTV_COL.THREEP_A]) or 0.0
            fgm_total = twom_total + tpm_total
            fga_total = twoa_total + tpa_total
            fg_pct = (fgm_total / fga_total) if fga_total > 0 else 0.0

            out.append(HistoricalNCAASeason(
                sr_player_id=str(pid),
                name=str(name).strip(),
                season=season_str,
                season_end_year=season_end_year,
                school=str(team).strip(),
                conference=str(row[BTV_COL.CONF] or "").strip(),
                class_year=str(row[BTV_COL.CLASS] or "").strip() or None,
                age_at_season=_derive_age(row[BTV_COL.BIRTHDATE], season_end_year),
                height=(str(row[BTV_COL.HEIGHT]).strip() if row[BTV_COL.HEIGHT] else None),
                position_role=(str(row[BTV_COL.POSITION_ROLE]).strip() if row[BTV_COL.POSITION_ROLE] else None),
                gp=gp,
                mpg=mpg,
                pts_pg=_to_float(row[BTV_COL.PTS_PG]) or 0.0,
                reb_pg=_to_float(row[BTV_COL.TRB_PG]) or 0.0,
                oreb_pg=_to_float(row[BTV_COL.OREB_PG]) or 0.0,
                dreb_pg=_to_float(row[BTV_COL.DREB_PG]) or 0.0,
                ast_pg=_to_float(row[BTV_COL.AST_PG]) or 0.0,
                stl_pg=_to_float(row[BTV_COL.STL_PG]) or 0.0,
                blk_pg=_to_float(row[BTV_COL.BLK_PG]) or 0.0,
                fgm_pg=fgm_total / gp,
                fga_pg=fga_total / gp,
                tpa_pg=tpa_total / gp,
                tpm_pg=tpm_total / gp,
                fta_pg=fta_total / gp,
                ftm_pg=ftm_total / gp,
                fg_pct=fg_pct,
                ft_pct=_to_float(row[BTV_COL.FT_PCT]) or 0.0,
                ts_pct=(_to_float(row[BTV_COL.TS]) or 0.0) / 100.0,
                efg_pct=(_to_float(row[BTV_COL.EFG]) or 0.0) / 100.0,
                usg_pct=_to_float(row[BTV_COL.USG]) or 0.0,
                ast_pct=_to_float(row[BTV_COL.AST_PCT]) or 0.0,
                to_pct=_to_float(row[BTV_COL.TO_PCT]) or 0.0,
                blk_pct=_to_float(row[BTV_COL.BLK_PCT]) or 0.0,
                stl_pct=_to_float(row[BTV_COL.STL_PCT]) or 0.0,
                bpm=_to_float(row[BTV_COL.BPM]),
            ))
        except (IndexError, KeyError, TypeError, ValueError) as e:
            log.debug("historical_ncaa: skipping malformed row: %s", e)
            continue
    return out


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, season_end_year: int) -> Path:
    return cache_dir / f"season_{season_end_year}.json"


def _load_season_cache(cache_dir: Path, season_end_year: int) -> Optional[list]:
    path = _cache_path(cache_dir, season_end_year)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("historical_ncaa: cache %s unreadable: %s", path, e)
        return None


def _save_season_cache(cache_dir: Path, season_end_year: int, payload: list) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, season_end_year)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
    except Exception as e:
        log.warning("historical_ncaa: failed to write cache %s: %s", path, e)


def _live_fetch_season(season_end_year: int, timeout: float = 60.0) -> Optional[list]:
    """Pull one season's full D1 advanced stats from barttorvik.

    Returns None on failure (network error, non-200, malformed JSON).
    """
    url = BTV_URL.format(year=season_end_year)
    req = _urlrequest.Request(url, headers={
        "User-Agent": BTV_USER_AGENT,
        "Accept": "application/json,text/javascript,*/*",
    })
    try:
        with _urlrequest.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                log.warning("historical_ncaa: %s returned HTTP %s", url, resp.status)
                return None
            body = resp.read().decode("utf-8")
    except Exception as e:
        log.warning("historical_ncaa: live fetch failed for %d: %s", season_end_year, e)
        return None
    try:
        data = json.loads(body)
    except Exception as e:
        log.warning("historical_ncaa: malformed JSON for %d: %s", season_end_year, e)
        return None
    if not isinstance(data, list):
        log.warning("historical_ncaa: unexpected shape for %d (got %s)", season_end_year, type(data).__name__)
        return None
    return data


# ---------------------------------------------------------------------------
# Backfill — one-shot.
# ---------------------------------------------------------------------------

def backfill_seasons(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force: bool = False,
    delay_s: float = NCAA_REQUEST_DELAY_S,
) -> dict:
    """Populate the NCAA cache from barttorvik.

    Returns ``{"fetched": N, "skipped": M, "failed": [years]}``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fetched, skipped, failed = 0, 0, []
    for y in range(start_year, end_year + 1):
        path = _cache_path(cache_dir, y)
        if path.exists() and not force:
            skipped += 1
            continue
        log.info("historical_ncaa: fetching %d ...", y)
        payload = _live_fetch_season(y)
        if not payload:
            failed.append(y)
            continue
        _save_season_cache(cache_dir, y, payload)
        fetched += 1
        time.sleep(delay_s)
    return {"fetched": fetched, "skipped": skipped, "failed": failed,
            "seasons_total": end_year - start_year + 1}


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def load_corpus(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    min_games: int = MIN_GAMES_NCAA,
    min_mpg: float = MIN_MPG_NCAA,
) -> list[HistoricalNCAASeason]:
    """Walk cached season files and return all NCAA player-seasons."""
    cache_dir = Path(cache_dir)
    out: list[HistoricalNCAASeason] = []
    for y in range(start_year, end_year + 1):
        payload = _load_season_cache(cache_dir, y)
        if payload is None:
            continue
        out.extend(parse_barttorvik_payload(
            payload, season_end_year=y,
            min_games=min_games, min_mpg=min_mpg,
        ))
    return out


def corpus_seasons_present(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> list[int]:
    """List the season-end-years cached on disk."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    out = []
    for p in sorted(cache_dir.glob("season_*.json")):
        try:
            out.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# BaseSource shim — registry parity, no Ranking rows emitted.
# ---------------------------------------------------------------------------

class HistoricalNCAA(BaseSource):
    slug = "historical_ncaa"
    name = "Historical NCAA D1 corpus (2008-present, barttorvik)"
    category = "model"
    update_frequency = "yearly"
    tos_compliant = True
    default_weight = 0.0
    homepage = "https://barttorvik.com/"
    notes = (
        "Every NCAA D1 player-season since 2008, pulled from "
        "barttorvik.com/getadvstats.php and cached as JSON under "
        "data/historical_ncaa/. Feeds the rookie college→NBA "
        "similarity chain in the career-arc engine. Does not emit "
        "Ranking rows."
    )

    CACHE_DIR = DEFAULT_CACHE_DIR
    START_YEAR = DEFAULT_START_YEAR
    END_YEAR = DEFAULT_END_YEAR

    def __init__(
        self,
        client=None,
        cache_dir: Optional[Path | str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ):
        super().__init__(client=client)
        if cache_dir is not None:
            self.CACHE_DIR = Path(cache_dir)
        elif os.environ.get("DYNASTY_BBALL_NCAA_CACHE_DIR"):
            self.CACHE_DIR = Path(os.environ["DYNASTY_BBALL_NCAA_CACHE_DIR"])
        if start_year is not None:
            self.START_YEAR = start_year
        if end_year is not None:
            self.END_YEAR = end_year

    def fetch(self) -> Iterator[RankingRecord]:
        """No Ranking rows — feeds the rookie similarity engine.

        Backfill only runs when ``DYNASTY_BBALL_NCAA_LIVE=1`` is set,
        protecting CI from external calls. Otherwise we verify the
        cache exists and log a summary.
        """
        live = os.environ.get("DYNASTY_BBALL_NCAA_LIVE") == "1"
        if live:
            summary = backfill_seasons(
                start_year=self.START_YEAR,
                end_year=self.END_YEAR,
                cache_dir=self.CACHE_DIR,
            )
            log.info("historical_ncaa: backfill summary %s", summary)
        seasons = corpus_seasons_present(self.CACHE_DIR)
        log.info("historical_ncaa: %d seasons cached at %s", len(seasons), self.CACHE_DIR)
        return iter([])
