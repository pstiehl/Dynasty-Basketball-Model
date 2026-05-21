"""Base source adapter — every ranking source implements this interface.

Adapters return normalized `RankingRecord` objects. The sync layer takes care of
resolving each record to a canonical Player row and writing to the DB.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Iterator
import httpx
from ..config import settings


@dataclass
class RankingRecord:
    """Normalized ranking record produced by a source adapter.

    Mirrors Dynasty-Football-Model's RankingRecord but swaps the football
    fields for basketball-specific ones (DARKO DPM, longevity, per-100
    counting stats). Most sources will set only a handful of these.
    """
    source_slug: str
    sleeper_id: Optional[str] = None        # canonical ID — preferred
    nba_id: Optional[str] = None
    bbref_id: Optional[str] = None
    full_name: str = ""                      # fallback for name matching
    position: Optional[str] = None
    overall_rank: Optional[int] = None
    position_rank: Optional[int] = None
    market_value: Optional[float] = None     # for value-based sources (KTC-style)
    tier: Optional[int] = None
    trend_30d: Optional[float] = None
    league_format: str = "points_dhk"        # points_dhk | points_default | 9cat | rookie
    is_dynasty: bool = True
    is_rookie_only: bool = False
    captured_at: datetime = field(default_factory=datetime.utcnow)

    # Optional Player enrichment — sync.py will set these on the resolved Player.
    nba_team: Optional[str] = None
    age: Optional[float] = None
    years_exp: Optional[int] = None
    draft_year: Optional[int] = None
    draft_round: Optional[int] = None
    draft_pick_overall: Optional[int] = None
    draft_team: Optional[str] = None
    college: Optional[str] = None
    est_retirement_age: Optional[float] = None
    years_remaining: Optional[float] = None

    # Basketball stat carriers (per-game or per-100, depending on the source).
    per_game_points: Optional[float] = None
    per_game_rebounds: Optional[float] = None
    per_game_assists: Optional[float] = None
    per_game_steals: Optional[float] = None
    per_game_blocks: Optional[float] = None
    per_game_threes: Optional[float] = None
    per_game_turnovers: Optional[float] = None

    # DARKO-style impact metrics.
    dpm: Optional[float] = None
    dpm_improvement: Optional[float] = None
    o_dpm: Optional[float] = None
    d_dpm: Optional[float] = None


class BaseSource(ABC):
    """Abstract adapter for a ranking source.

    Subclasses must set the class attributes and implement `fetch()`.
    """
    slug: str = ""
    name: str = ""
    category: str = "expert"          # market | expert | model | aggregator
    update_frequency: str = "daily"    # daily | weekly | event
    tos_compliant: bool = True
    default_weight: float = 1.0
    homepage: str = ""
    notes: str = ""

    def __init__(self, client: httpx.Client | None = None):
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds,
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
        )

    @abstractmethod
    def fetch(self) -> Iterator[RankingRecord]:
        """Fetch current rankings from the source, yielding RankingRecords."""
        ...

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
