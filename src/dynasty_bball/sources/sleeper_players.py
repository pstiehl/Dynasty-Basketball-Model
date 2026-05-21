"""Sleeper NBA adapter — used for canonical player ID + metadata resolution.

The Sleeper NBA player endpoint is the spine that links all sources. It
returns a dict keyed by sleeper_id with fields including position, team,
age, years_exp, fantasy_positions, birth_date, college.

Docs: https://docs.sleeper.com/  (use sport=nba)

This endpoint is ~few-MB and changes infrequently — run it weekly, not daily.
"""
from __future__ import annotations
from typing import Iterator
from .base import BaseSource, RankingRecord


class SleeperPlayers(BaseSource):
    slug = "sleeper_players"
    name = "Sleeper — NBA player metadata"
    category = "aggregator"
    update_frequency = "weekly"
    tos_compliant = True
    homepage = "https://docs.sleeper.com/"
    notes = "Used to build the canonical NBA player ID map. Does NOT provide rankings."

    URL = "https://api.sleeper.app/v1/players/nba"

    def fetch(self) -> Iterator[RankingRecord]:
        # Sleeper doesn't expose rankings; nothing to yield as a RankingRecord.
        return iter([])

    def fetch_players_dict(self) -> dict:
        """Returns the full Sleeper NBA player dict keyed by sleeper_id."""
        resp = self._client.get(self.URL)
        resp.raise_for_status()
        return resp.json()
