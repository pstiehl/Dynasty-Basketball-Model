"""Source adapter registry — list every source class here.

Order is irrelevant for sync (the launcher iterates registered slugs), but
imports happen at module load time so any new source must be importable.
"""
from typing import Type
from .base import BaseSource, RankingRecord
from .darko import DARKO
from .court_consensus import CourtConsensus
from .vecenie import Vecenie
from .sleeper_players import SleeperPlayers


REGISTRY: dict[str, Type[BaseSource]] = {
    cls.slug: cls
    for cls in [
        DARKO,
        CourtConsensus,
        Vecenie,
        SleeperPlayers,
    ]
}

__all__ = [
    "REGISTRY", "BaseSource", "RankingRecord",
    "DARKO", "CourtConsensus", "Vecenie", "SleeperPlayers",
]
