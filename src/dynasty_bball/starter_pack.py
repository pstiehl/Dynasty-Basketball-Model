"""Minimal placeholder starter pack.

In Dynasty-Football-Model this module holds transcribed top-N rookie lists
from major analysts. For the basketball repo's PR #1 we keep it empty —
DARKO covers the active player pool, and the rookie-focused public lists
(Sam Vecenie, B/R Top 100, Lance Stephenson Big Board) land as proper
adapters in subsequent PRs.

The function is still exported so ``launcher_headless`` can call it
without conditional branching. Returning 0 is fine.
"""
from __future__ import annotations


STARTER_PACK: list[dict] = []


def import_starter_pack() -> int:
    """No-op in PR #1. Returns 0."""
    return 0
