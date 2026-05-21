"""Refresh the cached NBA draft big-board.

Pulls 2008-2025 NBA draft history via ``nba_api`` and writes
``data/draft_stock/big_board.json``. This is the source of truth for
the rookie engine's draft-stock prior (PR #8).

Usage:

    DYNASTY_BBALL_DRAFT_LIVE=1 python scripts/refresh_draft_stock.py

The env-var gate keeps CI from hammering stats.nba.com.
"""
from __future__ import annotations

import logging
import sys

from dynasty_bball.sources.draft_stock import refresh_big_board


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prospects = refresh_big_board()
    if not prospects:
        print(
            "draft_stock refresh produced 0 prospects -- did you set "
            "DYNASTY_BBALL_DRAFT_LIVE=1 and have network?",
            file=sys.stderr,
        )
        return 1
    print(f"Wrote {len(prospects)} prospects to data/draft_stock/big_board.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
