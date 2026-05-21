#!/usr/bin/env python3
"""One-shot backfill of the historical NBA corpus.

Pulls LeagueDashPlayerStats for every season from 1980-81 to the
configured end season and caches each as JSON under
``data/historical_nba/league_<season>.json``.

This is intentionally NOT a CI step — running it from CI would hammer
stats.nba.com and trigger rate-limit bans. Run it once locally, commit
the data/historical_nba/ directory, and let CI use the cache.

Usage::

    python scripts/backfill_historical_nba.py
    python scripts/backfill_historical_nba.py --force                   # overwrite cache
    python scripts/backfill_historical_nba.py --start 2010 --end 2024-25
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

# Make src/ importable when run from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dynasty_bball.sources.historical_nba import (
    backfill_seasons,
    DEFAULT_START_YEAR,
    DEFAULT_END_SEASON,
    DEFAULT_CACHE_DIR,
    HISTORICAL_REQUEST_DELAY_S,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--start", type=int, default=DEFAULT_START_YEAR,
        help=f"Start year, e.g. 1980 (default: {DEFAULT_START_YEAR})",
    )
    parser.add_argument(
        "--end", type=str, default=DEFAULT_END_SEASON,
        help=f"End season string, e.g. 2024-25 (default: {DEFAULT_END_SEASON})",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR),
        help="Cache directory (default: data/historical_nba)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even seasons that already have a cache file",
    )
    parser.add_argument(
        "--delay", type=float, default=HISTORICAL_REQUEST_DELAY_S,
        help="Seconds to sleep between season calls (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("backfill")

    log.info(
        "Starting historical backfill: %d through %s (force=%s, delay=%.1fs)",
        args.start, args.end, args.force, args.delay,
    )
    summary = backfill_seasons(
        start_year=args.start,
        end_season=args.end,
        cache_dir=Path(args.cache_dir),
        force=args.force,
        delay_s=args.delay,
    )
    log.info(
        "Done. fetched=%d skipped=%d failed=%d (of %d seasons)",
        summary["fetched"], summary["skipped"], len(summary["failed"]),
        summary["seasons_total"],
    )
    if summary["failed"]:
        log.warning("Failed seasons: %s", ", ".join(summary["failed"]))
        sys.exit(2)


if __name__ == "__main__":
    main()
