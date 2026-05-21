"""Backfill the NCAA D1 historical corpus from barttorvik.

One-shot script — run locally, commit the resulting JSON files under
``data/historical_ncaa/``. CI never hits external sites.

Usage::

    python3 scripts/backfill_historical_ncaa.py            # fill missing
    python3 scripts/backfill_historical_ncaa.py --force    # refetch all
    DYNASTY_BBALL_NCAA_END_YEAR=2024 python3 scripts/backfill_historical_ncaa.py

Rate-limited at 1.0s between season calls. Total runtime ~20-30s for
17 seasons. ~36 MB on disk before pruning, ~25 MB compressed in git.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Make the src/ tree importable without an install step.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dynasty_bball.sources.historical_ncaa import (
    backfill_seasons,
    DEFAULT_START_YEAR,
    DEFAULT_END_YEAR,
    DEFAULT_CACHE_DIR,
)


def main():
    ap = argparse.ArgumentParser(description="Backfill the NCAA historical corpus.")
    ap.add_argument("--start", type=int, default=DEFAULT_START_YEAR,
                    help=f"Start season-end-year (default {DEFAULT_START_YEAR})")
    ap.add_argument("--end", type=int, default=DEFAULT_END_YEAR,
                    help=f"End season-end-year (default {DEFAULT_END_YEAR})")
    ap.add_argument("--force", action="store_true",
                    help="Refetch seasons even if cache exists.")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Backfilling NCAA D1 corpus {args.start}-{args.end} → {args.cache_dir}")
    summary = backfill_seasons(
        start_year=args.start,
        end_year=args.end,
        cache_dir=Path(args.cache_dir),
        force=args.force,
    )
    print("Summary:", summary)
    if summary["failed"]:
        print(f"WARNING: {len(summary['failed'])} seasons failed: {summary['failed']}")
        sys.exit(2)


if __name__ == "__main__":
    main()
