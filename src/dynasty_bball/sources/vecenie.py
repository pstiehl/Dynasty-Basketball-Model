"""Sam Vecenie Big Board adapter (CSV-only).

Sam Vecenie is The Athletic's lead NBA Draft analyst. His Big Board is
the de-facto industry consensus for *pre-NBA* prospect evaluation — the
NBA analog of Lance Zierlein for NFL — and his post-draft re-rankings
have a documented strong hit rate.

The Big Board itself is paywalled at The Athletic so we cannot scrape
it. This adapter mirrors the football repo's RAS/CFBD-Breakouts CSV
pattern: drop a CSV at ``data/vecenie/vecenie_big_board.csv`` and the
adapter will pick it up the next launcher run.

Expected schema (case-insensitive, flexible aliases per column):

  rank          required, integer, 1-based
  player_name   required, string                    (aliases: name, player, full_name)
  position      optional, one of PG/SG/SF/PF/C       (aliases: pos)
  tier          optional, integer 1–5
  notes         optional, free-form
  draft_year    optional, integer (e.g. 2025, 2026)

Records emit ``market_value = 0..100`` derived from rank inside the
file (rank 1 = 100.0, lower ranks taper linearly to 0 at rank == file
length). This puts Vecenie's signal on the same magnitude band as
Court Consensus and DARKO so the value-based scoring path picks it up
without leaning on rank-only normalization depth.

Vecenie is registered in ``weights.ROOKIE_SIGNAL_SOURCES`` — players
whose ONLY corroboration is the Big Board are filtered out of the top
of the dynasty composite (the "no-NBA-consensus" filter that prevents
hyped prospects from squatting on top spots before they've played).

Output format: ``points_dhk``. Vecenie's signal is format-agnostic
(it's about NBA talent, not category vs. points scoring), so we emit
the same record to both ``points_dhk`` and ``points_default`` on each
fetch.

If the CSV is missing — which it WILL be until someone (Phil) drops
one in — the adapter yields nothing and the launcher continues.
"""
from __future__ import annotations
import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .base import BaseSource, RankingRecord


log = logging.getLogger(__name__)


DEFAULT_CSV_PATH = Path("data/vecenie/vecenie_big_board.csv")

_HEADER_ALIASES = {
    "rank":        ("rank", "overall_rank", "big_board_rank"),
    "name":        ("player_name", "name", "player", "full_name"),
    "position":    ("position", "pos", "primary_position"),
    "tier":        ("tier",),
    "notes":       ("notes", "comment", "comments"),
    "draft_year":  ("draft_year", "year", "class"),
}


def _norm_key(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_").replace("-", "_")


def _pick(row: dict, aliases: tuple[str, ...]) -> Optional[str]:
    for k in aliases:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _to_int(v) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _norm_position(p) -> Optional[str]:
    if not p:
        return None
    s = str(p).strip().upper()
    if not s:
        return None
    # Accept multi-position strings like "PG/SG" — keep the primary.
    return s.split("/")[0].split(",")[0].strip() or None


def parse_vecenie_rows(
    rows: list[dict],
    captured_at: Optional[datetime] = None,
    league_format: str = "points_dhk",
) -> list[RankingRecord]:
    """Pure parser — turns dict rows into RankingRecords.

    Sorted by rank ascending so the scoring/value layer sees a clean
    1..N overall_rank sequence; market_value tapers from 100 down to 0.
    """
    captured_at = captured_at or datetime.utcnow()

    parsed: list[dict] = []
    for raw in rows:
        # Normalize keys for alias lookup.
        row = {_norm_key(k): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
        rank = _to_int(_pick(row, _HEADER_ALIASES["rank"]))
        name = _pick(row, _HEADER_ALIASES["name"])
        if rank is None or not name:
            continue
        parsed.append({
            "rank": rank,
            "name": str(name).strip(),
            "position": _norm_position(_pick(row, _HEADER_ALIASES["position"])),
            "tier": _to_int(_pick(row, _HEADER_ALIASES["tier"])),
            "draft_year": _to_int(_pick(row, _HEADER_ALIASES["draft_year"])),
        })

    if not parsed:
        return []

    parsed.sort(key=lambda r: r["rank"])
    n = len(parsed)
    max_rank = parsed[-1]["rank"] or n

    out: list[RankingRecord] = []
    # Per-position rank counters (within the Big Board universe).
    pos_counters: dict[str, int] = {}
    for p in parsed:
        pos = p["position"]
        pos_rank = None
        if pos:
            pos_counters[pos] = pos_counters.get(pos, 0) + 1
            pos_rank = pos_counters[pos]
        # Linear taper rank → value: rank=1 → 100, rank=max → 0.
        if max_rank <= 1:
            mv = 100.0
        else:
            mv = round(100.0 * (max_rank - p["rank"]) / (max_rank - 1), 3)
        out.append(RankingRecord(
            source_slug="vecenie",
            full_name=p["name"],
            position=pos,
            overall_rank=p["rank"],
            position_rank=pos_rank,
            market_value=mv,
            tier=p["tier"],
            league_format=league_format,
            is_dynasty=True,
            is_rookie_only=False,  # Vecenie's board covers veterans too in re-ranks
            draft_year=p["draft_year"],
            captured_at=captured_at,
        ))
    return out


class Vecenie(BaseSource):
    slug = "vecenie"
    name = "Sam Vecenie's Big Board (The Athletic)"
    category = "expert"
    update_frequency = "event"  # episodic, around draft cycles
    tos_compliant = True
    # 1.3 — elevated single-analyst weight. Vecenie has a documented
    # strong track record on NBA draft prospect translation. Matches
    # the weighting we'd give Lance Zierlein on the football side.
    default_weight = 1.3
    homepage = "https://theathletic.com/author/sam-vecenie/"
    notes = (
        "Single CSV drop required (data/vecenie/vecenie_big_board.csv). "
        "Paywalled source — no scrape possible. See data/vecenie/README.md "
        "for the file format."
    )

    CSV_PATH = DEFAULT_CSV_PATH

    def __init__(
        self,
        client=None,
        csv_path: Optional[Path | str] = None,
    ):
        super().__init__(client=client)
        if csv_path is not None:
            self.CSV_PATH = Path(csv_path)
        elif os.environ.get("DYNASTY_BBALL_VECENIE_CSV_PATH"):
            self.CSV_PATH = Path(os.environ["DYNASTY_BBALL_VECENIE_CSV_PATH"])

    def _read_rows(self) -> list[dict]:
        path = self.CSV_PATH
        if not path.exists():
            log.info("Vecenie: no CSV at %s — adapter yields nothing.", path)
            return []
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def fetch(self) -> Iterator[RankingRecord]:
        rows = self._read_rows()
        if not rows:
            return iter([])
        captured_at = datetime.utcnow()
        records = parse_vecenie_rows(
            rows, captured_at=captured_at, league_format="points_dhk"
        )
        records += parse_vecenie_rows(
            rows, captured_at=captured_at, league_format="points_default"
        )
        for r in records:
            yield r
