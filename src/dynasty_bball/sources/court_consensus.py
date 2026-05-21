"""Court Consensus source adapter.

Court Consensus (https://courtconsensus.com/) publishes crowdsourced
dynasty fantasy basketball rankings derived from head-to-head community
voting (ELO model, 76k+ data points and growing). Two value columns:

* ``elo_rating_points``     — ELO score for points-league formats
* ``elo_rating_categories`` — ELO score for category-league formats

We treat Court Consensus as a **market / consensus** source. It is the
NBA-side analog of FantasyCalc on the football side — community-derived
trade value, not an expert ranker.

Why it matters
--------------
DARKO alone over-rewards rookies whose composite leans on the longevity
bonus (DARKO yrs-remaining × 2.5). Phil flagged Kon Knueppel, Derik
Queen, and Donovan Clingan as visibly too high after PR #1. Court
Consensus is the consensus anchor that sands that down: in CC's table
those same rookies live in the #30–60 range, not the top 15.

Access — three-tiered fallback
------------------------------
The site is a Vite/React SPA backed by Supabase. Tier 1 reads the same
public Supabase REST endpoint the SPA itself uses; tiers 2 and 3 keep
the launcher resilient if anything changes upstream.

  Tier 1 — Supabase REST API (live).
    Anon key is publicly embedded in the site's JS bundle (gated by
    RLS for write/vote operations; reads are public). We pull all
    342-ish rows in one request with ``select=…&order=…`` and no
    further calls. One round-trip per launcher run, with a polite UA
    identifying us. See ToS §4 (no automated voting — we don't vote)
    and §5 (rate limiting — we make a single read).

  Tier 2 — HTML fallback.
    Hook reserved for future use if Supabase access ever closes off.
    Today the page is a JS-only SPA so HTML parsing yields nothing;
    rather than ship dead code, this tier is a stub.

  Tier 3 — Local CSV at ``data/court_consensus/court_consensus_dump.csv``.
    Mirrors the DARKO CSV-fallback pattern. Columns (case-insensitive,
    flexible aliases): rank, name (or player), team, age, position,
    elo_points (or value), elo_categories.

Output
------
One ``RankingRecord`` per real NBA player (PICK rows filtered). The
``market_value`` is the ELO rescaled 0–100 so it lives in the same
band the scoring layer normalizes against. The default scoring format
is ``points_dhk``; a duplicate row for ``points_default`` is emitted
because Phil's DHK league is points-league-flavored and CC's
``elo_rating_points`` is the right signal for both.

Attribution: rankings copyright Court Consensus. Site links back to
https://courtconsensus.com on the published rankings page.
"""
from __future__ import annotations
import csv
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import httpx

from .base import BaseSource, RankingRecord


log = logging.getLogger(__name__)


# Public Supabase project URL + anon key for courtconsensus.com. Both are
# in the site's JS bundle and are intended for unauthenticated reads. We
# hold a copy here so the launcher works even if their JS bundle URL
# changes; if either ever rotates we fall through to the CSV.
COURT_CONSENSUS_SUPABASE_URL = (
    "https://viayfcilcpoprzucmztv.supabase.co"
)
COURT_CONSENSUS_SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZpYXlmY2lsY3BvcHJ6dWNtenR2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk5NzIyOTUsImV4cCI6MjA3NTU0ODI5NX0."
    "O-SfvSlln2WX_HR0GbZC1jzuI9GlxIAS3OPJY7gMWTc"
)
# Path to refresh the key from in case it ever rotates — we sniff it out
# of the JS bundle the same way an honest contributor would, never
# scraping anything beyond what the browser already fetches.
COURT_CONSENSUS_BUNDLE_DISCOVERY_URL = "https://courtconsensus.com/rankings"

DEFAULT_CSV_PATH = Path("data/court_consensus/court_consensus_dump.csv")

# How many rows to ask for. CC has ~340 rows total today; 1000 gives us
# plenty of headroom without paging.
PAGE_SIZE = 1000


def _normalize_position(pos_raw) -> Optional[str]:
    """Extract a single primary position from CC's per-player position list.

    CC stores ``position`` as a JSONB array (e.g. ``["C"]``, ``["PG","SG"]``).
    We keep the first entry — the scoring/site layers expect a single
    position string.
    """
    if pos_raw is None:
        return None
    if isinstance(pos_raw, list):
        for p in pos_raw:
            if p:
                return str(p).upper()
        return None
    s = str(pos_raw).strip()
    if not s:
        return None
    # If it came back as the literal string '["C","PF"]', salvage the first.
    m = re.match(r"\[?\s*\"?([A-Z]+)\"?", s)
    if m:
        return m.group(1).upper()
    return s.upper()


def _is_real_player(row: dict) -> bool:
    """Filter out PICK rows ("2026 Pick 1.01", etc.)."""
    name = (row.get("name") or "").strip()
    if not name:
        return False
    pos = _normalize_position(row.get("position"))
    if pos == "PICK":
        return False
    # Defensive: literal "Pick" in the name even with a real position.
    if re.match(r"\d{4}\s+(Early|Mid|Late|Pick)\b", name, re.IGNORECASE):
        return False
    return True


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


# ---------------------------------------------------------------------------
# Row → RankingRecord (pure; trivially testable from fixtures)
# ---------------------------------------------------------------------------

def parse_court_consensus_rows(
    rows: list[dict],
    captured_at: Optional[datetime] = None,
    league_format: str = "points_dhk",
) -> list[RankingRecord]:
    """Parse Supabase ``players`` rows into RankingRecords.

    Filters out PICK rows, normalizes positions, computes per-position
    ranks within the filtered universe, rescales ELO 0–9999 down to
    0–100 so the scoring layer's value-based path picks it up at the
    right magnitude.
    """
    captured_at = captured_at or datetime.utcnow()

    real = [r for r in rows if _is_real_player(r)]

    # Sort by elo_rating_points desc (the points-league signal — the
    # default-format signal for Phil's league).
    def _elo(r):
        return _to_float(r.get("elo_rating_points")) or 0.0

    real.sort(key=_elo, reverse=True)

    # max ELO for 0–100 rescale. CC's top is pinned at 9999 today but
    # we don't assume that.
    max_elo = max((_elo(r) for r in real), default=0.0) or 1.0

    # Per-position ranking within the filtered universe (no PICK rows).
    pos_counter: dict[str, int] = {}

    out: list[RankingRecord] = []
    for rank, r in enumerate(real, start=1):
        pos = _normalize_position(r.get("position"))
        if pos:
            pos_counter[pos] = pos_counter.get(pos, 0) + 1
            pos_rank = pos_counter[pos]
        else:
            pos_rank = None

        elo_pts = _to_float(r.get("elo_rating_points"))
        market_value = round(100.0 * (elo_pts or 0.0) / max_elo, 3) if elo_pts is not None else None

        out.append(RankingRecord(
            source_slug="court_consensus",
            full_name=(r.get("name") or "").strip(),
            position=pos,
            nba_team=(r.get("team") or None),
            age=_to_float(r.get("age")),
            years_exp=_to_int(r.get("years_exp")),
            overall_rank=rank,
            position_rank=pos_rank,
            market_value=market_value,
            league_format=league_format,
            is_dynasty=True,
            captured_at=captured_at,
        ))
    return out


# ---------------------------------------------------------------------------
# Network: Supabase REST fetch
# ---------------------------------------------------------------------------

def _fetch_supabase_rows(client: httpx.Client) -> list[dict]:
    """Pull active players from the public Supabase REST endpoint.

    Returns [] if the response is non-200, malformed, or empty.
    """
    url = f"{COURT_CONSENSUS_SUPABASE_URL}/rest/v1/players"
    params = {
        "select": (
            "name,team,position,age,years_exp,elo_rating_points,"
            "elo_rating_categories,total_votes,status"
        ),
        "order": "elo_rating_points.desc",
        "limit": str(PAGE_SIZE),
    }
    headers = {
        "apikey": COURT_CONSENSUS_SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {COURT_CONSENSUS_SUPABASE_ANON_KEY}",
        "Accept": "application/json",
    }
    try:
        r = client.get(url, params=params, headers=headers, timeout=30)
    except Exception as e:
        log.warning("Court Consensus: Supabase GET raised: %s", e)
        return []
    if r.status_code != 200:
        log.warning("Court Consensus: Supabase returned %s", r.status_code)
        return []
    try:
        data = r.json()
    except Exception as e:
        log.warning("Court Consensus: Supabase response not JSON: %s", e)
        return []
    if not isinstance(data, list):
        log.warning("Court Consensus: unexpected payload shape: %r", type(data))
        return []
    return data


# ---------------------------------------------------------------------------
# CSV fallback — same pattern as DARKO's data/darko/darko_dump.csv path
# ---------------------------------------------------------------------------

def _load_csv_fallback(path: Path) -> list[dict]:
    """Read a CSV at ``path`` and project it onto the Supabase row shape.

    Tolerant of column naming. Supported aliases:
      * name         : name, player, full_name
      * team         : team, nba_team
      * position     : position, pos, primary_position
      * age          : age
      * elo_points   : elo_rating_points, elo_points, value, points_value
      * elo_cats     : elo_rating_categories, elo_categories, cats_value
      * total_votes  : total_votes, votes
    """
    if not path.exists():
        return []
    aliases = {
        "name":         ("name", "player", "full_name"),
        "team":         ("team", "nba_team"),
        "position":     ("position", "pos", "primary_position"),
        "age":          ("age",),
        "elo_pts":      ("elo_rating_points", "elo_points", "value", "points_value"),
        "elo_cats":     ("elo_rating_categories", "elo_categories", "cats_value"),
        "votes":        ("total_votes", "votes"),
    }
    def _norm(s: str) -> str:
        return (s or "").strip().lower().replace(" ", "_")
    def _pick(row, keys):
        for k in keys:
            if k in row and row[k] not in (None, ""):
                return row[k]
        return None

    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {_norm(k): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
            name = _pick(row, aliases["name"])
            if not name:
                continue
            rows.append({
                "name": name,
                "team": _pick(row, aliases["team"]),
                "position": _pick(row, aliases["position"]),
                "age": _pick(row, aliases["age"]),
                "elo_rating_points": _pick(row, aliases["elo_pts"]),
                "elo_rating_categories": _pick(row, aliases["elo_cats"]),
                "total_votes": _pick(row, aliases["votes"]),
                "status": "active",
            })
    return rows


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class CourtConsensus(BaseSource):
    slug = "court_consensus"
    name = "Court Consensus — crowdsourced dynasty NBA rankings"
    category = "market"
    update_frequency = "daily"
    tos_compliant = True
    # Default consensus weight (1.0). DARKO stays heavier at 1.5 because
    # it provides both an impact metric AND a longevity signal that CC
    # by design lacks; CC is pure community ELO. Once we have a
    # production backtest, both float on track-record multiplier.
    default_weight = 1.0
    homepage = "https://courtconsensus.com/rankings"
    notes = (
        "Crowdsourced ELO rankings derived from head-to-head community "
        "votes (76k+ data points and growing). Tier-1 access via the "
        "site's own public Supabase REST endpoint; CSV fallback at "
        "data/court_consensus/court_consensus_dump.csv."
    )

    CSV_FALLBACK = DEFAULT_CSV_PATH

    def __init__(
        self,
        client: httpx.Client | None = None,
        csv_path: Optional[Path | str] = None,
    ):
        super().__init__(client=client)
        if csv_path is not None:
            self.CSV_FALLBACK = Path(csv_path)
        elif os.environ.get("DYNASTY_BBALL_CC_CSV_PATH"):
            self.CSV_FALLBACK = Path(os.environ["DYNASTY_BBALL_CC_CSV_PATH"])

    def fetch(self) -> Iterator[RankingRecord]:
        rows = _fetch_supabase_rows(self._client)
        if not rows:
            log.warning(
                "Court Consensus: live fetch returned 0 rows — "
                "falling back to CSV at %s",
                self.CSV_FALLBACK,
            )
            rows = _load_csv_fallback(self.CSV_FALLBACK)
        if not rows:
            return iter([])

        captured_at = datetime.utcnow()

        # Emit twice — once per league format. CC's points-league ELO is
        # the right signal for both ``points_dhk`` (Phil's league) and
        # the generic ``points_default``; until we have a categories
        # adapter, both formats consume the same value.
        records = parse_court_consensus_rows(rows, captured_at=captured_at, league_format="points_dhk")
        records += parse_court_consensus_rows(rows, captured_at=captured_at, league_format="points_default")
        for r in records:
            yield r
