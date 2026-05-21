"""Draft-stock prior — penalize mid-major freshman noise in the rookie chain.

PR #7's college->NBA similarity engine surfaces every prospect that
passes a coarse BPM/PPG/USG filter, but mid-major freshmen with a hot
year (Moustapha Thiam, Austin Rapp, JT Toppin, Jayden Quaintance) sneak
into the top-20 because the barttorvik corpus has no notion of
"actually drafted." This module adds a *draft-stock multiplier* that
boosts real lottery picks and penalizes prospects who don't show up
on any NBA draft board.

Data sources (in priority order):

  1. **nba_api DraftHistory** -- the authoritative record of what was
     actually drafted (2024, 2025 covered). Free, reliable, already
     used by PR #4. Pick #1 -> tier=top_5, lottery -> tier=lottery,
     etc.
  2. **Barttorvik rec_rank** -- prep-recruit percentile (already in
     the NCAA corpus column 34). Used as a fallback when a prospect
     is in college but not yet drafted. rec_rank >= 99 = treat as
     lottery prior, 95-98.99 = first_round prior, etc.

The output is a cached ``data/draft_stock/big_board.json`` keyed by
canonical name (via ``name_resolver.canonical_key``) so the lookup
works across Sleeper / Bbref / nba_api spellings (Bub Carrington in
the draft, Carlton Carrington in barttorvik, etc.).

Multiplier table (applied to ``rookie_dynasty_value`` before rescale):

  * top_5         (pick #1-5)         -> 1.20  (the real lottery)
  * lottery       (pick #6-14)        -> 1.10
  * first_round   (pick #15-30)       -> 1.00  (baseline)
  * second_round  (pick #31-60)       -> 0.85
  * not_on_board  (low college BPM)   -> 0.50  (noise candidates)
  * not_on_board  + sub-p80 college   -> 0.30  (even noisier)

Live refresh is gated behind ``DYNASTY_BBALL_DRAFT_LIVE=1`` so CI never
hits stats.nba.com. The committed JSON cache (covering 2008-2025
draft classes) is enough for tests and the launcher.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)


DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "draft_stock" / "big_board.json"
DEFAULT_BACKFILL_START_YEAR = 2008  # matches NCAA corpus start
DEFAULT_BACKFILL_END_YEAR = 2025


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

# Multiplier applied to rookie_dynasty_value. Real lottery picks get a
# slight boost (the model's college-only view undervalues them because
# their college season is a small sample of an NBA-bound talent), real
# 2nd-round picks get docked, and prospects who don't show up on any
# board at all (the Thiam/Toppin/Quaintance class) get heavily
# penalized.
TIER_MULTIPLIER: dict[str, float] = {
    "top_5":         1.20,
    "lottery":       1.10,
    "first_round":   1.00,
    "second_round":  0.85,
    "not_on_board":  0.50,
    "noise":         0.30,   # not_on_board + sub-p80 college percentile
}

# Order matters when comparing tiers (e.g. "is this tier >= lottery?").
TIER_ORDER: list[str] = [
    "noise", "not_on_board", "second_round", "first_round",
    "lottery", "top_5",
]


def tier_for_pick(consensus_rank: int) -> str:
    """Map a 1-60 draft pick number to a tier name."""
    if consensus_rank <= 5:
        return "top_5"
    if consensus_rank <= 14:
        return "lottery"
    if consensus_rank <= 30:
        return "first_round"
    return "second_round"


def tier_for_rec_rank(rec_rank: Optional[float]) -> Optional[str]:
    """Map barttorvik prep-recruit percentile to a draft-stock tier.

    Returns ``None`` for prospects with no recruiting data -- those
    fall through to ``not_on_board`` / ``noise``.
    """
    if rec_rank is None:
        return None
    # 5-star-equivalent percentile -- behaves like a recruiting prior
    # only, NOT a draft-board prior. We use it sparingly.
    if rec_rank >= 99.5:
        return "lottery"        # consensus top-30 recruits trend lottery
    if rec_rank >= 97.0:
        return "first_round"    # 5-star territory
    if rec_rank >= 92.0:
        return "second_round"   # 4-star
    return None                  # below that, no signal


def multiplier_for_tier(tier: Optional[str]) -> float:
    """Return the rookie-value multiplier for a tier (default 0.50)."""
    if tier is None:
        return TIER_MULTIPLIER["not_on_board"]
    return TIER_MULTIPLIER.get(tier, 1.0)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DraftProspect:
    """One row in the big-board cache.

    ``consensus_rank`` is the 1-60 NBA draft pick number (or projected
    pick for current-class boards). ``tier`` is the derived tier name.
    ``source`` is the provider (``nba_api`` / ``rec_rank``).
    """
    name: str
    canonical_name: str
    school: Optional[str]
    draft_year: Optional[int]
    consensus_rank: Optional[int]
    tier: str
    source: str
    as_of: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "school": self.school,
            "draft_year": self.draft_year,
            "consensus_rank": self.consensus_rank,
            "tier": self.tier,
            "source": self.source,
            "as_of": self.as_of,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DraftProspect":
        return cls(
            name=d.get("name", ""),
            canonical_name=d.get("canonical_name", ""),
            school=d.get("school"),
            draft_year=d.get("draft_year"),
            consensus_rank=d.get("consensus_rank"),
            tier=d.get("tier", "first_round"),
            source=d.get("source", "nba_api"),
            as_of=d.get("as_of", ""),
        )


# ---------------------------------------------------------------------------
# Canonical-name helpers
# ---------------------------------------------------------------------------

def _canonical(name: str) -> str:
    """Defer to the project's name_resolver canonicalization."""
    # Import lazily so this module doesn't pull resolver at import time.
    from ..name_resolver import canonical_key
    return canonical_key(name) or name.strip().lower()


# ---------------------------------------------------------------------------
# Big-board fetch (live)
# ---------------------------------------------------------------------------

def fetch_nba_draft_year(year: int, *, request_timeout: int = 30) -> list[DraftProspect]:
    """Pull the actual draft for one year via nba_api DraftHistory.

    Returns a list of ``DraftProspect``. Empty list on failure.
    """
    try:
        from nba_api.stats.endpoints import drafthistory
    except ImportError:
        log.warning("draft_stock: nba_api not available; skipping year %s", year)
        return []
    try:
        resp = drafthistory.DraftHistory(
            season_year_nullable=str(year),
            timeout=request_timeout,
        )
        rows = resp.get_normalized_dict().get("DraftHistory", [])
    except Exception as e:
        log.warning("draft_stock: nba_api fetch for %s failed: %s", year, e)
        return []
    out: list[DraftProspect] = []
    for row in rows:
        pick = row.get("OVERALL_PICK")
        name = row.get("PLAYER_NAME") or ""
        school = row.get("ORGANIZATION")
        if not pick or not name:
            continue
        out.append(DraftProspect(
            name=name,
            canonical_name=_canonical(name),
            school=school,
            draft_year=year,
            consensus_rank=int(pick),
            tier=tier_for_pick(int(pick)),
            source="nba_api",
        ))
    log.info("draft_stock: fetched %d picks for draft year %s", len(out), year)
    return out


def fetch_all_nba_draft_history(
    start_year: int = DEFAULT_BACKFILL_START_YEAR,
    end_year: int = DEFAULT_BACKFILL_END_YEAR,
    *,
    sleep_between: float = 2.0,
) -> list[DraftProspect]:
    """Pull every NBA draft from start_year through end_year via nba_api."""
    import time
    out: list[DraftProspect] = []
    for year in range(start_year, end_year + 1):
        out.extend(fetch_nba_draft_year(year))
        if sleep_between > 0 and year != end_year:
            time.sleep(sleep_between)
    return out


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def save_big_board(
    prospects: Iterable[DraftProspect],
    path: Path = DEFAULT_CACHE_PATH,
) -> None:
    """Persist the cache. Sorted by (draft_year desc, consensus_rank asc) so
    diffs are deterministic and the JSON stays readable."""
    items = sorted(
        prospects,
        key=lambda p: (
            -(p.draft_year or 0),
            p.consensus_rank if p.consensus_rank is not None else 999,
            p.canonical_name,
        ),
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_prospects": len(items),
        "prospects": [p.to_dict() for p in items],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    log.info("draft_stock: wrote %d prospects to %s", len(items), path)


def load_big_board(path: Path = DEFAULT_CACHE_PATH) -> list[DraftProspect]:
    """Load the cached big board. Returns [] if missing."""
    if not path.exists():
        log.warning("draft_stock: cache missing at %s -- run refresh_big_board()", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [DraftProspect.from_dict(d) for d in payload.get("prospects", [])]


def refresh_big_board(
    cache_path: Path = DEFAULT_CACHE_PATH,
    start_year: int = DEFAULT_BACKFILL_START_YEAR,
    end_year: int = DEFAULT_BACKFILL_END_YEAR,
) -> list[DraftProspect]:
    """Refresh the cache (live, requires network). Gated behind env var."""
    if os.environ.get("DYNASTY_BBALL_DRAFT_LIVE") != "1":
        log.warning(
            "draft_stock: refresh requested but DYNASTY_BBALL_DRAFT_LIVE != 1; "
            "use the env var to opt into live network calls."
        )
        return load_big_board(cache_path)
    prospects = fetch_all_nba_draft_history(start_year=start_year, end_year=end_year)
    save_big_board(prospects, cache_path)
    return prospects


# ---------------------------------------------------------------------------
# Big-board index (used by the rookie engine)
# ---------------------------------------------------------------------------

@dataclass
class BigBoardIndex:
    """Lookup wrapper over ``prospects``.

    ``by_canonical`` keys on the resolver-canonical name so the rookie
    engine can resolve "Bub Carrington" (NBA draft) -> "Carlton
    Carrington" (barttorvik) by canonicalizing both sides.

    For prospects that appear in multiple draft years (a guy who
    declared, withdrew, then was drafted next year), we keep the
    record with the BEST (lowest) ``consensus_rank``. That's the most
    favorable real NBA outcome we have on file.
    """
    by_canonical: dict[str, DraftProspect]
    by_draft_year: dict[int, list[DraftProspect]]
    n_prospects: int

    def lookup(self, name: str, *, aliases: Optional[Iterable[str]] = None) -> Optional[DraftProspect]:
        """Look up a prospect by name.

        ``aliases`` is an optional iterable of additional name strings
        to try (e.g. a Sleeper full_name when the bbref name doesn't
        canonicalize to the same key). All are canonicalized.
        """
        keys = [_canonical(name)]
        if aliases:
            for a in aliases:
                if a:
                    keys.append(_canonical(a))
        for k in keys:
            if k and k in self.by_canonical:
                return self.by_canonical[k]
        return None

    def tier_for(self, name: str, *, aliases: Optional[Iterable[str]] = None) -> Optional[str]:
        p = self.lookup(name, aliases=aliases)
        return p.tier if p else None


def build_index(prospects: Iterable[DraftProspect]) -> BigBoardIndex:
    """Build a name-keyed index. Best-rank-wins on duplicates.

    Also folds in ``data/name_aliases.json`` so that lookups by
    alternate spellings (e.g. "Carlton Carrington" -> the NBA-drafted
    "Bub Carrington") find the same prospect record. This is the
    integration point with PR #6's name resolver.
    """
    by_can: dict[str, DraftProspect] = {}
    by_year: dict[int, list[DraftProspect]] = {}
    n = 0
    for p in prospects:
        n += 1
        if p.draft_year is not None:
            by_year.setdefault(p.draft_year, []).append(p)
        key = p.canonical_name or _canonical(p.name)
        if not key:
            continue
        existing = by_can.get(key)
        if existing is None:
            by_can[key] = p
            continue
        # Best-rank-wins: prefer the lower consensus_rank (and prefer
        # real NBA draft over rec_rank prior).
        existing_rank = existing.consensus_rank if existing.consensus_rank is not None else 999
        new_rank = p.consensus_rank if p.consensus_rank is not None else 999
        if (existing.source == "rec_rank" and p.source == "nba_api"):
            by_can[key] = p
        elif new_rank < existing_rank and existing.source == p.source:
            by_can[key] = p
    # Fold in alias map: every alias points at the canonical's prospect
    # record. This lets "Carlton Carrington" (barttorvik) resolve to
    # the "Bub Carrington" (nba_api) record.
    try:
        from ..name_resolver import load_alias_map
        alias_map = load_alias_map()
    except Exception as e:
        log.debug("draft_stock: alias map unavailable: %s", e)
        alias_map = {}
    for alias_key, canonical_key_str in alias_map.items():
        if not alias_key or not canonical_key_str:
            continue
        # alias_map is already keyed by canonical_key(alias) -> canonical_key(canonical)
        if alias_key in by_can:
            continue  # already a direct hit
        target = by_can.get(canonical_key_str)
        if target is not None:
            by_can[alias_key] = target
    return BigBoardIndex(by_canonical=by_can, by_draft_year=by_year, n_prospects=n)


def load_index(path: Path = DEFAULT_CACHE_PATH) -> BigBoardIndex:
    """Load the cache and build the lookup index in one call."""
    return build_index(load_big_board(path))


# ---------------------------------------------------------------------------
# Rookie-engine integration: per-player multiplier
# ---------------------------------------------------------------------------

# Recruiting-rank floors that govern the not-on-board fallback. A
# prospect who never appeared on a real NBA draft board is treated
# as noise UNLESS they were a consensus elite recruit (5-star /
# rec_rank >= ELITE_REC_RANK). Mid-tier 4-stars who didn't get
# drafted still get the harsher noise penalty -- the empirical PR #7
# noise (Toppin, Thiam, Rapp) sits in exactly this rec_rank band.
ELITE_REC_RANK = 99.5       # consensus top-10 high-school recruit
FOURSTAR_REC_RANK = 95.0    # 4-star territory


def compute_multiplier(
    *,
    name: str,
    index: BigBoardIndex,
    college_rec_rank: Optional[float] = None,
    college_percentile: Optional[float] = None,
    aliases: Optional[Iterable[str]] = None,
) -> tuple[float, str, Optional[DraftProspect]]:
    """Return ``(multiplier, tier_label, prospect|None)`` for one rookie.

    Lookup chain:
      1. NBA draft history (real picks 1-60) -> top_5 / lottery /
         first_round / second_round multipliers. Authoritative.
      2. Not on the board but a CONSENSUS ELITE recruit
         (rec_rank >= 99) -> ``rec_rank_prior:second_round`` (0.85x).
         These are guys who declared late or returned to school;
         worth more than pure noise but never more than baseline.
      3. Not on the board, 4-star recruit (rec_rank 95-98.99) ->
         ``not_on_board`` (0.50x). 4-stars without an NBA outcome
         are still mostly future role players.
      4. Not on the board, sub-4-star or unranked -> ``noise``
         (0.30x). This is the empirical PR #7 noise band
         (Toppin rec=78, Thiam rec=88.6, Rapp rec=None).

    The optional ``college_percentile`` parameter (player's
    rookie_dynasty_value rank in cohort) is currently informational --
    the noise gate is driven by the recruiting profile because that's
    what separates a real NBA prospect from a college outlier.
    """
    prospect = index.lookup(name, aliases=aliases)
    if prospect is not None:
        return multiplier_for_tier(prospect.tier), prospect.tier, prospect
    # Not on any draft board. The fallback tier is set by the
    # recruiting profile, NOT by the player's own rookie_dv (that
    # would create a feedback loop -- the whole point of the prior
    # is to discount players whose rookie_dv overstates their NBA path).
    #
    # The PR #7 noise pattern (Thiam, Toppin, Rapp, Quaintance) sits
    # in a wide rec_rank band (Rapp=None, Toppin=78, Thiam=88.6,
    # Quaintance=98.6). All FOUR didn't get drafted in 2025. So we
    # apply the noise penalty uniformly to all undrafted prospects,
    # with a small "elite-recruit floor" only for true consensus
    # top-5 high-schoolers (rec_rank >= ELITE_REC_RANK) -- those are
    # one-and-done future lottery picks delaying a year, not the
    # PR #7 false-positive noise.
    if college_rec_rank is not None and college_rec_rank >= ELITE_REC_RANK:
        return multiplier_for_tier("second_round"), "rec_rank_prior:elite", None
    if college_rec_rank is not None and college_rec_rank >= FOURSTAR_REC_RANK:
        # 4-stars without a draft outcome are still mostly future
        # role players, NOT first-rounders. Treat as not-on-board
        # baseline (0.50x); not full noise.
        return TIER_MULTIPLIER["noise"], "not_on_board:fourstar", None
    return TIER_MULTIPLIER["noise"], "noise", None


def apply_multipliers_to_rookie_entries(
    rookie_entries: dict,
    index: BigBoardIndex,
    *,
    fmt_keys: tuple[str, ...] = ("points_dhk", "points_default"),
) -> dict:
    """Mutate ``rookie_entries`` in place: scale dynasty_value by tier.

    ``rookie_entries`` is the ``projections_by_btv_pid`` map from
    ``career_arc.build_rookie_projections`` -- each value is a dict
    with ``name``, ``school``, ``points_dhk`` (CareerProjection),
    ``points_default`` (CareerProjection), and a list of comparables.

    For each player:
      1. Compute their college_percentile from rookie_dynasty_value
         (within-cohort, format-agnostic; we use points_dhk).
      2. Look up draft tier via canonical name.
      3. Multiply both formats' dynasty_value in place.
      4. Stash ``draft_stock_tier`` + ``draft_stock_multiplier`` +
         ``draft_stock_source`` on the entry for downstream rendering.

    The caller is expected to ``rescale_to_0_100`` afterwards so the
    adjusted values still land in 0-100.
    """
    if not rookie_entries:
        return {"n_adjusted": 0, "tier_counts": {}}
    # College percentile within cohort -- use points_dhk as canonical.
    fmt_canon = fmt_keys[0]
    values = []
    for entry in rookie_entries.values():
        proj = entry.get(fmt_canon)
        if proj is not None:
            values.append(float(proj.dynasty_value))
    if not values:
        return {"n_adjusted": 0, "tier_counts": {}}
    values.sort()
    n = len(values)

    def _percentile(v: float) -> float:
        # rank percentile in 0-100 scale
        import bisect
        idx = bisect.bisect_left(values, v)
        return 100.0 * idx / max(1, n - 1)

    tier_counts: dict[str, int] = {}
    n_adjusted = 0
    for pid, entry in rookie_entries.items():
        proj_canon = entry.get(fmt_canon)
        if proj_canon is None:
            continue
        pct = _percentile(float(proj_canon.dynasty_value))
        mult, tier_label, prospect = compute_multiplier(
            name=entry.get("name", ""),
            index=index,
            college_rec_rank=entry.get("rec_rank"),
            college_percentile=pct,
        )
        entry["draft_stock_tier"] = tier_label
        entry["draft_stock_multiplier"] = mult
        entry["draft_stock_source"] = (
            prospect.source if prospect is not None
            else ("rec_rank" if tier_label.startswith("rec_rank_prior") else "not_on_board")
        )
        entry["draft_stock_pick"] = prospect.consensus_rank if prospect else None
        entry["draft_stock_draft_year"] = prospect.draft_year if prospect else None
        entry["college_percentile"] = pct
        tier_counts[tier_label] = tier_counts.get(tier_label, 0) + 1
        if abs(mult - 1.0) > 1e-9:
            n_adjusted += 1
            for fmt in fmt_keys:
                proj = entry.get(fmt)
                if proj is None:
                    continue
                # Scale BOTH dynasty_value (current displayed value)
                # and dynasty_value_raw (the input to rescale_to_0_100).
                # The post-multiplier rescale uses dynasty_value_raw,
                # so failing to scale that field would silently revert
                # the multiplier.
                proj.dynasty_value = float(proj.dynasty_value) * mult
                if hasattr(proj, "dynasty_value_raw") and proj.dynasty_value_raw is not None:
                    proj.dynasty_value_raw = float(proj.dynasty_value_raw) * mult
    return {"n_adjusted": n_adjusted, "tier_counts": tier_counts}


# ---------------------------------------------------------------------------
# Module-level entry: load and apply
# ---------------------------------------------------------------------------

def load_index_or_empty(path: Path = DEFAULT_CACHE_PATH) -> BigBoardIndex:
    """Convenience: load_index, but never crash on a missing cache.

    Returns an empty index so callers can apply multipliers safely
    even when the cache hasn't been built yet (e.g. fresh checkouts).
    """
    try:
        return load_index(path)
    except Exception as e:
        log.warning("draft_stock: failed to load big board: %s", e)
        return BigBoardIndex(by_canonical={}, by_draft_year={}, n_prospects=0)
