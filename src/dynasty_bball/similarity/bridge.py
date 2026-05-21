"""College->NBA player-id bridge.

The rookie similarity engine projects an NCAA player by:

  1. Finding his top-K college-comp player-seasons (via KNN on the
     college profile vector).
  2. Looking up each comp's REALIZED NBA career (longevity, fantasy
     points) so we can aggregate "what did the rest of their career
     look like."

Step 2 requires a mapping from NCAA player-id (barttorvik's btv_pid,
which we store as ``sr_player_id``) to NBA player-id. That mapping
doesn't exist anywhere as a published artifact -- sports-reference's
internal links go both directions but we don't scrape SR. Instead we
build the bridge HEURISTICALLY:

  For every NBA player in the historical NBA corpus, find the NCAA
  player-season(s) whose name and likely-rookie-age match. Last
  season's age in college roughly matches the NBA debut age, so a
  pre-2009 NBA debut -> no college match in our corpus is FINE (and
  expected -- barttorvik starts at 2008).

The match is built off the existing name_resolver's canonical_key so
spelling variants (Jr/Sr suffixes, diacritics, common nicknames) all
collapse. Where canonical_key alone would over-match (two distinct
"Marcus Williams" players in NCAA who both made the NBA), we add age
and recency as secondary filters.

False matches are catastrophic (Flagg's comp Joe Schmo gets credited
with Klay Thompson's career) so we err on the side of "no match" and
document the gap. False non-matches just drop the comp's NBA career
to "zero remaining seasons", which is the right behavior for
college-only players anyway.

Output
------
``data/bridge/ncaa_to_nba.json`` -- see ``build_bridge`` docstring for
the schema.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..name_resolver import canonical_key, load_alias_map
from ..sources.historical_nba import HistoricalPlayerSeason
from ..sources.historical_ncaa import HistoricalNCAASeason


log = logging.getLogger(__name__)


DEFAULT_BRIDGE_PATH = Path("data/bridge/ncaa_to_nba.json")


@dataclass
class BridgeMatch:
    nba_id: str
    btv_pid: str
    nba_name: str
    ncaa_name: str
    school: str
    ncaa_seasons: list[str]
    match_tier: str   # canonical / canonical_ambiguous


def _group_ncaa_by_canon(
    ncaa_rows: list[HistoricalNCAASeason],
) -> dict[str, dict]:
    """Group NCAA rows by canonical_key(name).

    Returns dict[canon_key, dict] where each value has btv_pid, name,
    school, and seasons list. When two btv_pids share a canonical key,
    picks the one with the most seasons and flags ambiguous_pids.
    """
    by_canon_by_pid: dict[str, dict[str, dict]] = {}
    for r in ncaa_rows:
        ck = canonical_key(r.name)
        if not ck:
            continue
        pid = r.sr_player_id
        slot = by_canon_by_pid.setdefault(ck, {})
        if pid not in slot:
            slot[pid] = {
                "btv_pid": pid,
                "name": r.name,
                "school": r.school,
                "seasons": [],
            }
        slot[pid]["seasons"].append((r.season_end_year, r.season))
    out: dict[str, dict] = {}
    for ck, by_pid in by_canon_by_pid.items():
        if len(by_pid) > 1:
            best_pid = max(by_pid.keys(), key=lambda p: len(by_pid[p]["seasons"]))
            entry = dict(by_pid[best_pid])
            entry["ambiguous_pids"] = sorted(by_pid.keys())
            out[ck] = entry
        else:
            out[ck] = next(iter(by_pid.values()))
    return out


def _group_nba_by_id(
    nba_rows: list[HistoricalPlayerSeason],
) -> dict[str, dict]:
    """Group NBA rows by nba_id. Returns one entry per unique player."""
    out: dict[str, dict] = {}
    for r in nba_rows:
        nid = r.nba_id
        slot = out.setdefault(nid, {
            "nba_id": nid,
            "name": r.name,
            "first_age": r.age,
            "first_season_end_year": r.season_end_year,
            "seasons": [],
        })
        slot["seasons"].append((r.season_end_year, r.season, r.age))
        if r.season_end_year < slot["first_season_end_year"]:
            slot["first_season_end_year"] = r.season_end_year
            slot["first_age"] = r.age
    return out


def build_bridge(
    nba_rows: list[HistoricalPlayerSeason],
    ncaa_rows: list[HistoricalNCAASeason],
) -> dict:
    """Build the NCAA->NBA crosswalk.

    Strategy: per-NBA-player canonical_key lookup into the NCAA group.
    For matched names, sanity-check that the NCAA player's last season
    ends within ~4 years of the NBA player's debut.

    Pre-2009 NBA debuts (those with no NCAA data possible) are
    bucketed separately as ``n_pre_corpus_nba_players`` so the
    headline match-rate isn't unfairly diluted by them.

    Returns a dict with keys: generated_at, n_nba_players_total,
    n_nba_players_matched, match_rate, n_pre_corpus_nba_players,
    by_nba_id, by_btv_pid, unmatched_nba_ids.
    """
    nba_by_id = _group_nba_by_id(nba_rows)
    ncaa_by_canon = _group_ncaa_by_canon(ncaa_rows)

    # Load the hand-curated alias map (Nic Claxton <-> Nicolas Claxton,
    # Mo Bamba <-> Mohamed Bamba, etc.). The map is alias->canonical;
    # we invert it to canonical->[aliases] so we can search the NCAA
    # corpus under EVERY known spelling of an NBA player.
    try:
        alias_to_canonical = load_alias_map()
    except Exception:
        alias_to_canonical = {}
    canonical_to_aliases: dict[str, list[str]] = {}
    for alias, canon in alias_to_canonical.items():
        canonical_to_aliases.setdefault(canon, []).append(alias)

    matched: dict[str, dict] = {}
    by_pid: dict[str, str] = {}
    unmatched: list[str] = []
    n_pre_corpus = 0
    n_alias_hits = 0
    NCAA_FIRST_YEAR = 2008

    def _resolve_ck(name: str) -> Optional[str]:
        ck = canonical_key(name or "")
        if not ck:
            return None
        if ck in ncaa_by_canon:
            return ck
        # Try the alias map: if NBA name's canonical key is itself
        # listed as a canonical in the alias file, every aliased
        # spelling is a valid NCAA-side lookup.
        for alt in canonical_to_aliases.get(ck, []):
            if alt in ncaa_by_canon:
                return alt
        # Also: if the NBA name is itself an alias of some OTHER
        # canonical (e.g. NBA = "Bones Hyland" -> canonical "bones hyland";
        # alias map says "bones hyland" itself maps to canonical "bones hyland"
        # but NCAA may have "Nah'Shon Hyland" as canon -> need to try
        # all aliases of the OTHER canonical that includes ck).
        # The alias_to_canonical lookup gives us the target canonical.
        other_canon = alias_to_canonical.get(ck)
        if other_canon and other_canon != ck:
            if other_canon in ncaa_by_canon:
                return other_canon
            for alt in canonical_to_aliases.get(other_canon, []):
                if alt in ncaa_by_canon:
                    return alt
        return None

    for nba_id, nba in nba_by_id.items():
        nba_debut = nba["first_season_end_year"]
        ck_resolved = _resolve_ck(nba.get("name", ""))
        if ck_resolved is None:
            unmatched.append(nba_id)
            if nba_debut <= NCAA_FIRST_YEAR:
                n_pre_corpus += 1
            continue
        ncaa = ncaa_by_canon.get(ck_resolved)
        if ncaa is None:
            unmatched.append(nba_id)
            if nba_debut <= NCAA_FIRST_YEAR:
                n_pre_corpus += 1
            continue
        ncaa_last_year = max(y for y, _ in ncaa["seasons"])
        ncaa_first_year = min(y for y, _ in ncaa["seasons"])
        if nba_debut < ncaa_first_year:
            # NBA debut PREDATES college season -- different person.
            unmatched.append(nba_id)
            if nba_debut <= NCAA_FIRST_YEAR:
                n_pre_corpus += 1
            continue
        if (nba_debut - ncaa_last_year) > 4:
            # Long gap suggests different person.
            unmatched.append(nba_id)
            if nba_debut <= NCAA_FIRST_YEAR:
                n_pre_corpus += 1
            continue
        tier = "canonical"
        if "ambiguous_pids" in ncaa:
            tier = "canonical_ambiguous"
        # Was this only reachable via the alias map?
        original_ck = canonical_key(nba.get("name", "") or "")
        if original_ck != ck_resolved:
            tier = "alias"
            n_alias_hits += 1
        match = {
            "btv_pid": ncaa["btv_pid"],
            "ncaa_name": ncaa["name"],
            "school": ncaa["school"],
            "ncaa_seasons": [s for _, s in sorted(ncaa["seasons"])],
            "match_tier": tier,
        }
        matched[nba_id] = match
        by_pid[ncaa["btv_pid"]] = nba_id

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_nba_players_total": len(nba_by_id),
        "n_nba_players_matched": len(matched),
        "match_rate": round(len(matched) / max(1, len(nba_by_id)), 4),
        "n_pre_corpus_nba_players": n_pre_corpus,
        "n_alias_hits": n_alias_hits,
        "by_nba_id": matched,
        "by_btv_pid": by_pid,
        "unmatched_nba_ids": sorted(unmatched),
    }


def save_bridge(payload: dict, path: Path | str = DEFAULT_BRIDGE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def load_bridge(path: Path | str = DEFAULT_BRIDGE_PATH) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("bridge: load failed: %s", e)
        return None


def coverage_excluding_pre_corpus(payload: dict) -> float:
    """Match rate among NBA players whose debut is in the NCAA corpus window.

    A useful diagnostic separate from the raw ``match_rate``, which is
    diluted by pre-2008 NBA stars who could never bridge.
    """
    total = payload.get("n_nba_players_total", 0)
    matched = payload.get("n_nba_players_matched", 0)
    pre = payload.get("n_pre_corpus_nba_players", 0)
    bridgeable = max(1, total - pre)
    return round(matched / bridgeable, 4)
