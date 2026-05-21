"""Player name resolver — single source of truth for player identity.

Different sources spell the same player differently — DARKO uses full-name
versions ("Nicolas Claxton", "Alexandre Sarr"), Sleeper uses common short
forms ("Nic Claxton", "Alex Sarr"), Basketball-Reference uses nicknames
("Bones Hyland" for "Nah'Shon Hyland"), and several have diacritic
variants ("Jusuf Nurkić" vs "Jusuf Nurkic"). The composite ranker happily
emits a row for each spelling, producing duplicate-and-orphaned rows in
the rendered top-300.

The resolver consumes a stream of incoming records (with full_name, team,
position) and a pool of existing Player rows, then returns the best
existing match — or ``None`` if no confident match exists. The caller
then attaches the record to that Player (joining the source data) instead
of creating a new row.

Tier cascade (strict → loose):

  Tier 1 — Canonical key (diacritic-fold + suffix-strip + punctuation-strip).
           "Doncic"/"Dončić"/"DONCIC" all hash to ``luka doncic``.
           This catches >95% of cross-source dupes.

  Tier 2 — Last name + first-name initial + position-bucket + team.
           "Nic Claxton" + "Nicolas Claxton" both have last="claxton",
           first[0]="n", same team "BKN" → confident match.
           "Bub Carrington" + "Carlton Carrington" share last="carrington"
           and team "WAS" but first[0]="b" vs "c" — Tier 2 declines.

  Alias map — Hand-curated edge cases from ``data/name_aliases.json``.
              "Bub Carrington" ↔ "Carlton Carrington", "Bones Hyland"
              ↔ "Nah'Shon Hyland", and ~40 others. Consulted BEFORE
              Tier 3 so the looser fuzzy logic never has to guess on
              cases we already know the answer for.

  Tier 3 — Conservative fuzzy. Last names equal (post-normalization),
           position-bucket compatible, SAME team. First names then
           must satisfy ONE of: shared 2-char prefix, known
           diminutive, or token-set similarity ≥ 0.80.

Design principle: false merges are catastrophic, false misses just
exclude one player. Tier 3 always REQUIRES same team — without that
guard, two different "Anthony Davis" players could collapse. The
alias map is how we extend matching beyond what the algorithm can
safely guess.

All inputs are normalized via :func:`canonical_key` so callers don't
have to pre-clean strings.
"""
from __future__ import annotations
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team normalization — DARKO emits "Washington Wizards", Sleeper emits "WAS".
# ---------------------------------------------------------------------------

TEAM_NAME_TO_ABBREV: dict[str, str] = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "la clippers": "LAC",
    "los angeles clippers": "LAC",
    "los angeles lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "philadelphia sixers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}


def normalize_team(team: str | None) -> str | None:
    """Return the 3-letter NBA abbrev for any input team string, else None.

    Accepts abbrevs ("WAS"), full names ("Washington Wizards"), and
    BBRef-style variants. Returns the original (uppercased) value if it
    already looks like an abbrev. Returns ``None`` for empty input.
    """
    if not team:
        return None
    s = str(team).strip()
    if not s:
        return None
    # Already a 2-3 letter code?
    if len(s) <= 4 and s.replace(" ", "").isalpha():
        return s.upper()
    key = s.lower()
    if key in TEAM_NAME_TO_ABBREV:
        return TEAM_NAME_TO_ABBREV[key]
    # Strip leading "the " or trailing whitespace, try again.
    key = re.sub(r"^the\s+", "", key).strip()
    if key in TEAM_NAME_TO_ABBREV:
        return TEAM_NAME_TO_ABBREV[key]
    return None


# ---------------------------------------------------------------------------
# Position bucketing — collapse PG/SG → G, SF/PF → F, C → C.
# ---------------------------------------------------------------------------

POSITION_BUCKETS: dict[str, str] = {
    "PG": "G", "SG": "G", "G": "G",
    "SF": "F", "PF": "F", "F": "F",
    "C": "C",
    # Sleeper occasionally uses hybrid labels.
    "G-F": "G", "F-G": "G",
    "F-C": "F", "C-F": "C",
}


def position_bucket(pos: str | None) -> str | None:
    if not pos:
        return None
    return POSITION_BUCKETS.get(str(pos).strip().upper())


def positions_compatible(a: str | None, b: str | None) -> bool:
    """Two positions are 'compatible' if at least one is unknown OR they
    bucket to the same group OR they are adjacent G↔F / F↔C buckets.

    Conservative enough to prevent obvious miscategorizations (a centre
    is not a point guard) while tolerant of source-to-source disagreement
    on hybrid players.
    """
    ba, bb = position_bucket(a), position_bucket(b)
    if ba is None or bb is None:
        return True
    if ba == bb:
        return True
    return {ba, bb} in ({"G", "F"}, {"F", "C"})


# ---------------------------------------------------------------------------
# Name normalization — canonical key generation.
# ---------------------------------------------------------------------------

_SUFFIX_RE = re.compile(
    r"(?:,\s*|\s+)(jr|sr|ii|iii|iv|v)\.?$",
    flags=re.IGNORECASE,
)
_NON_ALNUM_SPACE = re.compile(r"[^a-z0-9\s\-]")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_suffix(name: str) -> tuple[str, str | None]:
    """Return (suffix-stripped name, suffix_or_none)."""
    suffix = None
    s = name
    while True:
        m = _SUFFIX_RE.search(s)
        if not m:
            break
        suffix = m.group(1).lower()
        s = _SUFFIX_RE.sub("", s).strip()
    return s, suffix


def canonical_key(name: str | None) -> str | None:
    """Lowercase + diacritic-fold + suffix-strip + punctuation-strip.

    Two names with the same canonical_key are *exact* matches under
    Tier 1. Idempotent. Returns ``None`` for empty/None input.
    """
    if not name:
        return None
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.strip().lower()
    s, _ = _strip_suffix(s)
    # Drop apostrophes, periods, commas — keep spaces and hyphens.
    s = s.replace("'", "").replace("’", "").replace(".", "").replace(",", "")
    s = _NON_ALNUM_SPACE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s or None


def _split_last_first(name: str) -> tuple[str, str]:
    """Return (last_token, first_token) from a canonical-keyed name.

    "lebron james" → ("james", "lebron"). Hyphenated last names stay
    intact: "nigel hayes-davis" → ("hayes-davis", "nigel").
    """
    parts = name.split()
    if not parts:
        return ("", "")
    if len(parts) == 1:
        return (parts[0], "")
    return (parts[-1], parts[0])


# Common first-name diminutives — used as a Tier-3 booster only.
_DIMINUTIVES: dict[str, set[str]] = {
    "alex": {"alexander", "alexandre", "alexandr", "aleksandar"},
    "nic": {"nicolas", "nicholas", "nick"},
    "nick": {"nicolas", "nicholas", "nic"},
    "mo": {"mohamed", "mohammed", "moe", "moses"},
    "rob": {"robert", "bobby"},
    "bobby": {"robert", "rob"},
    "cam": {"cameron"},
    "tj": {"thomas", "tyler"},
    "pj": {"patrick", "philip", "paul"},
    "og": {"ogugua"},
    "bub": {"carlton"},
    "bones": {"nahshon", "nah shon"},
    "nahshon": {"bones"},
}


def _share_prefix(a: str, b: str, n: int = 2) -> bool:
    return bool(a) and bool(b) and a[:n].lower() == b[:n].lower()


def _is_known_diminutive(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    if a in _DIMINUTIVES and b in _DIMINUTIVES[a]:
        return True
    if b in _DIMINUTIVES and a in _DIMINUTIVES[b]:
        return True
    return False


def _token_set_similarity(a: str, b: str) -> float:
    """Jaccard-ish similarity on whitespace tokens. Used as a final
    Tier-3 guard. Returns 0.0..1.0."""
    if not a or not b:
        return 0.0
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union)


# ---------------------------------------------------------------------------
# Alias map loading.
# ---------------------------------------------------------------------------

_DEFAULT_ALIAS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "name_aliases.json"
)


def load_alias_map(path: str | Path | None = None) -> dict[str, str]:
    """Build a canonical_key(alias) → canonical_key(canonical) map.

    Returns an empty dict if the file is missing.
    """
    p = Path(path) if path else _DEFAULT_ALIAS_PATH
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover — corrupt config shouldn't kill sync
        log.warning("Failed to load alias map %s: %s", p, e)
        return {}
    entries = data.get("entries", []) if isinstance(data, dict) else []
    out: dict[str, str] = {}
    for entry in entries:
        canon = entry.get("canonical")
        if not canon:
            continue
        canon_key = canonical_key(canon)
        if not canon_key:
            continue
        # The canonical name is its own alias (idempotent).
        out.setdefault(canon_key, canon_key)
        for alias in entry.get("aliases", []) or []:
            ak = canonical_key(alias)
            if ak:
                out[ak] = canon_key
    return out


# ---------------------------------------------------------------------------
# Candidate / query / result records.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolverCandidate:
    """Lightweight projection of a Player row for matching."""
    player_id: int
    full_name: str
    position: str | None
    nba_team: str | None
    # Cached canonical fields (computed on construction).
    canonical: str | None = None
    suffix: str | None = None
    last: str = ""
    first: str = ""
    team_abbrev: str | None = None
    pos_bucket: str | None = None

    @classmethod
    def from_fields(
        cls,
        player_id: int,
        full_name: str,
        position: str | None,
        nba_team: str | None,
    ) -> "ResolverCandidate":
        canon = canonical_key(full_name)
        # Capture the suffix (jr/sr/ii/iii/iv/v) separately. Used to
        # disambiguate the rare case of two real players with identical
        # canonical names where one has a suffix and the other doesn't.
        _, suffix = _strip_suffix(
            unicodedata.normalize("NFKD", str(full_name)).strip().lower()
        ) if full_name else ("", None)
        last, first = _split_last_first(canon) if canon else ("", "")
        return cls(
            player_id=player_id,
            full_name=full_name,
            position=position,
            nba_team=nba_team,
            canonical=canon,
            suffix=suffix,
            last=last,
            first=first,
            team_abbrev=normalize_team(nba_team),
            pos_bucket=position_bucket(position),
        )


@dataclass(frozen=True)
class ResolverQuery:
    full_name: str
    position: str | None = None
    nba_team: str | None = None

    @property
    def suffix(self) -> str | None:
        if not self.full_name:
            return None
        _, suf = _strip_suffix(
            unicodedata.normalize("NFKD", str(self.full_name)).strip().lower()
        )
        return suf


TIERS = ("tier1", "tier2", "tier3", "alias", "unresolved")


@dataclass
class ResolverStats:
    tier1: int = 0
    tier2: int = 0
    tier3: int = 0
    alias: int = 0
    unresolved: int = 0
    total: int = 0
    unmatched: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "tier1": self.tier1,
            "tier2": self.tier2,
            "tier3": self.tier3,
            "alias": self.alias,
            "unresolved": self.unresolved,
        }


# ---------------------------------------------------------------------------
# Main resolver.
# ---------------------------------------------------------------------------

class NameResolver:
    """Build once per sync from the current Player pool, then call
    :meth:`resolve` for each incoming source record.

    The resolver keeps in-memory indexes for each tier so resolution is
    O(1) for Tier 1 and Tier 2 and at-most O(k) for Tier 3 (k = number
    of last-name matches, usually 1-3).
    """

    def __init__(
        self,
        candidates: Iterable[ResolverCandidate],
        alias_map: dict[str, str] | None = None,
    ):
        self.alias_map = alias_map if alias_map is not None else load_alias_map()
        self._by_canonical: dict[str, ResolverCandidate] = {}
        # tier2 index: (last, first[0]) → [cand]. Position and team are
        # checked at resolve-time against the query, which lets us tolerate
        # missing position on either side.
        self._by_t2: dict[tuple, list[ResolverCandidate]] = {}
        # tier3 index: last → [cand]
        self._by_last: dict[str, list[ResolverCandidate]] = {}

        for c in candidates:
            if c.canonical:
                # Prefer the first one we see (DB IDs are stable). Don't
                # overwrite a row that already has full identity with a
                # later orphan-style row.
                if c.canonical not in self._by_canonical:
                    self._by_canonical[c.canonical] = c
                else:
                    # Tie-break: prefer the row with a known team and position.
                    existing = self._by_canonical[c.canonical]
                    if (not existing.team_abbrev or not existing.pos_bucket) and (
                        c.team_abbrev and c.pos_bucket
                    ):
                        self._by_canonical[c.canonical] = c

            if c.last:
                self._by_last.setdefault(c.last, []).append(c)
                t2_key = (c.last, (c.first or "")[:1])
                self._by_t2.setdefault(t2_key, []).append(c)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, query: ResolverQuery) -> tuple[ResolverCandidate | None, str]:
        """Return (candidate, tier_label). tier_label is one of:
        'tier1' | 'tier2' | 'tier3' | 'alias' | 'unresolved'.
        """
        canon = canonical_key(query.full_name)
        if not canon:
            return None, "unresolved"

        # ---- Tier 1: exact canonical hash ------------------------------
        hit = self._by_canonical.get(canon)
        if hit is not None:
            q_suf = query.suffix
            # Suffix tiebreak: if BOTH sides have a suffix and they differ,
            # reject the match — LeBron James Jr. vs LeBron James Sr.
            # would be two different people. If either side lacks a suffix,
            # we trust the canonical match (Phil's directive).
            if q_suf and hit.suffix and q_suf != hit.suffix:
                pass  # fall through to lower tiers
            else:
                return hit, "tier1"

        team_abbr = normalize_team(query.nba_team)
        pos_b = position_bucket(query.position)
        last, first = _split_last_first(canon)
        first_initial = first[:1] if first else ""

        # ---- Tier 2: same last + same first-initial + same team +
        #              compatible position bucket. Team match is REQUIRED.
        if last and first_initial and team_abbr:
            t2_candidates = self._by_t2.get((last, first_initial), [])
            matches = [
                c for c in t2_candidates
                if c.team_abbrev == team_abbr
                and positions_compatible(c.position, query.position)
            ]
            if matches:
                # Prefer a candidate with full identity (team + pos) over
                # an orphan row.
                ranked = sorted(
                    matches,
                    key=lambda c: (bool(c.team_abbrev), bool(c.pos_bucket)),
                    reverse=True,
                )
                return ranked[0], "tier2"

        # ---- Alias map ------------------------------------------------
        canon_alias = self.alias_map.get(canon)
        if canon_alias and canon_alias != canon:
            hit = self._by_canonical.get(canon_alias)
            if hit is not None:
                return hit, "alias"
            # Fall through — alias points somewhere we don't have yet.

        # ---- Tier 3: conservative fuzzy --------------------------------
        if last and team_abbr:
            for c in self._by_last.get(last, []):
                if not c.team_abbrev or c.team_abbrev != team_abbr:
                    continue
                if not positions_compatible(c.position, query.position):
                    continue
                # First-name guard: shared prefix, diminutive, or strong sim.
                cf = c.first or ""
                if not first and not cf:
                    return c, "tier3"
                if first and cf and (
                    _share_prefix(first, cf, 2)
                    or _is_known_diminutive(first, cf)
                    or _token_set_similarity(canon, c.canonical or "") >= 0.80
                ):
                    return c, "tier3"

        return None, "unresolved"

    # ------------------------------------------------------------------
    # Mutators — let new players join the pool as we go.
    # ------------------------------------------------------------------

    def add(self, candidate: ResolverCandidate) -> None:
        if candidate.canonical and candidate.canonical not in self._by_canonical:
            self._by_canonical[candidate.canonical] = candidate
        if candidate.last:
            self._by_last.setdefault(candidate.last, []).append(candidate)
            t2_key = (candidate.last, (candidate.first or "")[:1])
            self._by_t2.setdefault(t2_key, []).append(candidate)
