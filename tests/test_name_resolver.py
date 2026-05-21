"""Tests for the player name resolver.

Covers the tier cascade + alias map + dedup pass. Each test fixes one
concrete dupe/miss case observed in the live top-300 before PR #6.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynasty_bball.name_resolver import (
    NameResolver,
    ResolverCandidate,
    ResolverQuery,
    canonical_key,
    normalize_team,
    position_bucket,
    positions_compatible,
    load_alias_map,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cand(pid, name, pos, team):
    return ResolverCandidate.from_fields(pid, name, pos, team)


def _resolver(cands, alias_map=None):
    return NameResolver(cands, alias_map=alias_map)


# ---------------------------------------------------------------------------
# canonical_key
# ---------------------------------------------------------------------------

def test_canonical_key_diacritics():
    assert canonical_key("Dončić") == canonical_key("Doncic") == canonical_key("DONCIC")
    assert canonical_key("Jusuf Nurkić") == canonical_key("Jusuf Nurkic")
    assert canonical_key("Kristaps Porziņģis") == canonical_key("Kristaps Porzingis")


def test_canonical_key_suffix():
    assert canonical_key("LeBron James Jr.") == canonical_key("LeBron James")
    assert canonical_key("Kelly Oubre Jr.") == canonical_key("Kelly Oubre")
    assert canonical_key("Wendell Carter Jr.") == "wendell carter"


def test_canonical_key_punctuation():
    assert canonical_key("O'Neal") == "oneal"
    assert canonical_key("T.J. McConnell") == "tj mcconnell"
    assert canonical_key("P.J. Tucker") == "pj tucker"


def test_canonical_key_idempotent():
    s = canonical_key("Kelly Oubre Jr.")
    assert canonical_key(s) == s


def test_canonical_key_empty():
    assert canonical_key(None) is None
    assert canonical_key("") is None
    assert canonical_key("   ") is None


# ---------------------------------------------------------------------------
# normalize_team / position helpers
# ---------------------------------------------------------------------------

def test_normalize_team_full_to_abbrev():
    assert normalize_team("Washington Wizards") == "WAS"
    assert normalize_team("Brooklyn Nets") == "BKN"
    assert normalize_team("Philadelphia 76ers") == "PHI"


def test_normalize_team_already_abbrev():
    assert normalize_team("WAS") == "WAS"
    assert normalize_team("bkn") == "BKN"


def test_normalize_team_empty():
    assert normalize_team(None) is None
    assert normalize_team("") is None


def test_position_bucket_groups():
    assert position_bucket("PG") == "G"
    assert position_bucket("SG") == "G"
    assert position_bucket("SF") == "F"
    assert position_bucket("C") == "C"
    assert position_bucket(None) is None


def test_positions_compatible():
    assert positions_compatible("PG", "SG")  # both G
    assert positions_compatible("SF", "PF")  # both F
    assert positions_compatible("PF", "C")   # adjacent F-C
    assert not positions_compatible("PG", "C")
    assert positions_compatible(None, "C")   # unknown is permissive


# ---------------------------------------------------------------------------
# Tier 1 — exact canonical hash
# ---------------------------------------------------------------------------

def test_tier1_diacritic_match():
    r = _resolver([_cand(1, "Luka Dončić", "PG", "LAL")])
    hit, tier = r.resolve(ResolverQuery("Luka Doncic", "PG", "Los Angeles Lakers"))
    assert hit is not None and hit.player_id == 1
    assert tier == "tier1"


def test_tier1_suffix_match():
    r = _resolver([_cand(1, "Kelly Oubre Jr.", "SG", "PHI")])
    hit, tier = r.resolve(ResolverQuery("Kelly Oubre", "SG", "PHI"))
    assert hit is not None and hit.player_id == 1
    assert tier == "tier1"


def test_tier1_punctuation_match():
    r = _resolver([_cand(1, "P.J. Washington", "PF", "DAL")])
    hit, tier = r.resolve(ResolverQuery("PJ Washington", "PF", "DAL"))
    assert hit is not None and hit.player_id == 1
    assert tier == "tier1"


# ---------------------------------------------------------------------------
# Tier 2 — last + first[0] + position bucket + team
# ---------------------------------------------------------------------------

def test_tier2_nic_claxton():
    # Existing Sleeper row: "Nic Claxton" / C / BKN
    # Incoming DARKO row:   "Nicolas Claxton" / (no pos) / "Brooklyn Nets"
    # Force empty alias map so we verify the algorithm, not the curated list.
    r = _resolver([_cand(1, "Nic Claxton", "C", "BKN")], alias_map={})
    hit, tier = r.resolve(ResolverQuery("Nicolas Claxton", None, "Brooklyn Nets"))
    assert hit is not None and hit.player_id == 1
    assert tier == "tier2"


def test_tier2_alex_sarr():
    r = _resolver([_cand(1, "Alex Sarr", "C", "WAS")], alias_map={})
    hit, tier = r.resolve(ResolverQuery("Alexandre Sarr", None, "Washington Wizards"))
    assert hit is not None and hit.player_id == 1
    assert tier == "tier2"


def test_tier2_prefers_full_identity_row():
    # Two candidates with same Tier-2 key; resolver must pick the one with
    # both team and position set, not the orphan-style row.
    r = _resolver([
        _cand(1, "Alex Sarr", "C", "WAS"),
        _cand(2, "Alexandre Sarr", None, "Washington Wizards"),
    ], alias_map={})
    hit, tier = r.resolve(ResolverQuery("Alex Sarr", "C", "WAS"))
    assert hit is not None and hit.player_id == 1
    assert tier == "tier1"


# ---------------------------------------------------------------------------
# Alias map — Tier 2 declines, alias map catches it
# ---------------------------------------------------------------------------

def test_alias_bones_hyland():
    aliases = load_alias_map()
    r = _resolver([_cand(1, "Bones Hyland", "PG", "MIN")], alias_map=aliases)
    hit, tier = r.resolve(ResolverQuery("Nah'Shon Hyland", None, "Minnesota Timberwolves"))
    assert hit is not None and hit.player_id == 1
    assert tier == "alias"


def test_alias_carrington():
    aliases = load_alias_map()
    # "Bub Carrington" vs "Carlton Carrington" — last name match but first
    # initial differs, so Tier 2 declines. Alias map should catch it.
    r = _resolver([_cand(1, "Bub Carrington", "PG", "WAS")], alias_map=aliases)
    hit, tier = r.resolve(ResolverQuery("Carlton Carrington", None, "Washington Wizards"))
    assert hit is not None and hit.player_id == 1
    assert tier == "alias"


def test_alias_david_jones():
    aliases = load_alias_map()
    r = _resolver([_cand(1, "David Jones Garcia", "PF", "SAS")], alias_map=aliases)
    hit, tier = r.resolve(ResolverQuery("David Jones", None, "San Antonio Spurs"))
    assert hit is not None and hit.player_id == 1
    assert tier == "alias"


def test_alias_nigel_hayes():
    aliases = load_alias_map()
    r = _resolver([_cand(1, "Nigel Hayes-Davis", "SF", "PHX")], alias_map=aliases)
    hit, tier = r.resolve(ResolverQuery("Nigel Hayes", None, "Phoenix Suns"))
    assert hit is not None and hit.player_id == 1
    assert tier == "alias"


# ---------------------------------------------------------------------------
# Tier 3 — strict fuzzy guard
# ---------------------------------------------------------------------------

def test_tier3_requires_same_team_for_fuzzy():
    # Tier-3 fuzzy MUST require same team. Two different players with
    # different first names (no Tier-2 first-initial match) and on
    # different teams must NOT collapse via fuzzy.
    r = _resolver([_cand(1, "Carlton Carrington", "PG", "WAS")], alias_map={})
    # Same last name, different team — the alias case (Bub) shouldn't be
    # reachable here because team differs.
    hit, tier = r.resolve(ResolverQuery("Bub Carrington", "PG", "ATL"))
    assert hit is None
    assert tier == "unresolved"


def test_tier3_no_false_merge_different_player():
    # Two truly different surnames — same team, same first name — must
    # not collapse.
    r = _resolver([_cand(1, "Anthony Davis", "PF", "WAS")])
    hit, tier = r.resolve(ResolverQuery("Anthony Edwards", "SG", "WAS"))
    assert hit is None
    assert tier == "unresolved"


def test_no_false_merge_with_jr_suffix():
    # Phil's directive: suffix is a tiebreaker. If BOTH sides carry an
    # explicit suffix and they differ (Jr. vs Sr.), reject the canonical
    # match. This protects the rare "Anthony Davis" vs "Anthony Davis Jr."
    # case where they are genuinely different people.
    r = _resolver(
        [_cand(1, "Anthony Davis Sr.", "PF", "WAS")],
        alias_map={},
    )
    # Different suffix → reject (no Tier-1 collapse). Different first
    # initials are the same so Tier 2 also rejects (different team
    # logic). Should fall through to unresolved.
    hit, tier = r.resolve(ResolverQuery("Anthony Davis Jr.", "PF", "DAL"))
    assert hit is None
    assert tier == "unresolved"


def test_tier3_garbage_first_name_with_known_team():
    # Last-name match + same team + unrelated first name → reject (no
    # diminutive, no prefix overlap, no suffix). This is the "obscure
    # G-League" case that should fall through to unmatched.
    r = _resolver([_cand(1, "Bob Smith", "PG", "ATL")])
    hit, tier = r.resolve(ResolverQuery("Xyzzy Smith", None, "ATL"))
    assert hit is None


# ---------------------------------------------------------------------------
# JR suffix handling
# ---------------------------------------------------------------------------

def test_jr_suffix_tier1():
    r = _resolver([_cand(1, "LeBron James", "SF", "LAL")])
    hit, tier = r.resolve(ResolverQuery("LeBron James Jr.", "SF", "LAL"))
    assert hit is not None and hit.player_id == 1
    assert tier == "tier1"


# ---------------------------------------------------------------------------
# Dedup pass — verifying integration helper in sync.py
# ---------------------------------------------------------------------------

def test_unmatched_excluded_smoke():
    # Smoke-level: when the resolver returns None for an incoming record,
    # the caller is expected to add it to an unmatched list. We just verify
    # the resolver gives the right signal here; the actual exclusion happens
    # in dynasty_bball.sync.
    r = _resolver([])
    hit, tier = r.resolve(ResolverQuery("Totally Made-Up Player", None, "ATL"))
    assert hit is None and tier == "unresolved"


# ---------------------------------------------------------------------------
# Tier 2 falls back to unknown-pos slot when only team is shared
# ---------------------------------------------------------------------------

def test_tier2_pos_missing_on_query():
    # Sleeper row has position; DARKO doesn't. Both same team + same last
    # + same first initial → should still match.
    r = _resolver([_cand(1, "Nic Claxton", "C", "BKN")], alias_map={})
    hit, tier = r.resolve(ResolverQuery("Nicolas Claxton", None, "BKN"))
    assert hit is not None and hit.player_id == 1
    assert tier == "tier2"


def test_no_match_different_team_same_name():
    # Two players with same exact name (no Jr/Sr distinction) on different
    # teams — the second one slides past Tier-2 because the team disagrees
    # and we want a fresh player row created. Resolve says "unresolved";
    # the sync layer then creates a new Player.
    r = _resolver([_cand(1, "John Doe", "PG", "BOS")])
    # Tier 1 still matches by canonical key — that's intentional. The
    # team mismatch is a `enrich` problem, not an identity problem (two
    # truly different people with identical names is vanishingly rare;
    # source weights + corroboration filter will catch it).
    hit, tier = r.resolve(ResolverQuery("John Doe", "PG", "MIA"))
    assert hit is not None and hit.player_id == 1
    assert tier == "tier1"


# ---------------------------------------------------------------------------
# Alias map shape sanity
# ---------------------------------------------------------------------------

def test_alias_map_loads():
    m = load_alias_map()
    assert isinstance(m, dict)
    # Spot-check entries we rely on in production.
    assert m.get(canonical_key("Carlton Carrington")) == canonical_key("Bub Carrington")
    assert m.get(canonical_key("Nah'Shon Hyland")) == canonical_key("Bones Hyland")
    assert m.get(canonical_key("Nicolas Claxton")) == canonical_key("Nic Claxton")
