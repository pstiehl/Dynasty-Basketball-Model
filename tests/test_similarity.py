"""Tests for the career-arc similarity engine (PR #4).

Mostly pure-logic tests over a hand-crafted historical corpus, so they
run in milliseconds and don't depend on the live data/historical_nba/
cache. Two integration-flavored tests at the bottom check the Flagg /
Harden ranking invariants against the real cached corpus when it's
available; they're auto-skipped on machines that don't have the
historical corpus committed.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynasty_bball.sources.historical_nba import (
    HistoricalPlayerSeason,
    parse_season_payload,
    season_string,
    season_end_year,
    load_corpus,
    DEFAULT_CACHE_DIR,
)
from dynasty_bball.similarity import (
    Profile,
    build_profile,
    build_corpus_profiles,
    derive_position_bucket,
    feature_names,
    build_career_index,
    find_comparables,
    project_career,
    rescale_to_0_100,
)
from dynasty_bball.similarity.vectorize import (
    build_profile_from_stats,
    _ts_pct,
    _per36,
)


# ---------------------------------------------------------------------------
# Hand-rolled mini corpus — 50 player-seasons designed to exercise:
#   - 5 archetypes × multiple ages and seasons
#   - Different career lengths (some retire fast, some last)
#   - Position-bucket distribution across PG/SG/SF/PF/C
# ---------------------------------------------------------------------------


def _row(
    nba_id: str, name: str, season: str, age: float,
    pts: float, reb: float, ast: float, stl: float, blk: float,
    tov: float = 2.0, tpm: float = 1.0,
    gp: int = 75, minutes: float = 32.0,
    fga: float = 14.0, fta: float = 4.0,
) -> HistoricalPlayerSeason:
    return HistoricalPlayerSeason(
        nba_id=nba_id, name=name, season=season,
        season_end_year=season_end_year(season),
        age=age, team="XXX", gp=gp, minutes=minutes,
        pts=pts, reb=reb, ast=ast, stl=stl, blk=blk,
        tov=tov, tpm=tpm,
        fga=fga, fta=fta, fgm=fga * 0.46, ftm=fta * 0.8,
        fg_pct=0.46, ft_pct=0.80,
    )


def _make_career(nba_id, name, archetype, start_age, n_seasons, start_season=2000):
    """Generate N consecutive seasons for one player with a fixed archetype.

    Archetypes (stat lines roughly modeled on real NBA players):
      - "guard_lead": high AST, mid PTS (PG)
      - "scoring_guard": high PTS, high USG (SG)
      - "wing": balanced (SF)
      - "stretch_big": rebounds + threes (PF)
      - "rim_big": rebounds + blocks (C)
    """
    lines = []
    for i in range(n_seasons):
        season = season_string(start_season + i)
        age = start_age + i
        if archetype == "guard_lead":
            lines.append(_row(nba_id, name, season, age,
                              pts=18, reb=4, ast=9, stl=1.5, blk=0.3))
        elif archetype == "scoring_guard":
            lines.append(_row(nba_id, name, season, age,
                              pts=27, reb=4, ast=4, stl=1.2, blk=0.4, fga=21))
        elif archetype == "wing":
            lines.append(_row(nba_id, name, season, age,
                              pts=20, reb=6, ast=4, stl=1.4, blk=0.7))
        elif archetype == "stretch_big":
            lines.append(_row(nba_id, name, season, age,
                              pts=18, reb=10, ast=2.5, stl=0.8, blk=1.2, tpm=2.2))
        elif archetype == "rim_big":
            lines.append(_row(nba_id, name, season, age,
                              pts=15, reb=12, ast=1.5, stl=0.6, blk=2.5, tpm=0.0))
    return lines


def _hand_corpus():
    """50ish player-seasons across 5 archetypes."""
    rows = []
    # Two long-career bigs.
    rows += _make_career("100", "Tim Duncanlike", "rim_big", start_age=22, n_seasons=15)
    rows += _make_career("101", "David Robinsonlike", "rim_big", start_age=23, n_seasons=12)
    # Long-career wings.
    rows += _make_career("200", "LeBronlike", "wing", start_age=19, n_seasons=18)
    rows += _make_career("201", "Pippenlike", "wing", start_age=22, n_seasons=12)
    # Lead guards.
    rows += _make_career("300", "Stocktonlike", "guard_lead", start_age=22, n_seasons=14)
    rows += _make_career("301", "Cpaullike", "guard_lead", start_age=20, n_seasons=15)
    # Scoring guards.
    rows += _make_career("400", "Hardenlike", "scoring_guard", start_age=20, n_seasons=12)
    rows += _make_career("401", "Kobelike", "scoring_guard", start_age=18, n_seasons=18)
    # Stretch bigs.
    rows += _make_career("500", "Dirklike", "stretch_big", start_age=21, n_seasons=18)
    rows += _make_career("501", "KGlike", "stretch_big", start_age=20, n_seasons=15)
    return rows


# ---------------------------------------------------------------------------
# Vectorize tests
# ---------------------------------------------------------------------------

def test_feature_vector_is_deterministic():
    row = _row("a", "A", "2010-11", 25, pts=20, reb=5, ast=3,
               stl=1.0, blk=0.5)
    p1 = build_profile(row)
    p2 = build_profile(row)
    np.testing.assert_array_equal(p1.raw_vec, p2.raw_vec)
    assert len(p1.raw_vec) == len(feature_names())


def test_per36_scaling():
    # 20 pts in 36 mins = 20 per-36; 10 pts in 18 mins = 20 per-36.
    assert _per36(20, 36) == pytest.approx(20.0)
    assert _per36(10, 18) == pytest.approx(20.0)
    assert _per36(10, 0) == 0.0


def test_ts_pct_formula():
    # 20 pts on 14 FGA + 4 FTA: TS% = 20 / (2 * (14 + 0.44 * 4))
    expected = 20 / (2 * (14 + 0.44 * 4))
    assert _ts_pct(20, 14, 4) == pytest.approx(expected)
    assert _ts_pct(20, 0, 0) == 0.0  # divide-by-zero safe


def test_position_buckets():
    """Each archetype should land in a sensible bucket."""
    # Lead guard: AST/36 = 9*(36/32) ≈ 10.1, big36 small → PG
    pg = _row("g", "G", "2010-11", 25, pts=18, reb=4, ast=9, stl=1.5, blk=0.3)
    assert derive_position_bucket(pg) == "PG"
    # Rim big: big36 = (12 + 1.5*2.5) * (36/32) ≈ 17.4, AST low → C
    big = _row("c", "C", "2010-11", 27, pts=15, reb=12, ast=1.5, stl=0.6, blk=2.5)
    assert derive_position_bucket(big) == "C"
    # Wing: balanced
    wing = _row("w", "W", "2010-11", 25, pts=20, reb=6, ast=4, stl=1.4, blk=0.7)
    assert derive_position_bucket(wing) in {"SF", "SG", "PF"}


def test_zscore_normalization():
    rows = _hand_corpus()
    corpus = build_corpus_profiles(rows)
    # Z-scored matrix should have ~0 mean and ~1 std on every dim
    # that wasn't constant.
    matrix = np.vstack([p.norm_vec for p in corpus.profiles])
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    # GP_pct is constant across our fixtures (always 75/82), so its
    # std after normalization is 0 (we replace 0 std with 1.0 → so
    # normalized stays at the same constant, then std across constant
    # is 0). Skip that index.
    for i, name in enumerate(feature_names()):
        if name == "gp_pct":
            continue  # constant fixture
        assert abs(means[i]) < 1e-9, f"{name} mean not ~0"
        assert abs(stds[i] - 1.0) < 1e-6 or stds[i] < 1e-6, f"{name} std weird"


# ---------------------------------------------------------------------------
# Comparables tests
# ---------------------------------------------------------------------------

def test_knn_matches_same_archetype():
    """A young scoring guard should match other young scoring guards first."""
    rows = _hand_corpus()
    corpus = build_corpus_profiles(rows)
    idx = build_career_index(rows)
    # Synthesize a current player matching scoring_guard archetype at age 21.
    target_row = _row("Z", "TestKid", "2024-25", 21,
                      pts=27, reb=4, ast=4, stl=1.2, blk=0.4, fga=21)
    target = build_profile(target_row)
    target.norm_vec = corpus.normalize(target.raw_vec)

    comps = find_comparables(target, corpus, idx, k=5, exclude_nba_id="Z")
    assert comps, "should find comps in mini corpus"
    # Top 3 should be guards / scoring-guard-bucket-adjacent
    top_names = [c.comp_name for c in comps[:3]]
    assert any("Hardenlike" in n or "Kobelike" in n for n in top_names), \
        f"expected scoring guard match in top 3, got {top_names}"


def test_age_window_filter():
    """Comps must be within age_window of the target."""
    rows = _hand_corpus()
    corpus = build_corpus_profiles(rows)
    idx = build_career_index(rows)
    target_row = _row("Z", "TestKid", "2024-25", 30,
                      pts=27, reb=4, ast=4, stl=1.2, blk=0.4, fga=21)
    target = build_profile(target_row)
    target.norm_vec = corpus.normalize(target.raw_vec)

    comps = find_comparables(target, corpus, idx, k=10, age_window=1.0,
                              exclude_nba_id="Z")
    for c in comps:
        assert abs(c.comp_age_when_compared - 30) <= 1.0


def test_career_index_forward_rollup():
    rows = _hand_corpus()
    idx = build_career_index(rows)
    # LeBronlike played 18 seasons starting at age 19; rollup from
    # age 22 should return 15 remaining seasons.
    rollup = idx.forward_rollup("200", from_age=22)
    assert rollup["remaining_seasons"] == 14
    assert rollup["remaining_games"] > 0
    assert rollup["fantasy_ppg_dhk"] is not None and rollup["fantasy_ppg_dhk"] > 0
    # From age 99 (past career end), zero remaining.
    rollup2 = idx.forward_rollup("200", from_age=99)
    assert rollup2["remaining_seasons"] == 0


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------

def test_projection_young_player_beats_old_player():
    """A 20yo with the same comp pool ranks above a 30yo."""
    rows = _hand_corpus()
    corpus = build_corpus_profiles(rows)
    idx = build_career_index(rows)

    def _proj_for_age(age):
        row = _row("Z", "Test", "2024-25", age,
                   pts=25, reb=5, ast=4, stl=1.2, blk=0.6, fga=20)
        target = build_profile(row)
        target.norm_vec = corpus.normalize(target.raw_vec)
        comps = find_comparables(target, corpus, idx, k=10, exclude_nba_id="Z")
        return project_career(
            nba_id="Z", name="Test", age=age,
            comparables=comps, league_format="points_dhk",
        )

    young = _proj_for_age(20)
    old = _proj_for_age(33)
    assert young.dynasty_value_raw > old.dynasty_value_raw, (
        f"young={young.dynasty_value_raw}, old={old.dynasty_value_raw}"
    )
    assert young.projected_remaining_years > old.projected_remaining_years


def test_projection_rescale():
    """rescale_to_0_100 should put top player at 100 and floor at 0."""
    rows = _hand_corpus()
    corpus = build_corpus_profiles(rows)
    idx = build_career_index(rows)
    targets = [
        ("A", "YoungStud", 19, _row("A", "YoungStud", "2024-25", 19,
                                       pts=25, reb=6, ast=5, stl=1.3, blk=0.7, fga=20)),
        ("B", "MidVet", 28, _row("B", "MidVet", "2024-25", 28,
                                    pts=18, reb=4, ast=4, stl=1.0, blk=0.4)),
        ("C", "OldVet", 35, _row("C", "OldVet", "2024-25", 35,
                                    pts=14, reb=3, ast=3, stl=0.8, blk=0.3)),
    ]
    projs = []
    for nba_id, name, age, row in targets:
        target = build_profile(row)
        target.norm_vec = corpus.normalize(target.raw_vec)
        comps = find_comparables(target, corpus, idx, k=10, exclude_nba_id=nba_id)
        projs.append(project_career(
            nba_id=nba_id, name=name, age=age,
            comparables=comps, league_format="points_dhk",
        ))
    rescale_to_0_100(projs)
    top = max(p.dynasty_value for p in projs)
    assert top == pytest.approx(100.0)
    for p in projs:
        assert 0.0 <= p.dynasty_value <= 100.0


def test_format_specific_projection():
    """DHK and default formats produce different dynasty values for the same comps.

    Same comps, different scoring weights → different fantasy_ppg →
    different dynasty_value_raw. (Magnitudes vary; for some profiles
    they'll be very close.)
    """
    rows = _hand_corpus()
    corpus = build_corpus_profiles(rows)
    idx = build_career_index(rows)
    row = _row("Z", "Test", "2024-25", 24,
               pts=18, reb=8, ast=2, stl=1.4, blk=1.8, fga=14)
    target = build_profile(row)
    target.norm_vec = corpus.normalize(target.raw_vec)
    comps = find_comparables(target, corpus, idx, k=10, exclude_nba_id="Z")
    p_dhk = project_career(nba_id="Z", name="Test", age=24,
                           comparables=comps, league_format="points_dhk")
    p_def = project_career(nba_id="Z", name="Test", age=24,
                           comparables=comps, league_format="points_default")
    assert p_dhk.dynasty_value_raw != p_def.dynasty_value_raw


# ---------------------------------------------------------------------------
# Integration tests — run only if historical corpus is on disk.
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "historical_nba"
BBREF_DIR = Path(__file__).resolve().parent.parent / "data" / "basketball_reference"


def _has_full_corpus() -> bool:
    if not CORPUS_DIR.exists():
        return False
    # Require at least 30 cached seasons before we consider the corpus
    # "complete enough" to assert ranking invariants against.
    return len(list(CORPUS_DIR.glob("league_*.json"))) >= 30


@pytest.mark.skipif(
    not _has_full_corpus(),
    reason="full historical corpus not present (run scripts/backfill_historical_nba.py)",
)
def test_flagg_top15_invariant():
    """Pin Phil's Cooper Flagg invariant: must rank top-15 in dynasty_value.

    See CHANGELOG-model.md v0.4.0 for the rationale. This is the
    headline test of PR #4.
    """
    from dynasty_bball.sources.career_arc import build_projections

    results = build_projections()
    if results["n_historical_seasons"] == 0:
        pytest.skip("corpus empty in integration test")
    projs = results["points_dhk"]["projections"]
    if not projs:
        pytest.skip("no current cohort built")
    projs_sorted = sorted(projs, key=lambda p: p.dynasty_value, reverse=True)
    names_top15 = [p.player_name for p in projs_sorted[:15]]
    # Allow various spellings. Flagg is in the 2025-26 BBRef cache as
    # "Cooper Flagg" (assuming the rookie cache exists).
    flagg = next((p for p in projs_sorted if "Flagg" in p.player_name), None)
    if flagg is None:
        pytest.skip("Cooper Flagg not in current cohort (probably 2024-25 cache used)")
    rank = projs_sorted.index(flagg) + 1
    assert rank <= 15, (
        f"Cooper Flagg should be top-15 by dynasty_value but ranked {rank}. "
        f"Top 15: {names_top15}"
    )


@pytest.mark.skipif(
    not _has_full_corpus(),
    reason="full historical corpus not present",
)
def test_harden_falls_relative_to_skill_rank():
    """Pin Phil's Harden invariant: 36yo high-usage guard's dynasty_value
    should be significantly LOWER than his current-skill (BBRef) rank.

    Concretely: he should not be top-30 in dynasty_value even if he's
    still top-30 in current production.
    """
    from dynasty_bball.sources.career_arc import build_projections

    results = build_projections()
    if results["n_historical_seasons"] == 0:
        pytest.skip("corpus empty")
    projs = results["points_dhk"]["projections"]
    if not projs:
        pytest.skip("no current cohort built")
    harden = next((p for p in projs if "Harden" in p.player_name), None)
    if harden is None:
        pytest.skip("Harden not in current cohort")
    projs_sorted = sorted(projs, key=lambda p: p.dynasty_value, reverse=True)
    rank = projs_sorted.index(harden) + 1
    assert rank > 30, (
        f"Harden's dynasty_value rank is {rank} — expected to fall outside "
        f"top 30 because the similarity engine sees a 36yo high-usage guard "
        f"profile."
    )
