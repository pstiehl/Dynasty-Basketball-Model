"""Unit tests for the Basketball-Reference / nba_api adapter.

NO NETWORK. Uses tests/fixtures/basketball_reference_sample.json — a
curated 16-row slice of a real LeagueDashPlayerStats payload covering:

  * Pure scorers (Durant, Booker, Young, Curry, Lillard, Klay)
  * Stocks merchants (Amen / Ausar Thompson, Dyson Daniels, Camara, Herbert Jones)
  * Top stars (Jokić, Dončić, Wemby, SGA, Giannis)
  * One sub-min-games row (filter exerciser)

The whole point of this adapter is that points_dhk and points_default
produce **different** rankings — these tests pin that invariant.
"""
import json
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynasty_bball.sources.basketball_reference import (
    parse_leaguedash_payload,
    build_records,
    fantasy_ppg,
    _PlayerProduction,
    BasketballReference,
    MIN_GAMES_DEFAULT,
)


FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "basketball_reference_sample.json"
)


@pytest.fixture(scope="module")
def fixture_payload():
    with open(FIXTURE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def productions(fixture_payload):
    # min_games=0 so we keep the deliberately-low-GP row for the filter test
    # *separately*. The default-min-games run is exercised below.
    return parse_leaguedash_payload(fixture_payload, min_games=0)


def test_parse_returns_productions(fixture_payload):
    prods = parse_leaguedash_payload(fixture_payload, min_games=0)
    assert len(prods) >= 15
    for p in prods:
        assert isinstance(p, _PlayerProduction)
        assert p.name
        assert p.nba_id
        assert p.pts is not None and p.reb is not None and p.ast is not None


def test_min_games_filter_drops_short_seasons(fixture_payload):
    """Default MIN_GAMES filter must drop the sub-floor row."""
    prods_default = parse_leaguedash_payload(
        fixture_payload, min_games=MIN_GAMES_DEFAULT
    )
    prods_unfiltered = parse_leaguedash_payload(fixture_payload, min_games=0)
    # The fixture deliberately contains a low-GP row.
    assert len(prods_default) < len(prods_unfiltered)
    for p in prods_default:
        assert p.gp is None or p.gp >= MIN_GAMES_DEFAULT


def test_dhk_and_default_diverge_on_stocks_merchant(productions):
    """Amen Thompson must rank HIGHER (lower #) in points_dhk than in points_default.

    He's the canonical stocks-merchant case: high steals + rebounds + assists,
    low scoring. DHK pays stl=2.0 / blk=2.0 / pts=0.5 — favors him. Default
    pays pts=1.0 with stl=3.0 / blk=3.0 — still rewards stocks but the
    high pts weight lifts pure scorers above him more aggressively.
    """
    recs_dhk = build_records(productions, league_format="points_dhk")
    recs_def = build_records(productions, league_format="points_default")

    def rank_of(recs, name):
        for r in recs:
            if r.full_name == name:
                return r.overall_rank
        return None

    dhk = rank_of(recs_dhk, "Amen Thompson")
    df = rank_of(recs_def, "Amen Thompson")
    assert dhk is not None and df is not None, "Amen Thompson missing from fixture"
    assert dhk < df, (
        f"Expected Amen Thompson to rank higher in points_dhk than points_default; "
        f"got dhk={dhk}, default={df}"
    )


def test_dhk_and_default_diverge_on_pure_scorer(productions):
    """A pure scorer (Devin Booker) must rank HIGHER in points_default."""
    recs_dhk = build_records(productions, league_format="points_dhk")
    recs_def = build_records(productions, league_format="points_default")

    def rank_of(recs, name):
        for r in recs:
            if r.full_name == name:
                return r.overall_rank
        return None

    dhk = rank_of(recs_dhk, "Devin Booker")
    df = rank_of(recs_def, "Devin Booker")
    assert dhk is not None and df is not None, "Devin Booker missing from fixture"
    assert df < dhk, (
        f"Expected Devin Booker to rank higher in points_default than points_dhk; "
        f"got dhk={dhk}, default={df}"
    )


def test_rankings_not_identical_across_formats(productions):
    """Sanity: the two format rankings cannot be the same sequence."""
    recs_dhk = build_records(productions, league_format="points_dhk")
    recs_def = build_records(productions, league_format="points_default")
    order_dhk = [r.full_name for r in recs_dhk]
    order_def = [r.full_name for r in recs_def]
    assert order_dhk != order_def


def test_market_value_top_is_100(productions):
    recs_dhk = build_records(productions, league_format="points_dhk")
    assert recs_dhk[0].market_value == pytest.approx(100.0, abs=0.01)
    recs_def = build_records(productions, league_format="points_default")
    assert recs_def[0].market_value == pytest.approx(100.0, abs=0.01)


def test_market_value_descends(productions):
    recs = build_records(productions, league_format="points_dhk")
    vals = [r.market_value for r in recs]
    assert vals == sorted(vals, reverse=True)


def test_required_record_fields(productions):
    recs = build_records(productions, league_format="points_dhk")
    for r in recs:
        assert r.source_slug == "basketball_reference"
        assert r.full_name
        assert r.nba_id  # NBA Stats API gives us this for free — use it
        assert r.league_format == "points_dhk"
        assert r.overall_rank is not None and r.overall_rank >= 1
        assert r.market_value is not None and 0.0 <= r.market_value <= 100.0
        # Per-game counters propagate so downstream views can show them.
        assert r.per_game_points is not None
        assert r.per_game_rebounds is not None


def test_league_format_propagated(productions):
    recs_dhk = build_records(productions, league_format="points_dhk")
    recs_def = build_records(productions, league_format="points_default")
    assert {r.league_format for r in recs_dhk} == {"points_dhk"}
    assert {r.league_format for r in recs_def} == {"points_default"}


def test_fantasy_ppg_dhk_known_value():
    """Hand-check the formula against a known stat line.

    Amen Thompson fixture line: PTS=18.3 REB=7.8 AST=5.3 STL=1.5 BLK=0.6
    TOV=2.4 FG3M=0.3 → DHK fantasy_ppg =
        18.3*0.5 + 7.8*1.0 + 5.3*1.0 + 1.5*2.0 + 0.6*2.0 + 2.4*(-1.0) + 0.3*0.5
      = 9.15 + 7.8 + 5.3 + 3.0 + 1.2 - 2.4 + 0.15
      = 24.2
    """
    p = _PlayerProduction(
        nba_id="x", name="Amen Thompson", team="HOU", age=22.0, gp=79, minutes=33,
        pts=18.3, reb=7.8, ast=5.3, stl=1.5, blk=0.6, tov=2.4, tpm=0.3,
    )
    assert fantasy_ppg(p, "points_dhk") == pytest.approx(24.2, abs=0.001)


def test_fantasy_ppg_default_known_value():
    """And the same line under points_default:

    18.3*1.0 + 7.8*1.2 + 5.3*1.5 + 1.5*3.0 + 0.6*3.0 + 2.4*(-1.0) + 0.3*0.5
    = 18.3 + 9.36 + 7.95 + 4.5 + 1.8 - 2.4 + 0.15
    = 39.66
    """
    p = _PlayerProduction(
        nba_id="x", name="Amen Thompson", team="HOU", age=22.0, gp=79, minutes=33,
        pts=18.3, reb=7.8, ast=5.3, stl=1.5, blk=0.6, tov=2.4, tpm=0.3,
    )
    assert fantasy_ppg(p, "points_default") == pytest.approx(39.66, abs=0.01)


def test_empty_payload_yields_empty():
    assert parse_leaguedash_payload({}) == []
    assert parse_leaguedash_payload({"resultSets": []}) == []
    assert parse_leaguedash_payload(
        {"resultSets": [{"headers": [], "rowSet": []}]}
    ) == []


def test_empty_productions_yields_no_records():
    assert build_records([], league_format="points_dhk") == []


def test_adapter_uses_cache(tmp_path, fixture_payload, monkeypatch):
    """Adapter must prefer the cached JSON over a live call.

    Drop the fixture into the configured cache dir and verify fetch()
    yields records without touching the network.
    """
    cache_dir = tmp_path / "bbref_cache"
    cache_dir.mkdir()
    season = "2025-26"
    with open(cache_dir / f"leaguedash_{season}.json", "w") as f:
        json.dump(fixture_payload, f)

    # Force the env var off so we don't trigger a live fetch.
    monkeypatch.delenv("DYNASTY_BBALL_BBREF_LIVE", raising=False)

    src = BasketballReference(season=season, cache_dir=cache_dir)
    records = list(src.fetch())
    src.close()

    # Both formats should be emitted.
    formats = {r.league_format for r in records}
    assert formats == {"points_dhk", "points_default"}
    # And the same number per format.
    n_dhk = sum(1 for r in records if r.league_format == "points_dhk")
    n_def = sum(1 for r in records if r.league_format == "points_default")
    assert n_dhk == n_def
    assert n_dhk > 0


def test_adapter_missing_cache_yields_nothing(tmp_path, monkeypatch):
    """If cache is absent and live fetch is disabled/fails, adapter yields
    zero records — the launcher continues without erroring."""
    monkeypatch.delenv("DYNASTY_BBALL_BBREF_LIVE", raising=False)
    # Point cache to an empty dir so the load returns None, then make
    # _live_fetch fail by monkeypatching the module attribute.
    from dynasty_bball.sources import basketball_reference as bbref_mod
    monkeypatch.setattr(bbref_mod, "_live_fetch", lambda season: None)

    src = BasketballReference(season="9999-00", cache_dir=tmp_path / "empty")
    records = list(src.fetch())
    src.close()
    assert records == []
