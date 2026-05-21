"""Unit tests for the DARKO row parser.

NO NETWORK. Uses tests/fixtures/darko_sample.json — a slice of the real
DARKO Shiny response captured during scaffold work. Verifies the
adapter's row parser turns the DataTables JSON into well-formed
RankingRecords that join the DPM table with the survival table by
normalized player name.
"""
import json
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynasty_bball.sources.darko import parse_darko_rows


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "darko_sample.json")


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURE) as f:
        return json.load(f)


def test_parser_returns_one_record_per_table_row(fixture):
    records = parse_darko_rows(
        fixture["table_rows"], fixture["surv_rows"],
        captured_at=datetime(2026, 5, 20),
    )
    # One record per row in the DPM table.
    assert len(records) == len(fixture["table_rows"])


def test_records_have_required_fields(fixture):
    records = parse_darko_rows(fixture["table_rows"], fixture["surv_rows"])
    for r in records:
        assert r.source_slug == "darko"
        assert r.full_name
        assert r.league_format == "points_dhk"
        # market_value is the composite scalar we built.
        assert r.market_value is not None
        assert r.overall_rank is not None and r.overall_rank >= 1


def test_records_sorted_by_market_value_desc(fixture):
    records = parse_darko_rows(fixture["table_rows"], fixture["surv_rows"])
    values = [r.market_value for r in records]
    assert values == sorted(values, reverse=True)
    # And ranks are assigned 1..N in order.
    assert [r.overall_rank for r in records] == list(range(1, len(records) + 1))


def test_dpm_carried_through(fixture):
    records = parse_darko_rows(fixture["table_rows"], fixture["surv_rows"])
    # At least one record should carry a non-None DPM (every active DARKO
    # row has one).
    assert any(r.dpm is not None for r in records)
    # And the O-DPM + D-DPM should at least exist on some.
    assert any(r.o_dpm is not None for r in records)
    assert any(r.d_dpm is not None for r in records)


def test_survival_join_lands_for_known_player(fixture):
    """Pick a player who exists in both tables and verify the join."""
    # Build {name -> surv row} for the fixture.
    surv_names = {row[0] for row in fixture["surv_rows"]}
    dpm_names = {row[1] for row in fixture["table_rows"]}
    common = (surv_names & dpm_names)
    assert common, "fixture sanity: at least one player should be in both tables"

    records = parse_darko_rows(fixture["table_rows"], fixture["surv_rows"])
    # The matched players should pick up age + years_remaining.
    matched = [r for r in records if r.full_name in common]
    assert matched, "no records produced for shared players"
    assert any(r.age is not None for r in matched)
    assert any(r.years_remaining is not None for r in matched)


def test_handles_empty_survival_gracefully():
    """If surv_table is empty, records are still produced (just no longevity)."""
    with open(FIXTURE) as f:
        fixture = json.load(f)
    records = parse_darko_rows(fixture["table_rows"], surv_rows=[])
    assert len(records) == len(fixture["table_rows"])
    # All longevity fields None.
    assert all(r.age is None for r in records)
    assert all(r.years_remaining is None for r in records)
    # market_value still computed (DPM alone).
    assert all(r.market_value is not None for r in records)
