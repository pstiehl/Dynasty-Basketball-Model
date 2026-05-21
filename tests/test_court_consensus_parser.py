"""Unit tests for the Court Consensus row parser.

NO NETWORK. Uses tests/fixtures/court_consensus_sample.json — a real
slice of the CC public Supabase players-table response, augmented with
three synthetic PICK rows so the filter can be exercised. Verifies the
parser strips PICK rows, normalizes positions, sorts by ELO desc, and
rescales market_value into 0–100.
"""
import json
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynasty_bball.sources.court_consensus import (
    parse_court_consensus_rows,
    _is_real_player,
    _normalize_position,
)


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "court_consensus_sample.json")


@pytest.fixture(scope="module")
def fixture_rows():
    with open(FIXTURE) as f:
        return json.load(f)["rows"]


def test_pick_rows_filtered_out(fixture_rows):
    records = parse_court_consensus_rows(
        fixture_rows, captured_at=datetime(2026, 5, 21), league_format="points_dhk"
    )
    names = {r.full_name for r in records}
    # All three synthetic picks must be gone.
    assert "2026 Pick 1.01" not in names
    assert "2026 Pick 1.02" not in names
    assert "2026 Early 1st" not in names
    # Picks in the source file?
    pick_count = sum(1 for r in fixture_rows if _normalize_position(r["position"]) == "PICK")
    assert len(records) == len(fixture_rows) - pick_count


def test_sorted_by_market_value_desc(fixture_rows):
    records = parse_court_consensus_rows(fixture_rows)
    vals = [r.market_value for r in records]
    assert vals == sorted(vals, reverse=True)
    # And ranks are 1..N in order.
    assert [r.overall_rank for r in records] == list(range(1, len(records) + 1))


def test_required_record_fields(fixture_rows):
    records = parse_court_consensus_rows(fixture_rows, league_format="points_dhk")
    for r in records:
        assert r.source_slug == "court_consensus"
        assert r.full_name
        assert r.position in {"PG", "SG", "SF", "PF", "C"}
        assert r.league_format == "points_dhk"
        assert r.market_value is not None
        assert 0.0 <= r.market_value <= 100.0
        assert r.overall_rank is not None and r.overall_rank >= 1
        # Position-rank is assigned for every player with a position.
        assert r.position_rank is not None and r.position_rank >= 1


def test_market_value_top_rescaled_to_100(fixture_rows):
    records = parse_court_consensus_rows(fixture_rows)
    # Top record's market_value should be exactly 100.0 (rescaled).
    assert records[0].market_value == pytest.approx(100.0, abs=0.01)
    # And the top name should be Wemby (the real fixture leader).
    assert records[0].full_name == "Victor Wembanyama"


def test_position_normalization_from_list():
    # Real CC payloads have position as a JSON array.
    assert _normalize_position(["C"]) == "C"
    assert _normalize_position(["PG", "SG"]) == "PG"
    # Stringified literal also tolerated.
    assert _normalize_position('["SF"]') == "SF"
    # None / empty.
    assert _normalize_position(None) is None
    assert _normalize_position([]) is None


def test_position_ranks_increment_per_position(fixture_rows):
    records = parse_court_consensus_rows(fixture_rows)
    # Build per-position-rank sequence.
    seen: dict[str, int] = {}
    for r in records:
        seen[r.position] = seen.get(r.position, 0) + 1
        assert r.position_rank == seen[r.position]


def test_league_format_propagated(fixture_rows):
    records_dhk = parse_court_consensus_rows(fixture_rows, league_format="points_dhk")
    records_def = parse_court_consensus_rows(fixture_rows, league_format="points_default")
    assert {r.league_format for r in records_dhk} == {"points_dhk"}
    assert {r.league_format for r in records_def} == {"points_default"}
    # Both formats should produce the same number of records (filtering
    # doesn't depend on format).
    assert len(records_dhk) == len(records_def)


def test_is_real_player_filter():
    assert _is_real_player({"name": "Cooper Flagg", "position": ["SF"]})
    assert not _is_real_player({"name": "2026 Pick 1.01", "position": ["PICK"]})
    assert not _is_real_player({"name": "", "position": ["C"]})
    # Defensive: explicit "Early 1st" name even with a valid-looking position.
    assert not _is_real_player({"name": "2026 Early 1st", "position": ["PG"]})


def test_empty_rows_yields_empty_records():
    records = parse_court_consensus_rows([])
    assert records == []
