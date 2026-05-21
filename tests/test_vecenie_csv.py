"""Unit tests for the Vecenie CSV adapter.

NO NETWORK. Uses tests/fixtures/vecenie_sample.csv — a tiny 5-row
sample. Verifies the adapter parses required fields, derives
market_value from rank linearly, and yields zero records when the CSV
is missing.
"""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynasty_bball.sources.vecenie import (
    Vecenie,
    parse_vecenie_rows,
)


FIXTURE_CSV = os.path.join(os.path.dirname(__file__), "fixtures", "vecenie_sample.csv")


def _load_csv_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def csv_rows():
    return _load_csv_rows(FIXTURE_CSV)


def test_parser_produces_one_record_per_row(csv_rows):
    records = parse_vecenie_rows(csv_rows, league_format="points_dhk")
    assert len(records) == len(csv_rows)


def test_required_fields_present(csv_rows):
    records = parse_vecenie_rows(csv_rows)
    for r in records:
        assert r.source_slug == "vecenie"
        assert r.full_name
        assert r.overall_rank is not None and r.overall_rank >= 1
        assert r.market_value is not None
        assert 0.0 <= r.market_value <= 100.0


def test_market_value_linear_from_rank(csv_rows):
    records = parse_vecenie_rows(csv_rows)
    # Rank 1 → 100, max rank (5) → 0.
    by_rank = {r.overall_rank: r.market_value for r in records}
    assert by_rank[1] == pytest.approx(100.0, abs=0.01)
    assert by_rank[5] == pytest.approx(0.0, abs=0.01)
    # Monotonically decreasing.
    vals = [by_rank[k] for k in sorted(by_rank)]
    assert vals == sorted(vals, reverse=True)


def test_tier_and_draft_year_carried(csv_rows):
    records = parse_vecenie_rows(csv_rows)
    by_name = {r.full_name: r for r in records}
    assert by_name["Cooper Flagg"].tier == 1
    assert by_name["Cooper Flagg"].draft_year == 2025
    assert by_name["VJ Edgecombe"].tier == 2


def test_position_normalized(csv_rows):
    records = parse_vecenie_rows(csv_rows)
    by_name = {r.full_name: r for r in records}
    assert by_name["Cooper Flagg"].position == "SF"
    assert by_name["Dylan Harper"].position == "PG"


def test_position_ranks_assigned(csv_rows):
    records = parse_vecenie_rows(csv_rows)
    # SF appears 3x in the fixture (Flagg #1, Bailey #3, Knueppel #5).
    # Their position-ranks should be 1, 2, 3 in big-board order.
    sf_records = sorted(
        [r for r in records if r.position == "SF"],
        key=lambda r: r.overall_rank,
    )
    assert [r.position_rank for r in sf_records] == [1, 2, 3]
    assert [r.full_name for r in sf_records] == ["Cooper Flagg", "Ace Bailey", "Kon Knueppel"]


def test_adapter_yields_nothing_when_csv_missing(tmp_path):
    """The launcher needs this — adapter must not raise when file absent."""
    missing = tmp_path / "vecenie_does_not_exist.csv"
    assert not missing.exists()
    adapter = Vecenie(csv_path=missing)
    assert list(adapter.fetch()) == []


def test_adapter_reads_real_csv(tmp_path):
    """End-to-end: pass the fixture CSV path to the adapter directly."""
    adapter = Vecenie(csv_path=FIXTURE_CSV)
    records = list(adapter.fetch())
    # Each row produces TWO records (one per league format).
    assert len(records) == 10  # 5 rows × 2 formats
    formats = {r.league_format for r in records}
    assert formats == {"points_dhk", "points_default"}


def test_empty_rows_yields_empty():
    assert parse_vecenie_rows([]) == []


def test_handles_missing_optional_fields():
    rows = [
        {"rank": "1", "player_name": "Test Player"},
        {"rank": "2", "player_name": "Other Player", "position": "PG"},
    ]
    records = parse_vecenie_rows(rows)
    assert len(records) == 2
    assert records[0].position is None
    assert records[0].tier is None
    assert records[1].position == "PG"
