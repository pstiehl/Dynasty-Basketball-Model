"""Smoke test — schema + sync + scoring end-to-end, no network.

Uses two in-memory fake sources to validate the pipeline. Mirrors
Dynasty-Football-Model's tests/smoke_test.py.

Run with: ``python tests/smoke_test.py``
"""
import os
import sys
from datetime import datetime

os.environ["DATABASE_URL"] = "sqlite:///./test_dynasty_bball.db"
if os.path.exists("./test_dynasty_bball.db"):
    os.remove("./test_dynasty_bball.db")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynasty_bball.db.session import init_db, get_session
from dynasty_bball.db.models import Player, CompositeScore
from dynasty_bball.sources.base import BaseSource, RankingRecord
from dynasty_bball.sources import REGISTRY
from dynasty_bball.sync import sync_source
from dynasty_bball.scoring import compute_composite_scores


# 6 synthetic players keyed by sleeper_id
PLAYERS = [
    ("100", "Player A", "PG"),
    ("200", "Player B", "C"),
    ("300", "Player C", "SF"),
    ("400", "Player D", "SG"),
    ("500", "Player E", "PF"),
    ("600", "Player F", "PG"),
]


class _FakeMarket(BaseSource):
    slug = "fake_market"
    name = "Fake market source"
    category = "market"
    default_weight = 1.0
    homepage = "http://example.com"

    def fetch(self):
        order = ["200", "100", "500", "400", "600", "300"]
        for rank, sid in enumerate(order, start=1):
            sid_, name, pos = next(p for p in PLAYERS if p[0] == sid)
            yield RankingRecord(
                source_slug=self.slug, sleeper_id=sid_,
                full_name=name, position=pos,
                overall_rank=rank, market_value=10000 - rank * 800,
            )


class _FakeExpert(BaseSource):
    slug = "fake_expert"
    name = "Fake expert source"
    category = "expert"
    default_weight = 1.0
    homepage = "http://example.com"

    def fetch(self):
        order = ["200", "500", "100", "300", "400", "600"]
        for rank, sid in enumerate(order, start=1):
            sid_, name, pos = next(p for p in PLAYERS if p[0] == sid)
            yield RankingRecord(
                source_slug=self.slug, sleeper_id=sid_,
                full_name=name, position=pos, overall_rank=rank,
            )


REGISTRY["fake_market"] = _FakeMarket
REGISTRY["fake_expert"] = _FakeExpert


def main():
    print("1. init_db...")
    init_db()

    print("2. sync fake_market...")
    assert sync_source("fake_market") == 6

    print("3. sync fake_expert...")
    assert sync_source("fake_expert") == 6

    print("4. compute composite scores (points_dhk)...")
    n = compute_composite_scores(league_format="points_dhk")
    assert n == 6, f"expected 6, got {n}"

    print("5. composite top:")
    top_name = None
    with get_session() as session:
        scores = (
            session.query(CompositeScore, Player)
            .join(Player)
            .order_by(CompositeScore.overall_rank)
            .all()
        )
        for cs, p in scores:
            print(
                f"   #{cs.overall_rank}  {p.full_name:10}  pos={p.position}  "
                f"score={cs.score:.2f}  tier={cs.tier}"
            )
        # Capture the top name while still inside the session.
        top_name = scores[0][1].full_name

    # Player B is #1 in both sources — should be the top composite.
    assert top_name == "Player B", f"expected Player B at top, got {top_name}"

    print("\nSMOKE TEST PASSED.")


if __name__ == "__main__":
    main()
