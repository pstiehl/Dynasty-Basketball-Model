"""League import — pull rosters from Sleeper NBA and evaluate them.

This is the KTC-style "rate my league / rate my team" feature. The point
isn't to recompute the underlying model (that's `scoring.py`); it's to
*apply* the latest composite scores to a user's actual rosters and
surface:

- Per-team total dynasty value
- Per-team position-group breakdown (PG / SG / SF / PF / C)
- Per-team top-5 assets and "weak spots"
- League-wide power rankings (teams sorted by total value)

Sleeper's NBA endpoints are the same shape as its NFL endpoints — just
swap the player dictionary and the scoring_settings/roster_positions.

Usage
-----
::

    from dynasty_bball.league import evaluate_sleeper_league

    report = evaluate_sleeper_league("1349496244468199424",
                                     league_format="points_dhk")
    print(report.power_rankings)

Or via the CLI::

    python -m dynasty_bball.cli league sleeper 1349496244468199424
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional
import httpx
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .db.session import get_session
from .db.models import Player, CompositeScore
from .config import settings


SLEEPER_BASE = "https://api.sleeper.app/v1"

# Starting positions used for "weakness" flagging. In Phil's league this
# is moot (all UTIL slots, no positional starters) but the model still
# wants a position-balance read.
_STARTING_POSITIONS = ("PG", "SG", "SF", "PF", "C")
_WEAKNESS_TIER_THRESHOLD = 3


@dataclass
class TeamReport:
    """One team's evaluation."""
    team_id: str
    display_name: str
    total_score: float
    avg_score: float
    players_evaluated: int
    players_unrated: int
    position_totals: dict[str, float] = field(default_factory=dict)
    top_assets: list[dict] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    roster: list[dict] = field(default_factory=list)


@dataclass
class LeagueReport:
    """League-wide evaluation."""
    platform: str
    league_id: str
    name: str
    league_format: str
    teams: list[TeamReport] = field(default_factory=list)
    power_rankings: list[dict] = field(default_factory=list)
    league_avg_score: float = 0.0
    scoring_settings: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "league_id": self.league_id,
            "name": self.name,
            "league_format": self.league_format,
            "league_avg_score": round(self.league_avg_score, 2),
            "scoring_settings": self.scoring_settings,
            "power_rankings": self.power_rankings,
            "teams": [
                {
                    "team_id": t.team_id,
                    "display_name": t.display_name,
                    "total_score": round(t.total_score, 2),
                    "avg_score": round(t.avg_score, 2),
                    "players_evaluated": t.players_evaluated,
                    "players_unrated": t.players_unrated,
                    "position_totals": {k: round(v, 2) for k, v in t.position_totals.items()},
                    "top_assets": t.top_assets,
                    "weaknesses": t.weaknesses,
                }
                for t in self.teams
            ],
        }


def _latest_composite_by_player(
    session: Session, league_format: str
) -> dict[int, CompositeScore]:
    latest_ts = session.execute(
        select(func.max(CompositeScore.generated_at))
        .where(CompositeScore.league_format == league_format)
    ).scalar_one_or_none()
    if latest_ts is None:
        return {}
    rows = session.execute(
        select(CompositeScore)
        .where(CompositeScore.league_format == league_format)
        .where(CompositeScore.generated_at == latest_ts)
    ).scalars().all()
    return {r.player_id: r for r in rows}


def _fetch_sleeper_league(client: httpx.Client, league_id: str) -> tuple[dict, list[dict], list[dict]]:
    league = client.get(f"{SLEEPER_BASE}/league/{league_id}").json()
    users = client.get(f"{SLEEPER_BASE}/league/{league_id}/users").json()
    rosters = client.get(f"{SLEEPER_BASE}/league/{league_id}/rosters").json()
    return league, users, rosters


def evaluate_sleeper_league(
    league_id: str,
    league_format: str = "points_dhk",
    client: Optional[httpx.Client] = None,
) -> LeagueReport:
    """Pull a Sleeper NBA league and evaluate every team against the latest model.

    Uses the latest composite_scores snapshot for the given league_format.
    Players not present in the model are counted as `players_unrated`.
    """
    own_client = client is None
    client = client or httpx.Client(
        timeout=settings.request_timeout_seconds,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
    )

    try:
        league, users, rosters = _fetch_sleeper_league(client, league_id)
    finally:
        if own_client:
            client.close()

    user_id_to_name: dict[str, str] = {}
    for u in users or []:
        user_id_to_name[u["user_id"]] = u.get("display_name") or u.get("username") or u["user_id"]

    teams_raw: list[tuple[str, str, list[str]]] = []
    for r in rosters or []:
        team_id = str(r.get("roster_id", "?"))
        owner = user_id_to_name.get(r.get("owner_id", ""), f"Team {team_id}")
        players = [str(p) for p in (r.get("players") or []) if p]
        teams_raw.append((team_id, owner, players))

    return _build_report(
        platform="sleeper_nba",
        league_id=league_id,
        league_name=league.get("name", f"Sleeper NBA league {league_id}"),
        league_format=league_format,
        teams_raw=teams_raw,
        scoring_settings=league.get("scoring_settings"),
    )


def _build_report(
    *,
    platform: str,
    league_id: str,
    league_name: str,
    league_format: str,
    teams_raw: Iterable[tuple[str, str, list[str]]],
    scoring_settings: Optional[dict] = None,
) -> LeagueReport:
    with get_session() as session:
        all_ext_ids: set[str] = set()
        teams_raw = list(teams_raw)
        for _, _, ext_ids in teams_raw:
            all_ext_ids.update(ext_ids)

        players_by_ext: dict[str, Player] = {}
        if all_ext_ids:
            rows = session.execute(
                select(Player).where(Player.sleeper_id.in_(all_ext_ids))
            ).scalars().all()
            for p in rows:
                if p.sleeper_id:
                    players_by_ext[p.sleeper_id] = p

        composite_by_pid = _latest_composite_by_player(session, league_format)

        teams: list[TeamReport] = []
        for team_id, display_name, ext_ids in teams_raw:
            roster_rows: list[dict] = []
            position_totals: dict[str, float] = {}
            best_at_pos: dict[str, dict] = {}
            evaluated = 0
            unrated = 0
            total_score = 0.0

            for ext in ext_ids:
                player = players_by_ext.get(ext)
                cs = composite_by_pid.get(player.id) if player else None
                if player is None:
                    unrated += 1
                    continue
                if cs is None:
                    unrated += 1
                    roster_rows.append({
                        "ext_id": ext, "name": player.full_name,
                        "position": player.position, "team": player.nba_team,
                        "score": None, "rank": None, "tier": None,
                    })
                    continue
                evaluated += 1
                total_score += cs.score
                roster_rows.append({
                    "ext_id": ext, "name": player.full_name,
                    "position": player.position, "team": player.nba_team,
                    "score": round(cs.score, 2),
                    "rank": cs.overall_rank,
                    "tier": cs.tier,
                    "divergence": cs.rank_divergence,
                })
                pos = player.position
                if pos:
                    position_totals[pos] = position_totals.get(pos, 0.0) + cs.score
                    if pos not in best_at_pos or cs.score > best_at_pos[pos]["score"]:
                        best_at_pos[pos] = {
                            "name": player.full_name,
                            "score": cs.score,
                            "rank": cs.overall_rank,
                            "tier": cs.tier,
                        }

            ranked_roster = sorted(
                (r for r in roster_rows if r.get("score") is not None),
                key=lambda r: r["score"],
                reverse=True,
            )
            top_assets = ranked_roster[:5]

            weaknesses: list[str] = []
            for pos in _STARTING_POSITIONS:
                best = best_at_pos.get(pos)
                if best is None:
                    # In a UTIL-only league this is informational, not a true weakness.
                    weaknesses.append(f"no rated {pos} on roster")
                elif (best.get("tier") or 99) > _WEAKNESS_TIER_THRESHOLD:
                    weaknesses.append(
                        f"weak {pos}: best is {best['name']} (Tier {best['tier']}, rank {best['rank']})"
                    )

            avg_score = (total_score / evaluated) if evaluated else 0.0
            teams.append(TeamReport(
                team_id=team_id,
                display_name=display_name,
                total_score=total_score,
                avg_score=avg_score,
                players_evaluated=evaluated,
                players_unrated=unrated,
                position_totals=position_totals,
                top_assets=top_assets,
                weaknesses=weaknesses,
                roster=roster_rows,
            ))

    league_avg_score = (
        sum(t.total_score for t in teams) / len(teams) if teams else 0.0
    )
    sorted_teams = sorted(teams, key=lambda t: t.total_score, reverse=True)
    power_rankings = [
        {
            "rank": i,
            "team_id": t.team_id,
            "display_name": t.display_name,
            "total_score": round(t.total_score, 2),
            "vs_league_avg": round(t.total_score - league_avg_score, 2),
        }
        for i, t in enumerate(sorted_teams, start=1)
    ]

    return LeagueReport(
        platform=platform,
        league_id=league_id,
        name=league_name,
        league_format=league_format,
        teams=teams,
        power_rankings=power_rankings,
        league_avg_score=league_avg_score,
        scoring_settings=scoring_settings,
    )
