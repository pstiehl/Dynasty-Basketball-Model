"""Typer CLI — main entry point.

Usage::

    python -m dynasty_bball.cli init-db
    python -m dynasty_bball.cli sync-players
    python -m dynasty_bball.cli sync darko
    python -m dynasty_bball.cli sync-all
    python -m dynasty_bball.cli score --league-format points_dhk
    python -m dynasty_bball.cli top --n 25
    python -m dynasty_bball.cli sources
    python -m dynasty_bball.cli backtest darko --years 2022,2023 --window 3
    python -m dynasty_bball.cli league sleeper 1349496244468199424
    python -m dynasty_bball.cli managers sleeper 1349496244468199424
    python -m dynasty_bball.cli prefetch-leagues
"""
from __future__ import annotations
import json
import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select, func

from .db.session import init_db, get_session
from .db.models import Source, Player, CompositeScore, Ranking
from .sources import REGISTRY
from .sync import sync_source, sync_sleeper_players
from .scoring import compute_composite_scores
from .backtest import backtest_source


app = typer.Typer(help="Dynasty fantasy basketball composite model CLI", add_completion=False)
console = Console()


@app.command("init-db")
def cli_init_db():
    """Create all database tables."""
    init_db()
    console.print("[green]✓[/green] Database initialized.")


@app.command("sync-players")
def cli_sync_players():
    """Pull the Sleeper NBA player dictionary — required before other sources."""
    console.print("Pulling Sleeper NBA player map (may take 30s)...")
    n = sync_sleeper_players()
    console.print(f"[green]✓[/green] Upserted {n} NBA players from Sleeper.")


@app.command("sync")
def cli_sync(slug: str):
    """Sync a single source by slug. E.g. `sync darko`"""
    if slug not in REGISTRY:
        console.print(f"[red]Unknown source:[/red] {slug}")
        console.print(f"Available: {', '.join(REGISTRY.keys())}")
        raise typer.Exit(1)
    console.print(f"Syncing [cyan]{slug}[/cyan]...")
    n = sync_source(slug)
    console.print(f"[green]✓[/green] Wrote {n} ranking rows from {slug}.")


@app.command("sync-all")
def cli_sync_all():
    """Sync all registered sources (skips Sleeper player map — use sync-players)."""
    for slug in REGISTRY:
        if slug == "sleeper_players":
            continue
        console.print(f"Syncing [cyan]{slug}[/cyan]...")
        try:
            n = sync_source(slug)
            console.print(f"  [green]✓[/green] {n} rows")
        except Exception as e:
            console.print(f"  [red]✗ {e}[/red]")


@app.command("score")
def cli_score(
    league_format: str = typer.Option("points_dhk", "--league-format", "-f"),
    depth: int = typer.Option(300, "--depth"),
):
    """Compute composite scores for a league format."""
    console.print(f"Scoring [cyan]{league_format}[/cyan] (depth={depth})...")
    n = compute_composite_scores(league_format=league_format, depth=depth)
    console.print(f"[green]✓[/green] Wrote {n} composite score rows.")


@app.command("top")
def cli_top(
    n: int = typer.Option(25, "--n"),
    league_format: str = typer.Option("points_dhk", "--league-format", "-f"),
    position: str = typer.Option(None, "--position", "-p"),
):
    """Show the most recent composite top-N for a league format."""
    with get_session() as session:
        latest_ts = session.execute(
            select(func.max(CompositeScore.generated_at))
            .where(CompositeScore.league_format == league_format)
        ).scalar_one_or_none()
        if latest_ts is None:
            console.print("[yellow]No composite scores yet. Run `score` first.[/yellow]")
            raise typer.Exit(0)

        q = (
            select(CompositeScore, Player)
            .join(Player, CompositeScore.player_id == Player.id)
            .where(CompositeScore.league_format == league_format)
            .where(CompositeScore.generated_at == latest_ts)
            .order_by(CompositeScore.overall_rank)
        )
        if position:
            q = q.where(Player.position == position.upper())
        # Eagerly materialize into tuples so the rendering loop doesn't lazy-load
        # after the session closes.
        rows = [
            (
                cs.overall_rank, p.full_name, p.position, p.nba_team,
                p.age, p.years_remaining, cs.tier, cs.score,
            )
            for cs, p in session.execute(q.limit(n)).all()
        ]

    table = Table(title=f"Composite Top-{n} — {league_format} — {latest_ts:%Y-%m-%d %H:%M}")
    table.add_column("#", justify="right")
    table.add_column("Player")
    table.add_column("Pos")
    table.add_column("Team")
    table.add_column("Age", justify="right")
    table.add_column("Yrs", justify="right")
    table.add_column("Tier", justify="right")
    table.add_column("Score", justify="right")

    for overall_rank, full_name, position_, nba_team, age, yrs_left, tier, score in rows:
        table.add_row(
            str(overall_rank),
            full_name,
            position_ or "-",
            nba_team or "-",
            f"{age:.1f}" if age else "-",
            f"{yrs_left:.1f}" if yrs_left else "-",
            str(tier or "-"),
            f"{score:.2f}",
        )
    console.print(table)


@app.command("sources")
def cli_sources():
    """Show registered sources and last sync status."""
    with get_session() as session:
        rows = session.execute(select(Source).order_by(Source.slug)).scalars().all()

    table = Table(title="Sources")
    table.add_column("Slug")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Freq")
    table.add_column("Weight", justify="right")
    table.add_column("Last Sync")
    table.add_column("Status")

    if not rows:
        console.print("[yellow]No sources in DB yet. Run a `sync` to register one.[/yellow]")
        for slug, cls in REGISTRY.items():
            console.print(f"  [dim]registered (not yet synced):[/dim] {slug} — {cls.name}")
        return

    for s in rows:
        table.add_row(
            s.slug, s.name, s.category, s.update_frequency,
            f"{s.default_weight:.2f}",
            s.last_synced_at.strftime("%Y-%m-%d %H:%M") if s.last_synced_at else "-",
            s.last_sync_status or "-",
        )
    console.print(table)


@app.command("backtest")
def cli_backtest(
    source_slug: str,
    years: str = typer.Option(..., "--years"),
    window: int = typer.Option(3, "--window"),
    position: str = typer.Option(None, "--position", "-p"),
):
    """Backtest a source against actual NBA production. (Stub in PR #1.)"""
    cohort_years = [int(y.strip()) for y in years.split(",")]
    result = backtest_source(source_slug, cohort_years, window_years=window, position=position)
    if result is None:
        console.print("[yellow]Insufficient data to backtest (Production loader not in PR #1).[/yellow]")
        raise typer.Exit(0)
    console.print_json(json.dumps(result, default=str))


@app.command("inspect")
def cli_inspect(name: str):
    """Show all rankings + score history for a player by name (substring match)."""
    with get_session() as session:
        players = session.execute(
            select(Player).where(Player.full_name.ilike(f"%{name}%")).limit(5)
        ).scalars().all()
        if not players:
            console.print(f"[yellow]No players found matching '{name}'[/yellow]")
            return
        for p in players:
            console.print(
                f"\n[bold cyan]{p.full_name}[/bold cyan] "
                f"({p.position}, {p.nba_team})  sleeper_id={p.sleeper_id}  "
                f"age={p.age} yrs_left={p.years_remaining}"
            )
            rankings = session.execute(
                select(Ranking, Source)
                .join(Source, Ranking.source_id == Source.id)
                .where(Ranking.player_id == p.id)
                .order_by(Ranking.captured_at.desc())
                .limit(10)
            ).all()
            for r, s in rankings:
                console.print(
                    f"  {r.captured_at:%Y-%m-%d}  {s.slug:>20}  "
                    f"rank={r.overall_rank}  val={r.market_value}  fmt={r.league_format}"
                )


@app.command("league")
def cli_league(
    platform: str = typer.Argument(..., help="sleeper (NBA only)"),
    league_id: str = typer.Argument(...),
    league_format: str = typer.Option("points_dhk", "--league-format", "-f"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Pull a Sleeper NBA league and rate every team."""
    from .league import evaluate_sleeper_league

    if platform.lower() not in ("sleeper", "sleeper_nba"):
        console.print(f"[red]Unknown platform:[/red] {platform!r}. Use 'sleeper'.")
        raise typer.Exit(1)

    report = evaluate_sleeper_league(league_id, league_format=league_format)

    if as_json:
        console.print_json(json.dumps(report.to_dict(), default=str))
        return

    console.print(f"\n[bold cyan]{report.name}[/bold cyan]  ({report.platform} {report.league_id}, {report.league_format})")
    console.print(f"League avg roster value: [bold]{report.league_avg_score:.1f}[/bold]\n")

    rank_table = Table(title="Power rankings (by total roster value)")
    rank_table.add_column("#", justify="right")
    rank_table.add_column("Team")
    rank_table.add_column("Total", justify="right")
    rank_table.add_column("vs Avg", justify="right")
    for row in report.power_rankings:
        diff = row["vs_league_avg"]
        diff_str = f"+{diff}" if diff >= 0 else f"{diff}"
        rank_table.add_row(
            str(row["rank"]), row["display_name"],
            f"{row['total_score']:.1f}", diff_str,
        )
    console.print(rank_table)

    for t in report.teams:
        console.print(
            f"\n[bold]{t.display_name}[/bold]  total={t.total_score:.1f}  "
            f"avg={t.avg_score:.1f}  rated={t.players_evaluated}  unrated={t.players_unrated}"
        )
        if t.top_assets:
            console.print("  top 5 assets:")
            for a in t.top_assets:
                console.print(
                    f"    • {a['name']:<26} {a['position'] or '-':>3}  "
                    f"rank={a['rank']:>3}  tier=T{a['tier']}  score={a['score']:.1f}"
                )


@app.command("managers")
def cli_managers(
    platform: str = typer.Argument(...),
    league_id: str = typer.Argument(...),
    league_format: str = typer.Option("points_dhk", "--league-format", "-f"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Manager skill rankings from draft + trade history."""
    from .manager import manager_report_sleeper

    if platform.lower() not in ("sleeper", "sleeper_nba"):
        console.print(f"[red]Unknown platform:[/red] {platform!r}. Use 'sleeper'.")
        raise typer.Exit(1)

    report = manager_report_sleeper(league_id, league_format=league_format)

    if as_json:
        console.print_json(json.dumps(report, default=str))
        return

    console.print(
        f"\n[bold cyan]Manager rankings[/bold cyan]  (sleeper {league_id})  "
        f"picks={report['n_picks']}  trades={report['n_trades']}\n"
    )

    table = Table(title="Manager skill rankings")
    table.add_column("#", justify="right")
    table.add_column("Manager")
    table.add_column("Skill", justify="right")
    table.add_column("Picks", justify="right")
    table.add_column("Draft Δ (avg)", justify="right")
    table.add_column("z_draft", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Trade Δ (total)", justify="right")
    table.add_column("z_trade", justify="right")
    table.add_column("Notes")
    for m in report["managers"]:
        table.add_row(
            str(m["skill_rank"]), m["display_name"],
            f"{m['skill_score']:+.2f}",
            str(m["n_picks"]),
            f"{m['draft_delta_avg']:+.1f}" if m["n_picks"] else "-",
            f"{m['z_draft']:+.2f}" if m["n_picks"] else "-",
            str(m["n_trades"]),
            f"{m['trade_delta_total']:+.1f}" if m["n_trades"] else "-",
            f"{m['z_trade']:+.2f}" if m["n_trades"] else "-",
            ", ".join(m["notes"]) or "",
        )
    console.print(table)


@app.command("prefetch-leagues")
def cli_prefetch_leagues():
    """Run the leagues.json pre-fetcher and write JSON into dynasty_site/leagues/."""
    from pathlib import Path as _P
    import sys as _sys
    _sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "scripts"))
    import prefetch_leagues
    summary = prefetch_leagues.prefetch_all()
    console.print_json(json.dumps(summary, default=str))


if __name__ == "__main__":
    app()
