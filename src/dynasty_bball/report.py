"""Generate a multi-page HTML site of the dynasty basketball model.

Output structure::

    dynasty_site/
        index.html                — landing page: methodology, top divergences
        rankings.html             — full top-300 ranking table
        league.html               — paste-a-Sleeper-NBA-league-id evaluator
        sources.html              — per-source methodology
        players/<slug>.html       — per-player detail page
        assets/style.css          — shared styles
        assets/model_scores.json  — composite scores keyed by sleeper_id

Visual reference: courtconsensus.com (clean white/dark theme, NBA-orange
accents). Built from scratch; no copied code.
"""
from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
from sqlalchemy import select, func

from .db.session import get_session
from .db.models import Player, CompositeScore, Source


# ---------------------------------------------------------------------------
# Source descriptions (extend as more adapters land)
# ---------------------------------------------------------------------------

SOURCE_DESCRIPTIONS: dict[str, dict] = {
    "darko": {
        "blurb": (
            "Kostya Medvedovsky's Daily Adjusted and Regressed Kalman "
            "Optimized projections — current-season DPM impact metric "
            "plus a survival model for years remaining."
        ),
        "type": "Analytics model",
        "strength": (
            "The only free, public, transparent player-impact metric "
            "(O-DPM / D-DPM) with a built-in longevity signal. Both "
            "inputs are exactly what a dynasty model needs."
        ),
        "weakness": (
            "Hosted as a Shiny app — no documented JSON API, so we scrape "
            "the underlying WebSocket protocol. Outage / breakage risk "
            "is mitigated by a CSV fallback at data/darko/darko_dump.csv."
        ),
        "weight_justification": (
            "Default weight 1.5 — the highest in PR #1. Justified because "
            "DARKO bundles impact metric + longevity in one source, which "
            "no public alternative does. Track-record multiplier will "
            "adjust this once we have NBA production backtests."
        ),
    },
    "court_consensus": {
        "blurb": (
            "Crowdsourced ELO-based dynasty NBA rankings from "
            "courtconsensus.com. Players are ranked via head-to-head "
            "community voting (76k+ data points and growing); the ELO "
            "score is the consensus signal."
        ),
        "type": "Market / consensus",
        "strength": (
            "Anchors the model against community consensus. Rookie ranks "
            "that DARKO's longevity bonus over-inflates are sanded down "
            "by CC's voted-on ranking of those same prospects (typical "
            "CC placement for hyped rookies: #30–60, not the top 15)."
        ),
        "weakness": (
            "Crowd-derived — prone to recency bias and hype. Not an "
            "impact metric or a longevity model; complements DARKO "
            "rather than competing with it."
        ),
        "weight_justification": (
            "Default weight 1.0 — standard consensus baseline. DARKO "
            "remains heavier at 1.5 because it carries impact + "
            "longevity that CC by design lacks."
        ),
    },
    "vecenie": {
        "blurb": (
            "Sam Vecenie's Big Board (The Athletic). The de-facto "
            "industry consensus for pre-NBA prospect evaluation; ranks "
            "are loaded from a local CSV drop because the underlying "
            "board is paywalled."
        ),
        "type": "Expert (single analyst)",
        "strength": (
            "Documented strong track record on NBA draft prospect "
            "translation. The football-side analog would be Lance "
            "Zierlein."
        ),
        "weakness": (
            "Manual CSV drop — only as fresh as the last update. "
            "Rookie-signal source: players who appear ONLY on the Big "
            "Board are filtered out of the top of the composite to "
            "prevent draft-prospect squatting."
        ),
        "weight_justification": (
            "Default weight 1.3 — elevated single-analyst weight. "
            "Track-record multiplier will adjust once we have a "
            "production backtest."
        ),
    },
    "basketball_reference": {
        "blurb": (
            "Realized per-game NBA box-score production via nba_api's "
            "LeagueDashPlayerStats endpoint. The first format-aware "
            "signal in the model — fantasy_ppg is computed per league "
            "format from scoring.LEAGUE_SCORING so points_dhk and "
            "points_default produce different rankings."
        ),
        "type": "Model (realized production)",
        "strength": (
            "Ground truth. Every other source is opinion or projection; "
            "this one is what actually happened on the floor last season. "
            "Drives the divergence between scoring formats."
        ),
        "weakness": (
            "Backward-looking. A player who just broke out mid-season will "
            "be undervalued; a player who just fell off will be overvalued. "
            "DD% / TD% intentionally skipped in v1 to keep API cost low."
        ),
        "weight_justification": (
            "Default weight 1.2. Below DARKO (1.5, forward-looking impact "
            "+ longevity) because BBRef is retrospective; above Court "
            "Consensus (1.0) because it's hard ground truth rather than "
            "crowd opinion. Track-record multiplier will float this once "
            "the backtest pipeline runs."
        ),
    },
    "career_arc": {
        "blurb": (
            "KNN over a 1980-present historical NBA corpus. For each "
            "current player at age A, finds the top-20 historical "
            "player-seasons at age A±1 with matching production profile, "
            "then projects remaining-career fantasy points by aggregating "
            "those comps' actual remaining careers."
        ),
        "type": "Model (similarity)",
        "strength": (
            "The only source in the composite that produces a "
            "forward-looking, longevity-aware dynasty value grounded in "
            "actual historical career arcs. Replaces DARKO's broken "
            "survival curves (Cooper Flagg retiring at 28, etc.) as the "
            "primary longevity input."
        ),
        "weakness": (
            "Historical comps are profile-based, not narrative-based: "
            "a player with an unusual injury history (e.g. an ankle that "
            "will rob 5 years off a long-career comp) gets credit for "
            "the median comp's longevity. Censored comps (players whose "
            "careers are still active at corpus end) are treated as lower "
            "bounds, which biases projections for active-era comps slightly "
            "low. The position bucket is heuristic, derived from stats, "
            "not from official roster designations."
        ),
        "weight_justification": (
            "Default weight 1.8 — the highest in the composite. Justified "
            "because dynasty value is fundamentally about what a player "
            "will produce going forward, and this is the only source that "
            "directly projects that. DARKO's weight dropped 1.5 → 0.8 in "
            "the same release because its longevity signal is now "
            "superseded; DARKO contributes only its current-skill DPM."
        ),
    },
    "historical_nba": {
        "blurb": (
            "Every NBA player-season since 1980, pulled via "
            "nba_api.LeagueDashPlayerStats and cached as JSON under "
            "data/historical_nba/. Backs the career_arc similarity engine; "
            "does not emit Ranking rows directly."
        ),
        "type": "Reference data",
        "strength": "45 seasons of consistent box-score data.",
        "weakness": (
            "Backward-looking by definition. Pre-1980 seasons excluded "
            "because 3PT and full STL/BLK tracking weren't standardized."
        ),
        "weight_justification": (
            "N/A — backs career_arc, not weighted directly."
        ),
    },
    "sleeper_players": {
        "blurb": "Sleeper's canonical NBA player ID map. Used internally; does not contribute to scoring.",
        "type": "Reference data",
        "strength": "Links every source to a single player identity.",
        "weakness": "Not a ranking source.",
        "weight_justification": "N/A — does not contribute to composite scoring.",
    },
}

POSITION_COLOR = {
    "PG": "#ea580c",  # NBA orange
    "SG": "#dc2626",
    "SF": "#16a34a",
    "PF": "#2563eb",
    "C":  "#7c3aed",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(name: str, player_id: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return f"{s or 'player'}-{player_id}"


def _esc(s) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _pos_badge(pos: str | None) -> str:
    if not pos:
        return '<span class="pos-badge" style="background:#9ca3af">—</span>'
    color = POSITION_COLOR.get(pos.upper(), "#6b7280")
    return f'<span class="pos-badge" style="background:{color}">{_esc(pos)}</span>'


def _divergence_chip(div: int | None) -> str:
    if div is None:
        return '<span class="div-chip div-none">—</span>'
    if div >= 20:
        return f'<span class="div-chip div-up-big">▲ +{div}</span>'
    if div >= 5:
        return f'<span class="div-chip div-up">▲ +{div}</span>'
    if div <= -20:
        return f'<span class="div-chip div-down-big">▼ {div}</span>'
    if div <= -5:
        return f'<span class="div-chip div-down">▼ {div}</span>'
    return f'<span class="div-chip div-flat">{div:+d}</span>'


def _shared_css() -> str:
    return """
:root {
  --bg: #ffffff; --card: #ffffff; --border: #e5e7eb; --text: #0f172a;
  --muted: #64748b; --accent: #ea580c; --accent-dark: #c2410c;
  --hover: #fff7ed; --header-bg: #0f172a; --header-text: #f8fafc;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.55;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
header.site {
  background: var(--header-bg);
  color: var(--header-text); padding: 20px 36px;
  border-bottom: 3px solid var(--accent);
}
header.site .row { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 16px; }
header.site h1 { margin: 0; font-size: 20px; font-weight: 700; letter-spacing: -0.01em; }
header.site h1 a { color: var(--header-text); }
header.site h1 .accent { color: var(--accent); }
header.site nav a {
  color: var(--header-text); opacity: 0.75; margin-left: 22px; font-size: 14px; font-weight: 500;
}
header.site nav a:hover { opacity: 1; text-decoration: none; }
header.site nav a.active { opacity: 1; border-bottom: 2px solid var(--accent); padding-bottom: 4px; }
header.site .meta { opacity: 0.6; font-size: 12px; margin-top: 4px; }
.container { max-width: 1240px; margin: 0 auto; padding: 28px 36px; }
.container.narrow { max-width: 900px; }
h2 { color: var(--text); font-size: 22px; margin-top: 32px; font-weight: 600; }
h2 .accent { color: var(--accent); }
h3 { color: var(--text); font-size: 16px; margin-top: 22px; font-weight: 600; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px 24px; margin-bottom: 18px; }
.lede { font-size: 15px; color: var(--muted); margin: 8px 0 18px 0; max-width: 720px; }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 22px; }
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; }
.kpi .num { font-size: 24px; font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }
.kpi .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
table { width: 100%; background: var(--card); border-collapse: collapse;
  border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
th { background: #f8fafc; padding: 11px 14px; text-align: left;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); border-bottom: 1px solid var(--border); font-weight: 700; }
td { padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 14px; vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tr.player-row:hover { background: var(--hover); cursor: pointer; }
td.rank { font-weight: 700; color: var(--accent); width: 50px; }
td.name { font-weight: 600; }
td.score { font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; color: var(--accent); }
td.dpm, td.years, td.team, td.tier, td.consensus { color: var(--muted); font-variant-numeric: tabular-nums; }
.pos-badge { display: inline-block; color: white; padding: 3px 8px; border-radius: 4px;
  font-size: 11px; font-weight: 700; min-width: 32px; text-align: center; }
.controls { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  padding: 14px 18px; margin-bottom: 18px; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.controls input, .controls select { font: inherit; padding: 7px 11px;
  border: 1px solid var(--border); border-radius: 6px; background: white; }
.controls input { flex: 1; min-width: 220px; }
.stats { color: var(--muted); font-size: 13px; margin-left: auto; }
.div-chip { display: inline-block; padding: 3px 9px; border-radius: 12px; font-size: 11px;
  font-weight: 600; font-variant-numeric: tabular-nums; }
.div-up { background: #ecfdf5; color: #047857; }
.div-up-big { background: #16a34a; color: white; }
.div-down { background: #fef2f2; color: #b91c1c; }
.div-down-big { background: #dc2626; color: white; }
.div-flat { background: #f3f4f6; color: #6b7280; }
.div-none { background: #f3f4f6; color: var(--muted); font-style: italic; }
.callout { background: #fff7ed; border: 1px solid #fdba74;
  border-left: 4px solid var(--accent); border-radius: 6px; padding: 14px 18px;
  color: #7c2d12; margin: 16px 0; font-size: 14px; }
.callout strong { color: var(--accent-dark); }
.player-header { background: var(--header-bg); color: var(--header-text); padding: 28px 36px; border-bottom: 3px solid var(--accent); }
.player-header h1 { margin: 0; font-size: 28px; }
.player-header .sub { opacity: 0.75; font-size: 14px; margin-top: 4px; }
.player-header .metrics { display: flex; gap: 28px; margin-top: 18px; flex-wrap: wrap; }
.player-header .metric .num { font-size: 26px; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--accent); }
.player-header .metric .label { opacity: 0.75; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
footer { color: var(--muted); font-size: 12px; padding: 32px 40px; text-align: center; border-top: 1px solid var(--border); margin-top: 40px; }
.tag { display: inline-block; padding: 2px 9px; border-radius: 10px;
  font-size: 11px; font-weight: 600; background: #eef2ff; color: #4338ca; }
.tag.tag-market { background: #ecfdf5; color: #065f46; }
.tag.tag-aggregator { background: #eff6ff; color: #1d4ed8; }
.tag.tag-expert { background: #fef3c7; color: #92400e; }
.tag.tag-model { background: #fae8ff; color: #86198f; }
"""


def _site_header(active: str, latest_ts: datetime | None, league_format: str) -> str:
    league_label = {
        "points_dhk": "Dynasty Hoop Kings",
        "points_default": "Standard Sleeper NBA points",
    }.get(league_format, league_format)
    ts = latest_ts.strftime("%B %d, %Y at %I:%M %p UTC") if latest_ts else "—"

    def link(href, label, key):
        cls = ' class="active"' if key == active else ""
        return f'<a href="{href}"{cls}>{label}</a>'

    return f"""<header class="site">
  <div class="row">
    <div>
      <h1><a href="index.html">Dynasty Basketball <span class="accent">Model</span></a></h1>
      <div class="meta">Composite ranking · Updated {_esc(ts)} · Format: {_esc(league_label)}</div>
    </div>
    <nav>
      {link("index.html", "Rankings", "index")}
      {link("rankings.html", "Top 300", "rankings")}
      {link("league.html", "League Import", "league")}
      {link("sources.html", "Sources & Methodology", "sources")}
    </nav>
  </div>
</header>"""


def _footer() -> str:
    return (
        '<footer>'
        'Dynasty Basketball Model · open source on '
        '<a href="https://github.com/pstiehl/Dynasty-Basketball-Model">GitHub</a> · '
        'DARKO data © Kostya Medvedovsky'
        '</footer>'
    )


def _page(title: str, header_html: str, body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
{header_html}
{body_html}
{_footer()}
</body>
</html>"""


def _page_player(title: str, header_html: str, body_html: str) -> str:
    """Same as _page but relative asset path differs for the players/ subdir."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<link rel="stylesheet" href="../assets/style.css">
</head>
<body>
{header_html}
{body_html}
{_footer()}
</body>
</html>"""


def _latest_composite(session, league_format: str, limit: int = 300):
    latest_ts = session.execute(
        select(func.max(CompositeScore.generated_at))
        .where(CompositeScore.league_format == league_format)
    ).scalar_one_or_none()
    if latest_ts is None:
        return None, []
    rows = session.execute(
        select(CompositeScore, Player)
        .join(Player, CompositeScore.player_id == Player.id)
        .where(CompositeScore.league_format == league_format)
        .where(CompositeScore.generated_at == latest_ts)
        .order_by(CompositeScore.overall_rank)
        .limit(limit)
    ).all()
    return latest_ts, rows


def _all_sources(session):
    return list(session.execute(select(Source).order_by(Source.slug)).scalars().all())


# ---------------------------------------------------------------------------
# index.html
# ---------------------------------------------------------------------------

def _build_index(rows, sources, latest_ts, league_format: str) -> str:
    n_players = len(rows)
    n_sources = sum(1 for s in sources if s.last_synced_at)

    # Top 25 table
    top_html = ""
    for cs, p in rows[:25]:
        slug = _slugify(p.full_name, p.id)
        dpm_str = "—"
        # Pull DPM from breakdown if we can.
        try:
            bd = json.loads(cs.breakdown_json or "{}")
        except Exception:
            bd = {}
        # Surface years remaining if available on Player.
        yr_str = f"{p.years_remaining:.1f}" if p.years_remaining is not None else "—"
        age_str = f"{p.age:.1f}" if p.age is not None else "—"
        top_html += f"""<tr class="player-row" onclick="location='players/{slug}.html'">
<td class="rank">{cs.overall_rank}</td>
<td class="name">{_esc(p.full_name)}</td>
<td>{_pos_badge(p.position)}</td>
<td class="team">{_esc(p.nba_team or '—')}</td>
<td class="years">{age_str}</td>
<td class="years">{yr_str}</td>
<td class="tier">T{cs.tier or '—'}</td>
<td class="score">{cs.score:.1f}</td>
</tr>"""

    body = f"""<div class="container">

<h2>Dynasty Basketball <span class="accent">Top 300</span></h2>
<p class="lede">A composite dynasty NBA ranking blending public analytics
sources (DARKO impact projections + longevity, with more sources landing
each PR). Built using the same deterministic, backtest-driven weighting
model as
<a href="https://github.com/pstiehl/Dynasty-Football-Model">Dynasty-Football-Model</a>.</p>

<div class="kpi-row">
  <div class="kpi"><div class="num">{n_players:,}</div><div class="label">Players ranked</div></div>
  <div class="kpi"><div class="num">{n_sources:,}</div><div class="label">Active sources</div></div>
  <div class="kpi"><div class="num">DHK</div><div class="label">Scoring · Phil's Sleeper league</div></div>
</div>

<div class="callout"><strong>Foundation source:</strong> DARKO. The only free,
public, transparent player-impact metric with a built-in longevity model —
exactly the two inputs a dynasty model wants. See
<a href="sources.html">Sources &amp; Methodology</a> for the full picture.</div>

<h2>Top 25</h2>
<table>
<thead><tr>
  <th>#</th><th>Player</th><th>Pos</th><th>Team</th>
  <th>Age</th><th>Yrs Left</th><th>Tier</th>
  <th style="text-align:right">Value</th>
</tr></thead>
<tbody>{top_html}</tbody>
</table>

<p style="margin-top:14px"><a href="rankings.html">See the full Top 300 →</a></p>

</div>"""

    return _page(
        "Dynasty Basketball Model",
        _site_header("index", latest_ts, league_format),
        body,
    )


# ---------------------------------------------------------------------------
# rankings.html — full top-N
# ---------------------------------------------------------------------------

def _build_rankings(rows, latest_ts, league_format: str) -> str:
    rows_html = ""
    for cs, p in rows:
        slug = _slugify(p.full_name, p.id)
        cons_str = str(cs.consensus_rank) if cs.consensus_rank else '—'
        yr_str = f"{p.years_remaining:.1f}" if p.years_remaining is not None else "—"
        age_str = f"{p.age:.1f}" if p.age is not None else "—"
        rows_html += f"""<tr class="player-row" data-name="{_esc(p.full_name.lower())}" data-position="{_esc(p.position or '')}" onclick="location='players/{slug}.html'">
<td class="rank">{cs.overall_rank}</td>
<td class="name">{_esc(p.full_name)}</td>
<td>{_pos_badge(p.position)}</td>
<td class="team">{_esc(p.nba_team or '—')}</td>
<td class="years">{age_str}</td>
<td class="years">{yr_str}</td>
<td class="tier">T{cs.tier or '—'}</td>
<td class="consensus">{cons_str}</td>
<td>{_divergence_chip(cs.rank_divergence)}</td>
<td class="score">{cs.score:.1f}</td>
</tr>"""

    body = f"""<div class="container">

<div class="controls">
  <input type="text" id="search" placeholder="Search by player name…">
  <select id="pos-filter">
    <option value="">All positions</option>
    <option value="PG">PG</option><option value="SG">SG</option>
    <option value="SF">SF</option><option value="PF">PF</option>
    <option value="C">C</option>
  </select>
  <span class="stats" id="stats">{len(rows)} players</span>
</div>

<table>
<thead><tr>
  <th>#</th><th>Player</th><th>Pos</th><th>Team</th>
  <th>Age</th><th>Yrs Left</th><th>Tier</th>
  <th>Consensus</th><th>Δ</th>
  <th style="text-align:right">Value</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>

<p style="color:var(--muted);font-size:13px;margin-top:14px">
Click any row to see the per-source breakdown that produced the player's score.
</p>

</div>
<script>
const search = document.getElementById('search');
const posFilter = document.getElementById('pos-filter');
const rows = document.querySelectorAll('.player-row');
const stats = document.getElementById('stats');
function apply() {{
  const q = search.value.toLowerCase().trim();
  const pos = posFilter.value;
  let n = 0;
  rows.forEach(r => {{
    const matchName = !q || r.dataset.name.includes(q);
    const matchPos = !pos || r.dataset.position === pos;
    const show = matchName && matchPos;
    r.style.display = show ? '' : 'none';
    if (show) n++;
  }});
  stats.textContent = n + ' players';
}}
search.addEventListener('input', apply);
posFilter.addEventListener('change', apply);
</script>"""

    return _page(
        "Dynasty Basketball Model — Rankings",
        _site_header("rankings", latest_ts, league_format),
        body,
    )


# ---------------------------------------------------------------------------
# sources.html
# ---------------------------------------------------------------------------

def _build_sources_page(sources, latest_ts, league_format: str) -> str:
    cards = ""
    for s in sources:
        meta = SOURCE_DESCRIPTIONS.get(s.slug, {})
        last_sync = s.last_synced_at.strftime("%Y-%m-%d %H:%M") if s.last_synced_at else "—"
        cards += f"""<div class="card">
<h3>{_esc(s.name)} <span class="tag tag-{_esc(s.category)}">{_esc(s.category)}</span></h3>
<div style="font-size:13px;color:var(--muted);margin-bottom:10px">
  Weight: <strong>{s.default_weight:.2f}</strong> · Frequency: {_esc(s.update_frequency)} ·
  Last sync: {_esc(last_sync)} ·
  <a href="{_esc(s.url or '#')}" target="_blank">homepage</a>
</div>
<p>{_esc(meta.get("blurb") or s.notes or "")}</p>
{('<p><strong>Strength.</strong> ' + _esc(meta["strength"]) + '</p>') if meta.get("strength") else ""}
{('<p><strong>Weakness.</strong> ' + _esc(meta["weakness"]) + '</p>') if meta.get("weakness") else ""}
{('<p><strong>Weight justification.</strong> ' + _esc(meta["weight_justification"]) + '</p>') if meta.get("weight_justification") else ""}
</div>"""

    body = f"""<div class="container narrow">

<h2>Sources <span class="accent">&amp; methodology</span></h2>
<p class="lede">Every source the model uses, what it brings, and why it gets
the weight it does. The composite score is a weighted average; each source
weight is <code>default_weight × track_record_multiplier</code>. Track-record
multipliers come from backtested Spearman correlation with realized NBA
fantasy production once we have a Production loader (next PR).</p>

<h3>The deterministic weighting model</h3>
<div class="card">
<p>No hand-coded per-(source, position) overrides. No years-pro decay.
Just one number per source, optionally adjusted by a position-specific
backtest correlation. The same source can't show two different weights
for two different players in the breakdown JSON.</p>
<p>This is a direct port of
<a href="https://github.com/pstiehl/Dynasty-Football-Model/blob/main/docs/CHANGELOG-model.md">
Dynasty-Football-Model v0.10's weighting redesign</a>.</p>
</div>

<h3>Active sources</h3>
{cards}

</div>"""

    return _page(
        "Dynasty Basketball Model — Sources",
        _site_header("sources", latest_ts, league_format),
        body,
    )


# ---------------------------------------------------------------------------
# league.html — client-side Sleeper NBA league evaluator
# ---------------------------------------------------------------------------

def _build_league_page(latest_ts, league_format: str) -> str:
    body = """<div class="container">

<h2>Rate my <span class="accent">Sleeper NBA league</span></h2>
<p class="lede">Paste a Sleeper NBA league ID and the page will fetch the
rosters live, then score every team against the latest model snapshot.
Phil's <strong>Dynasty Hoop Kings</strong> is pre-baked below if you
just want to look around.</p>

<div class="card">
<div class="controls" style="margin-bottom:0">
  <input type="text" id="league-id" placeholder="Sleeper NBA league ID, e.g. 1349496244468199424"
         value="1349496244468199424">
  <button id="go" style="background:var(--accent);color:white;border:none;border-radius:6px;padding:8px 16px;cursor:pointer;font:inherit;font-weight:600">
    Evaluate
  </button>
</div>
</div>

<div id="status" style="color:var(--muted);font-size:13px;margin:10px 0"></div>
<div id="result"></div>

</div>

<script>
const SLEEPER_BASE = "https://api.sleeper.app/v1";
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

let modelScores = null;

async function ensureScores() {
  if (modelScores) return modelScores;
  statusEl.textContent = "Loading model scores...";
  const r = await fetch("assets/model_scores.json");
  modelScores = await r.json();
  return modelScores;
}

async function loadPrefetched(slug) {
  try {
    const r = await fetch(`leagues/${slug}.json`);
    if (!r.ok) return null;
    return await r.json();
  } catch (e) { return null; }
}

function renderTeam(t) {
  const top = (t.top_assets || []).map(a =>
    `<li>${a.name} <span style="color:var(--muted)">${a.position||"-"} · rank ${a.rank} · T${a.tier} · ${a.score}</span></li>`
  ).join("");
  return `<div class="card">
    <h3>${t.display_name} — total ${t.total_score.toFixed(1)} · avg ${t.avg_score.toFixed(1)}</h3>
    <div style="font-size:13px;color:var(--muted);margin-bottom:8px">
      ${t.players_evaluated} rated · ${t.players_unrated} unrated
    </div>
    <strong>Top assets</strong>
    <ol>${top}</ol>
  </div>`;
}

function renderReport(report) {
  const pr = (report.power_rankings || []).map(r =>
    `<tr><td class="rank">${r.rank}</td><td class="name">${r.display_name}</td>
     <td class="score">${r.total_score.toFixed(1)}</td>
     <td class="years">${r.vs_league_avg >= 0 ? "+" : ""}${r.vs_league_avg.toFixed(1)}</td></tr>`
  ).join("");
  const teams = (report.teams || []).map(renderTeam).join("");
  resultEl.innerHTML = `
    <h2>${report.name}</h2>
    <p class="lede">League avg roster value: <strong>${report.league_avg_score.toFixed(1)}</strong></p>
    <table>
      <thead><tr><th>#</th><th>Team</th><th style="text-align:right">Total</th><th>vs avg</th></tr></thead>
      <tbody>${pr}</tbody>
    </table>
    <div style="margin-top:22px">${teams}</div>
  `;
}

async function evaluate() {
  const id = document.getElementById("league-id").value.trim();
  if (!id) return;
  statusEl.textContent = "Checking for pre-baked league...";
  const pre = await loadPrefetched(`sleeper_nba-${id}`);
  if (pre && pre.team_report) {
    statusEl.textContent = "Loaded pre-baked snapshot.";
    renderReport(pre.team_report);
    return;
  }
  await ensureScores();
  statusEl.textContent = "Fetching league...";
  try {
    const [league, users, rosters] = await Promise.all([
      fetch(`${SLEEPER_BASE}/league/${id}`).then(r => r.json()),
      fetch(`${SLEEPER_BASE}/league/${id}/users`).then(r => r.json()),
      fetch(`${SLEEPER_BASE}/league/${id}/rosters`).then(r => r.json()),
    ]);
    const userById = {};
    (users || []).forEach(u => userById[u.user_id] = u.display_name || u.username || u.user_id);
    const teams = (rosters || []).map(r => {
      const owner = userById[r.owner_id] || `Team ${r.roster_id}`;
      const playerIds = (r.players || []).filter(Boolean);
      let total = 0, evaluated = 0, unrated = 0;
      const ranked = [];
      playerIds.forEach(pid => {
        const s = modelScores[String(pid)];
        if (s) { total += s.score; evaluated++; ranked.push(s); }
        else { unrated++; }
      });
      ranked.sort((a, b) => b.score - a.score);
      return {
        display_name: owner,
        total_score: total,
        avg_score: evaluated ? total / evaluated : 0,
        players_evaluated: evaluated,
        players_unrated: unrated,
        top_assets: ranked.slice(0, 5).map(s => ({
          name: s.name, position: s.position, rank: s.rank, tier: s.tier, score: s.score,
        })),
      };
    });
    const avg = teams.length ? teams.reduce((a, t) => a + t.total_score, 0) / teams.length : 0;
    teams.sort((a, b) => b.total_score - a.total_score);
    const pr = teams.map((t, i) => ({
      rank: i + 1, display_name: t.display_name,
      total_score: t.total_score, vs_league_avg: t.total_score - avg,
    }));
    renderReport({
      name: league.name || `Sleeper NBA league ${id}`,
      league_avg_score: avg,
      teams,
      power_rankings: pr,
    });
    statusEl.textContent = "Done.";
  } catch (e) {
    statusEl.textContent = "Failed: " + e;
  }
}

document.getElementById("go").addEventListener("click", evaluate);
document.addEventListener("DOMContentLoaded", evaluate);
</script>"""

    return _page(
        "Dynasty Basketball Model — Rate My League",
        _site_header("league", latest_ts, league_format),
        body,
    )


# ---------------------------------------------------------------------------
# players/<slug>.html
# ---------------------------------------------------------------------------

def _load_career_arc_sidecar() -> dict:
    """Load the career_arc comparables sidecar JSON if present.

    Written by the career_arc adapter during sync. Format::

        {
          "generated_at": iso,
          "current_season": "2025-26",
          "by_nba_id": {
            "<nba_id>": {
              "top_comparables": [{name, season, age, similarity,
                                    remaining_seasons, remaining_games,
                                    remaining_fp_dhk, remaining_fp_default,
                                    bucket_match, censored}, ...],
              "n_comparables": int,
              "by_format": {"points_dhk": {dynasty_value, ...},
                            "points_default": {...}}
            }, ...
          }
        }

    Returns ``{}`` if the file is missing (e.g. historical corpus not
    yet committed) so player pages still render.
    """
    path = Path("data/career_arc/comparables.json")
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _build_player_page(cs, p, all_sources, latest_ts, league_format: str,
                       career_arc_sidecar: dict | None = None) -> str:
    try:
        bd = json.loads(cs.breakdown_json or "{}")
    except Exception:
        bd = {}

    source_by_slug = {s.slug: s for s in all_sources}
    sidecar = career_arc_sidecar or {}
    entry = (sidecar.get("by_nba_id") or {}).get(p.nba_id or "", {})
    comps = entry.get("top_comparables", []) or []
    by_fmt = (entry.get("by_format") or {}).get(league_format, {})

    rows_html = ""
    for slug, info in sorted(bd.items(), key=lambda kv: -kv[1].get("weight", 0)):
        src = source_by_slug.get(slug)
        name = src.name if src else slug
        rows_html += f"""<tr>
<td class="name">{_esc(name)}</td>
<td><span class="tag tag-{_esc(info.get('category') or '')}">{_esc(info.get('category') or '')}</span></td>
<td class="years">{info.get('raw_rank') or '—'}</td>
<td class="score">{info.get('score'):.1f}</td>
<td class="years">{info.get('weight'):.2f}</td>
</tr>"""

    dpm_str = "—"
    yr_str = f"{p.years_remaining:.1f}" if p.years_remaining is not None else "—"
    age_str = f"{p.age:.1f}" if p.age is not None else "—"
    retire_str = f"{p.est_retirement_age:.1f}" if p.est_retirement_age is not None else "—"

    header_html = f"""<div class="player-header">
  <h1>{_esc(p.full_name)}</h1>
  <div class="sub">{_esc(p.position or '—')} · {_esc(p.nba_team or '—')} · Rank #{cs.overall_rank} · Tier {cs.tier}</div>
  <div class="metrics">
    <div class="metric"><div class="num">{cs.score:.1f}</div><div class="label">Composite</div></div>
    <div class="metric"><div class="num">{age_str}</div><div class="label">Age</div></div>
    <div class="metric"><div class="num">{yr_str}</div><div class="label">Yrs Remaining</div></div>
    <div class="metric"><div class="num">{retire_str}</div><div class="label">Est. retirement</div></div>
  </div>
  <div style="margin-top:14px"><a href="../rankings.html" style="color:var(--header-text);opacity:0.8;font-size:13px">← back to rankings</a></div>
</div>"""

    # Career-Arc comparables block. Only render if we have data.
    comps_html = ""
    if comps:
        rows = ""
        for c in comps:
            ppg_key = "remaining_fp_dhk" if league_format == "points_dhk" else "remaining_fp_default"
            ppg = c.get(ppg_key)
            censored_flag = " (still active)" if c.get("censored") else ""
            bucket_flag = "" if c.get("bucket_match") else " · adj. bucket"
            rows += (
                f"<tr>"
                f"<td class=\"name\">{_esc(c.get('name',''))}</td>"
                f"<td class=\"years\">{_esc(c.get('season',''))}</td>"
                f"<td class=\"years\">{c.get('age','')}</td>"
                f"<td class=\"score\">{c.get('similarity', 0):.3f}</td>"
                f"<td class=\"years\">{c.get('remaining_seasons', '')}{censored_flag}{bucket_flag}</td>"
                f"<td class=\"score\">{ppg if ppg is not None else '—'}</td>"
                f"</tr>"
            )
        dv = by_fmt.get("dynasty_value")
        py = by_fmt.get("projected_remaining_years")
        tp = by_fmt.get("projected_total_fantasy_points")
        headline = []
        if dv is not None:
            headline.append(f"dynasty_value <strong>{dv:.1f}</strong>")
        if py is not None:
            headline.append(f"projected remaining years <strong>{py:.1f}</strong>")
        if tp is not None:
            headline.append(f"projected remaining fantasy pts <strong>{int(tp):,}</strong>")
        headline_line = " · ".join(headline) if headline else ""
        comps_html = f"""
<h2>Career-Arc Comparables</h2>
<p class="lede">Top-5 most similar historical NBA player-seasons at this
age (±1), by production profile. The model's dynasty value is the
similarity-weighted projection of these careers' remaining production
(time-discounted 5%/yr).</p>
<p class="lede" style="margin-top:-6px"><em>{headline_line}</em></p>

<table>
<thead><tr>
  <th>Comparable player</th><th>Their season</th><th>Age</th>
  <th style="text-align:right">Similarity</th>
  <th>Their remaining years</th>
  <th style="text-align:right">Their remaining fppg ({_esc(league_format)})</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
"""

    body = f"""<div class="container">

<h2>Per-source breakdown</h2>
<p class="lede">How the composite was built. Each row shows the source's
normalized 0–100 score for {_esc(p.full_name)}, its raw rank within that
source's universe, and the effective weight applied.</p>

<table>
<thead><tr>
  <th>Source</th><th>Category</th><th>Raw rank</th>
  <th style="text-align:right">Score</th><th>Weight</th>
</tr></thead>
<tbody>{rows_html or '<tr><td colspan=5 style="color:var(--muted)">No source contributions yet.</td></tr>'}</tbody>
</table>

{comps_html}

</div>"""

    return _page_player(
        f"{p.full_name} — Dynasty Basketball Model",
        header_html,
        body,
    )


# ---------------------------------------------------------------------------
# generate_site
# ---------------------------------------------------------------------------

def generate_site(
    output_dir: str = "dynasty_site",
    league_format: str = "points_dhk",
    limit: int = 300,
) -> str:
    """Generate the multi-page site. Returns the absolute path to index.html."""
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "assets").mkdir(exist_ok=True)
    (out_root / "players").mkdir(exist_ok=True)
    (out_root / "leagues").mkdir(exist_ok=True)

    (out_root / "assets" / "style.css").write_text(_shared_css(), encoding="utf-8")

    with get_session() as session:
        latest_ts, rows = _latest_composite(session, league_format, limit)
        sources = _all_sources(session)

        if not rows:
            (out_root / "index.html").write_text(_page(
                "Dynasty Basketball Model — No Data",
                _site_header("index", None, league_format),
                """<div class="container narrow">
<div class="callout"><strong>No rankings have been generated yet.</strong>
Run the launcher again — make sure the sync step completes successfully.</div>
</div>""",
            ), encoding="utf-8")
            return str(out_root / "index.html")

        (out_root / "index.html").write_text(
            _build_index(rows, sources, latest_ts, league_format), encoding="utf-8"
        )
        (out_root / "rankings.html").write_text(
            _build_rankings(rows, latest_ts, league_format), encoding="utf-8"
        )
        (out_root / "sources.html").write_text(
            _build_sources_page(sources, latest_ts, league_format), encoding="utf-8"
        )

        # Unbounded model_scores.json keyed by sleeper_id (for live league.html).
        all_rows_for_json = session.execute(
            select(CompositeScore, Player)
            .join(Player, CompositeScore.player_id == Player.id)
            .where(CompositeScore.league_format == league_format)
            .where(CompositeScore.generated_at == latest_ts)
            .order_by(CompositeScore.overall_rank)
        ).all()
        scores_lookup: dict[str, dict] = {}
        for cs, p in all_rows_for_json:
            if not p.sleeper_id:
                continue
            scores_lookup[str(p.sleeper_id)] = {
                "name": p.full_name,
                "position": p.position,
                "team": p.nba_team,
                "score": round(cs.score, 2),
                "rank": cs.overall_rank,
                "tier": cs.tier,
                "position_rank": cs.position_rank,
            }
        (out_root / "assets" / "model_scores.json").write_text(
            json.dumps(scores_lookup, separators=(",", ":")), encoding="utf-8"
        )
        (out_root / "league.html").write_text(
            _build_league_page(latest_ts, league_format), encoding="utf-8"
        )

        career_arc_sidecar = _load_career_arc_sidecar()
        for cs, p in rows:
            slug = _slugify(p.full_name, p.id)
            (out_root / "players" / f"{slug}.html").write_text(
                _build_player_page(
                    cs, p, sources, latest_ts, league_format,
                    career_arc_sidecar=career_arc_sidecar,
                ),
                encoding="utf-8",
            )

    return str(out_root / "index.html")
