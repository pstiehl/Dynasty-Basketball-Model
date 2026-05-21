"""DARKO source adapter.

DARKO (https://apanalytics.shinyapps.io/DARKO/) is Kostya Medvedovsky's
public NBA Daily Adjusted and Regressed Kalman Optimized projection
system. It outputs two league-wide tables:

  * ``table``       — current-season per-player DPM (defense plus-minus
    style impact metric), broken into O-DPM / D-DPM, plus per-100 stats.
  * ``surv_table``  — a survival/longevity model with estimated retirement
    age and per-year continued-playing probabilities for the next 12 years.

The site is a Shiny app with no public JSON API. We talk to it over its
WebSocket protocol, then POST the table's DataTables ajax endpoint to
get the raw rows. This is the only meaningful free source of (a) a true
impact-style player rating that is *not* paywalled (RAPM is gone,
RAPTOR is retired) and (b) a longevity signal — both of which are
exactly the right inputs for a dynasty model.

The composite "longevity-adjusted DPM" scalar we put into
``RankingRecord.market_value`` is::

    score = 50 + (dpm * 5) + (years_remaining * 2.5)
            + clamp(dpm_improvement, 0, ∞) * 5

Which puts:

    peak elite DPM      (+7, 7yr left,  +0)   ~ 50 + 35 + 17.5 + 0 ≈ 102
    solid vet           (+2, 6yr left,  +0)   ~ 50 + 10 + 15 + 0 ≈ 75
    fading vet          ( 0, 1yr left,  -1)   ~ 50 + 0 + 2.5 + 0 ≈ 52
    young rising rookie ( 0, 11yr left, +1.5) ~ 50 + 0 + 27.5 + 7.5 ≈ 85

That scalar is what the composite scoring layer normalizes 0..100. The
underlying DPM, O-DPM, D-DPM, dpm_improvement, years_remaining,
est_retirement_age fields are also persisted (on the Player row + as
Evaluations) so downstream views can show them directly.

Resilience: if the live WebSocket scrape fails (Shiny is flaky), we
fall back to a local CSV dump at ``data/darko/darko_dump.csv`` if one
exists. Without either, the adapter yields nothing — the headless
launcher then falls back to the starter pack so the site still builds.
"""
from __future__ import annotations
import asyncio
import csv
import json
import logging
import random
import re
import string
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlencode

import httpx

from .base import BaseSource, RankingRecord


log = logging.getLogger(__name__)


DARKO_URL = "https://apanalytics.shinyapps.io/DARKO/"

# Column-index references for the two DataTables payloads. These match the
# header rows captured by /tmp/darko_probe6.py (and the fixture files in
# tests/fixtures). If DARKO ever adds/removes columns these will need to
# move; until then they're stable.
TABLE_COLS = {
    "Team": 0, "Player": 1, "Experience": 2, "DPM": 3, "DPM Improvement": 4,
    "O-DPM": 5, "D-DPM": 6, "Box DPM": 7, "Box O-DPM": 8, "Box D-DPM": 9,
    "FGA/100": 10, "FG2%": 11, "FG3A/100": 12, "FG3%": 13, "FG3ARate%": 14,
    "RimFGA/100": 15, "RimFG%": 16, "FTA/100": 17, "FT%": 18, "FTARate%": 19,
    "USG%": 20, "REB/100": 21, "AST/100": 22, "AST%": 23, "BLK/100": 24,
    "BLK%": 25, "STL/100": 26, "STL%": 27, "TOV/100": 28,
}

SURV_COLS = {
    "Player": 0, "Rookie Season": 1, "Career Games": 2, "Age": 3,
    "Est. Retirement Age": 4, "Years Remaining": 5,
}

# How long to wait for Shiny to emit its data values after init. Generous
# because Shiny is slow on cold startup.
WS_QUIET_TIMEOUT = 8.0
WS_OVERALL_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# Async scrape — async/await is forced on us by the websockets lib. The
# public `fetch()` wraps it in asyncio.run().
# ---------------------------------------------------------------------------

async def _scrape_darko_async() -> tuple[list[list], list[list], list[str], list[str]]:
    """Talk to the Shiny app and return (table_rows, surv_rows, headers, surv_headers).

    Empty lists on failure (caller decides how to handle).
    """
    import websockets  # local import — keeps base CLI lighter

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        resp = await client.get(DARKO_URL)
        resp.raise_for_status()
        html = resp.text
        session_cookie = client.cookies.get("session")

    m = re.search(r"(w_[a-f0-9]{32})", html)
    if not m:
        raise RuntimeError("Could not find worker id in DARKO HTML")
    worker_id = m.group(1)
    token = "".join(random.choices(string.digits + string.ascii_letters, k=8))
    ws_url = f"wss://apanalytics.shinyapps.io/DARKO/_w_{worker_id[2:]}/{token}/websocket"
    headers = {"Cookie": f"session={session_cookie}"}

    table_rows: list[list] = []
    surv_rows: list[list] = []
    table_headers: list[str] = []
    surv_headers: list[str] = []

    async with websockets.connect(
        ws_url,
        additional_headers=headers,
        max_size=200 * 1024 * 1024,
        open_timeout=15,
        close_timeout=5,
    ) as ws:
        greeting = await asyncio.wait_for(ws.recv(), timeout=10)
        try:
            config = json.loads(greeting)["config"]
        except Exception as e:
            raise RuntimeError(f"Bad Shiny greeting: {e}; raw={greeting[:200]}") from e
        _ = config["sessionId"]
        _ = config["workerId"]

        init = {
            "method": "init",
            "data": {
                "sliderMin": [0, 40], "sliderBPM": [-5, 7],
                "chart_options": "DPM", "time_type": "game_num",
                ".clientdata_output_table_hidden": False,
                ".clientdata_output_surv_table_hidden": False,
                ".clientdata_pixelratio": 1,
                ".clientdata_url_protocol": "https:",
                ".clientdata_url_hostname": "apanalytics.shinyapps.io",
                ".clientdata_url_port": "",
                ".clientdata_url_pathname": "/DARKO/",
                ".clientdata_url_search": "",
                ".clientdata_url_hash_initial": "", ".clientdata_url_hash": "",
                ".clientdata_singletons": "", ".clientdata_allowDataUriScheme": True,
            },
        }
        await ws.send(json.dumps(init))

        # Collect messages until quiet.
        messages: list[str] = []
        loop = asyncio.get_event_loop()
        deadline = loop.time() + WS_OVERALL_TIMEOUT
        while loop.time() < deadline:
            try:
                m_ = await asyncio.wait_for(ws.recv(), timeout=WS_QUIET_TIMEOUT)
                messages.append(m_)
            except asyncio.TimeoutError:
                break
            except Exception:
                break

        # Extract ajax urls + header rows for both tables.
        ajax_urls: dict[str, str] = {}
        all_headers: dict[str, list[str]] = {}
        n_cols: dict[str, int] = {}
        for raw in messages:
            try:
                d = json.loads(raw)
            except Exception:
                continue
            if "values" not in d:
                continue
            for tname in ("table", "surv_table"):
                if tname not in d["values"]:
                    continue
                payload = d["values"][tname]
                if not isinstance(payload, dict) or "x" not in payload:
                    continue
                x = payload["x"]
                ajax = (x.get("options") or {}).get("ajax", {})
                cont = x.get("container", "")
                hdrs = re.findall(r"<th[^>]*>([^<]+)</th>", cont)
                ajax_urls[tname] = ajax.get("url")
                all_headers[tname] = hdrs
                n_cols[tname] = len(hdrs)

        if not ajax_urls:
            raise RuntimeError("DARKO Shiny session emitted no ajax URLs")

        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            client.cookies.set("session", session_cookie)
            for tname, ajax_url in ajax_urls.items():
                url = f"https://apanalytics.shinyapps.io/DARKO/{ajax_url}"
                cols = n_cols[tname]
                post = [
                    ("draw", "1"),
                    ("start", "0"),
                    ("length", "10000"),
                    ("search[value]", ""),
                    ("search[regex]", "false"),
                    ("search[caseInsensitive]", "true"),
                    ("search[smart]", "true"),
                    ("escape", "true"),
                    ("order[0][column]", "0"),
                    ("order[0][dir]", "asc"),
                ]
                for i in range(cols):
                    post += [
                        (f"columns[{i}][data]", str(i)),
                        (f"columns[{i}][name]", ""),
                        (f"columns[{i}][searchable]", "true"),
                        (f"columns[{i}][orderable]", "true"),
                        (f"columns[{i}][search][value]", ""),
                        (f"columns[{i}][search][regex]", "false"),
                    ]
                body = urlencode(post)
                r = await client.post(
                    url,
                    content=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if r.status_code != 200:
                    log.warning("DARKO %s POST returned %s", tname, r.status_code)
                    continue
                try:
                    jr = r.json()
                except Exception as e:
                    log.warning("DARKO %s POST not JSON: %s", tname, e)
                    continue
                rows = jr.get("data", []) or []
                if tname == "table":
                    table_rows = rows
                    table_headers = all_headers[tname]
                elif tname == "surv_table":
                    surv_rows = rows
                    surv_headers = all_headers[tname]

    return table_rows, surv_rows, table_headers, surv_headers


# ---------------------------------------------------------------------------
# Row parsing — kept pure (no I/O) so it's easy to test against fixtures.
# ---------------------------------------------------------------------------

def parse_darko_rows(
    table_rows: list[list],
    surv_rows: list[list],
    captured_at: Optional[datetime] = None,
) -> list[RankingRecord]:
    """Parse raw DARKO rows into RankingRecords (name-normalized join).

    Joins ``table`` (DPM + per-100 stats) with ``surv_table`` (age,
    years_remaining) by player name with light normalization. Builds the
    composite longevity-adjusted DPM scalar into ``market_value`` so the
    scoring layer's value-based path picks it up.
    """
    from ..names import normalize as _norm

    captured_at = captured_at or datetime.utcnow()

    # Build a normalized-name → surv-row index.
    surv_by_name: dict[str, list] = {}
    for r in surv_rows:
        try:
            n = _norm(r[SURV_COLS["Player"]])
        except (IndexError, TypeError):
            continue
        if n:
            surv_by_name[n] = r

    # Also a fallback: last-name match for hyphen/punctuation oddities. Cheap
    # and only used if the full-name normalize miss.
    surv_by_last: dict[str, list] = {}
    for r in surv_rows:
        try:
            full = r[SURV_COLS["Player"]]
        except IndexError:
            continue
        last = (full or "").split()[-1].lower() if full else ""
        last = re.sub(r"[^\w]", "", last)
        if last and last not in surv_by_last:
            surv_by_last[last] = r

    out: list[RankingRecord] = []
    overall_rank = 0  # we'll sort by score and re-rank later

    intermediates: list[tuple[float, dict]] = []

    for row in table_rows:
        try:
            team = row[TABLE_COLS["Team"]]
            player = row[TABLE_COLS["Player"]]
            dpm = _to_float(row[TABLE_COLS["DPM"]])
            dpm_imp = _to_float(row[TABLE_COLS["DPM Improvement"]])
            o_dpm = _to_float(row[TABLE_COLS["O-DPM"]])
            d_dpm = _to_float(row[TABLE_COLS["D-DPM"]])
            usg = _to_float(row[TABLE_COLS["USG%"]])
            reb = _to_float(row[TABLE_COLS["REB/100"]])
            ast = _to_float(row[TABLE_COLS["AST/100"]])
            blk = _to_float(row[TABLE_COLS["BLK/100"]])
            stl = _to_float(row[TABLE_COLS["STL/100"]])
            tov = _to_float(row[TABLE_COLS["TOV/100"]])
            fg3a = _to_float(row[TABLE_COLS["FG3A/100"]])
            fg3p = _to_float(row[TABLE_COLS["FG3%"]])
        except (IndexError, TypeError):
            continue

        if not player:
            continue

        # Pull longevity from surv_table.
        nkey = _norm(player)
        surv_row = surv_by_name.get(nkey)
        if surv_row is None:
            last = re.sub(r"[^\w]", "", (player.split()[-1] if player else "")).lower()
            surv_row = surv_by_last.get(last)

        age = years_left = retire_age = None
        if surv_row is not None:
            try:
                age = _to_float(surv_row[SURV_COLS["Age"]])
                retire_age = _to_float(surv_row[SURV_COLS["Est. Retirement Age"]])
                years_left = _to_float(surv_row[SURV_COLS["Years Remaining"]])
            except (IndexError, TypeError):
                pass

        # Composite longevity-adjusted DPM score.
        dpm_v = dpm if dpm is not None else 0.0
        yrs_v = years_left if years_left is not None else 0.0
        improvement_bonus = 5.0 * max(0.0, dpm_imp) if dpm_imp is not None else 0.0
        score = 50.0 + (dpm_v * 5.0) + (yrs_v * 2.5) + improvement_bonus

        # Estimate per-game counting stats from per-100 by assuming ~32 minutes
        # at ~100 possessions per 48 minutes ≈ 32/48 * 100 = 66.7 possessions
        # per game. This is rough but lets the production-based scoring layer
        # do something meaningful for the dhk format below.
        pg = 66.7 / 100.0
        per_game = {
            "rebounds": (reb or 0.0) * pg,
            "assists": (ast or 0.0) * pg,
            "steals": (stl or 0.0) * pg,
            "blocks": (blk or 0.0) * pg,
            "turnovers": (tov or 0.0) * pg,
            "threes": (fg3a * (fg3p or 0.0)) * pg if fg3a is not None else None,
        }

        # USG% is normalized 0..1 in the payload. Use it to back out an
        # approximate per-game points estimate (USG% × ~22 reference baseline).
        per_game_pts = ((usg or 0.0) * 22.0) if usg is not None else None

        intermediates.append((
            score,
            {
                "team": team, "player": player,
                "dpm": dpm, "dpm_imp": dpm_imp, "o_dpm": o_dpm, "d_dpm": d_dpm,
                "age": age, "retire_age": retire_age, "years_left": years_left,
                "per_game_pts": per_game_pts, **per_game,
            },
        ))

    # Sort descending by composite scalar — produces overall_rank.
    intermediates.sort(key=lambda t: t[0], reverse=True)
    for rank, (score, m) in enumerate(intermediates, start=1):
        out.append(RankingRecord(
            source_slug="darko",
            full_name=m["player"],
            nba_team=m["team"],
            overall_rank=rank,
            market_value=round(score, 3),
            league_format="points_dhk",
            is_dynasty=True,
            captured_at=captured_at,
            age=m["age"],
            est_retirement_age=m["retire_age"],
            years_remaining=m["years_left"],
            dpm=m["dpm"],
            dpm_improvement=m["dpm_imp"],
            o_dpm=m["o_dpm"],
            d_dpm=m["d_dpm"],
            per_game_points=m["per_game_pts"],
            per_game_rebounds=m["rebounds"],
            per_game_assists=m["assists"],
            per_game_steals=m["steals"],
            per_game_blocks=m["blocks"],
            per_game_threes=m["threes"],
            per_game_turnovers=m["turnovers"],
        ))
    return out


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CSV fallback — if the Shiny scrape fails, look for a local dump.
# ---------------------------------------------------------------------------

def _load_csv_fallback(path: Path) -> list[RankingRecord]:
    """Read a darko_dump.csv with a flexible header set and yield records."""
    if not path.exists():
        return []
    records: list[RankingRecord] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    intermediates = []
    for r in rows:
        player = r.get("Player") or r.get("player") or r.get("name")
        if not player:
            continue
        dpm = _to_float(r.get("DPM") or r.get("dpm"))
        years_left = _to_float(r.get("Years Remaining") or r.get("years_remaining"))
        dpm_imp = _to_float(r.get("DPM Improvement") or r.get("dpm_improvement"))
        score = 50.0 + ((dpm or 0.0) * 5.0) + ((years_left or 0.0) * 2.5) \
            + 5.0 * max(0.0, dpm_imp or 0.0)
        intermediates.append((score, r, player, dpm, years_left, dpm_imp))
    intermediates.sort(key=lambda t: t[0], reverse=True)
    for rank, (score, r, player, dpm, yrs, imp) in enumerate(intermediates, start=1):
        records.append(RankingRecord(
            source_slug="darko",
            full_name=player,
            nba_team=r.get("Team") or r.get("team"),
            overall_rank=rank,
            market_value=round(score, 3),
            league_format="points_dhk",
            dpm=dpm,
            years_remaining=yrs,
            dpm_improvement=imp,
            age=_to_float(r.get("Age") or r.get("age")),
        ))
    return records


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class DARKO(BaseSource):
    slug = "darko"
    name = "DARKO — Daily Adjusted and Regressed Kalman Optimized projections"
    category = "model"
    update_frequency = "daily"
    tos_compliant = True
    # Weight reduced from 1.5 → 0.8 in v0.4.0 (PR #4). DARKO is now
    # used as a current-skill / impact signal only; longevity is
    # owned by the career_arc similarity engine (default_weight=1.8),
    # which dethrones DARKO's broken survival curves (Cooper Flagg
    # retiring at 28, etc.). See docs/CHANGELOG-model.md v0.4.0.
    default_weight = 0.8
    homepage = "https://apanalytics.shinyapps.io/DARKO/"
    notes = (
        "Kostya Medvedovsky's public NBA impact projection system. "
        "Combines current-season DPM (defense-plus-minus style impact) "
        "with a survival/longevity model. Highest weight in the composite "
        "because it's the only free, public, transparent player-impact "
        "rating with a built-in longevity signal — exactly the inputs a "
        "dynasty model needs. Scraped via Shiny WebSocket protocol."
    )

    CSV_FALLBACK = Path("data/darko/darko_dump.csv")

    def fetch(self) -> Iterator[RankingRecord]:
        try:
            table_rows, surv_rows, _, _ = asyncio.run(_scrape_darko_async())
        except Exception as e:
            log.warning("DARKO live scrape failed: %s — falling back to CSV", e)
            yield from _load_csv_fallback(self.CSV_FALLBACK)
            return

        if not table_rows:
            log.warning("DARKO live scrape yielded zero rows — falling back to CSV")
            yield from _load_csv_fallback(self.CSV_FALLBACK)
            return

        yield from parse_darko_rows(table_rows, surv_rows, captured_at=datetime.utcnow())
