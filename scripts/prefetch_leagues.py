"""Pre-fetch leagues listed in leagues.json into static JSON for the site.

For each entry in `leagues.json`:
  - Fetch league + rosters via `dynasty_bball.league.evaluate_sleeper_league`.
  - Fetch drafts + trades + compute manager rankings via `dynasty_bball.manager`.
  - Write `dynasty_site/leagues/sleeper_nba-<league_id>.json`.

Also writes `dynasty_site/leagues/index.json` — a manifest the site uses
to populate the league selector.

Usage::

    python scripts/prefetch_leagues.py
    # or via the headless launcher (called automatically)
"""
from __future__ import annotations
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LEAGUES_CONFIG = REPO_ROOT / "leagues.json"
OUTPUT_DIR = REPO_ROOT / "dynasty_site" / "leagues"


def _load_config() -> list[dict]:
    if not LEAGUES_CONFIG.exists():
        return []
    with open(LEAGUES_CONFIG) as f:
        data = json.load(f) or {}
    return data.get("leagues", []) or []


def _prefetch_sleeper_nba(entry: dict) -> dict:
    from dynasty_bball.league import evaluate_sleeper_league
    from dynasty_bball.manager import manager_report_sleeper

    league_id = str(entry["league_id"])
    league_format = entry.get("league_format", "points_dhk")

    report = evaluate_sleeper_league(league_id, league_format=league_format).to_dict()
    managers = manager_report_sleeper(league_id, league_format=league_format)
    return {
        "platform": "sleeper_nba",
        "league_id": league_id,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "team_report": report,
        "manager_report": managers,
    }


def prefetch_all(output_dir: Path = OUTPUT_DIR) -> dict:
    """Pre-fetch all leagues in leagues.json. Returns summary dict.

    Writes one file per league plus an index.json manifest.
    """
    entries = _load_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict] = []
    errors: list[dict] = []

    for entry in entries:
        platform = (entry.get("platform") or "").lower()
        league_id = entry.get("league_id")
        if not league_id:
            errors.append({"entry": entry, "error": "missing league_id"})
            continue

        try:
            if platform in ("sleeper_nba", "sleeper"):
                payload = _prefetch_sleeper_nba(entry)
            else:
                errors.append({"entry": entry, "error": f"unknown platform: {platform}"})
                continue
        except Exception as e:  # noqa: BLE001
            errors.append({
                "entry": entry,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })
            continue

        slug = f"sleeper_nba-{league_id}"
        out_path = output_dir / f"{slug}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))

        manifest_entries.append({
            "slug": slug,
            "platform": "sleeper_nba",
            "league_id": str(league_id),
            "name": (payload.get("team_report") or {}).get("name") or slug,
            "n_teams": len(((payload.get("team_report") or {}).get("teams") or [])),
            "n_managers": len(((payload.get("manager_report") or {}).get("managers") or [])),
            "fetched_at": payload["fetched_at"],
        })

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "leagues": manifest_entries,
        "errors": errors,
    }
    with open(output_dir / "index.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    summary = prefetch_all()
    print(f"Pre-fetched {len(summary['leagues'])} leagues, {len(summary['errors'])} errors.")
    for L in summary["leagues"]:
        print(f"  {L['slug']:>40}  teams={L['n_teams']:>2}  managers={L['n_managers']:>2}  ({L['name']})")
    for err in summary["errors"]:
        print(f"  [error] {err['entry']}: {err['error']}")


if __name__ == "__main__":
    main()
