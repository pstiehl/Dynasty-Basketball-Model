"""Player-season vectorization.

Turn a player's box-score line into a profile vector suitable for
cosine / Euclidean comparison against the historical corpus. The
vector is built from:

  * **Production rates** (per-36 PTS, REB, AST, STL, BLK, 3PM, TOV).
    Per-36 instead of per-game so part-time players in the historical
    corpus aren't penalized purely for low minutes — we're matching
    on *style*, not playing-time accident.
  * **Usage proxies** — FGA/36, FTA/36. Tells us whether the player
    is a primary option or a low-usage role player.
  * **Efficiency** — TS% (true shooting). The single most
    information-dense efficiency stat.
  * **Durability** — GP/82, MIN/G. Penalizes injury-prone seasons
    when matching for dynasty projection (we want healthy comps).

All features are z-score normalized against the full corpus per
feature so cosine/Euclidean distances are scale-invariant. Z-score
mean and std are computed once at corpus build time and stored on
the ``CorpusProfiles`` container.

Position bucket
---------------
Historical box-score rows from ``LeagueDashPlayerStats`` don't carry
position. Rather than burn 20K extra API calls on ``CommonPlayerInfo``
to label every historical player, we *derive* a play-style bucket
from their stats:

  * High AST/36 → guard-leaning (PG, SG)
  * High REB+BLK/36 → big-leaning (PF, C)
  * Mid-AST + mid-REB + decent 3PT volume → wing (SF, PF)

The buckets aren't a strict label — Phil's spec says "cross-bucket
allowed for combo wings" — so we use them for *soft* filtering in
the KNN: same bucket gets a small similarity bonus, adjacent buckets
are still eligible.

College / rookie hooks
----------------------
``vectorize_college_season(...)`` is a stub kept here so PR #5 (the
rookie/college engine) lands without restructuring this module. The
intent: pull college per-game stats from sports-reference, vectorize
in the same space (after a simple translation factor for pace and
league quality), then find NBA comps for current draft prospects.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# numpy is already on the system (used by nba_api transitively); we
# keep similarity math here in pure numpy for speed and determinism.
import numpy as np

from ..sources.historical_nba import HistoricalPlayerSeason


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# Order matters — the vector dimensions are positional. Don't reorder
# without rebuilding any persisted corpus statistics.
FEATURE_NAMES: tuple[str, ...] = (
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "stl_per36",
    "blk_per36",
    "tpm_per36",
    "tov_per36",
    "fga_per36",
    "fta_per36",
    "ts_pct",
    "gp_pct",          # GP / 82
    "mpg",             # minutes per game (raw, not per-36 since IT is the durability signal)
)


def feature_names() -> tuple[str, ...]:
    return FEATURE_NAMES


# Position buckets — declared here so the comparables engine can
# compute bucket adjacency without re-implementing the mapping.
POSITION_BUCKETS: tuple[str, ...] = ("PG", "SG", "SF", "PF", "C")

# Two buckets are "adjacent" if they're within one position slot on
# the PG → C axis. Combo wings (SG↔SF, SF↔PF) are the most common
# cross-bucket matches in basketball.
ADJACENT_BUCKETS = {
    "PG": {"PG", "SG"},
    "SG": {"PG", "SG", "SF"},
    "SF": {"SG", "SF", "PF"},
    "PF": {"SF", "PF", "C"},
    "C":  {"PF", "C"},
}


# ---------------------------------------------------------------------------
# Profile container
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    """A single player-season's profile vector, plus identity fields."""
    nba_id: str
    name: str
    season: str
    season_end_year: int
    age: float
    team: Optional[str]
    position_bucket: str
    # Raw (un-normalized) feature vector. Length == len(FEATURE_NAMES).
    raw_vec: np.ndarray
    # Z-score normalized vector (filled by zscore_normalize).
    norm_vec: Optional[np.ndarray] = None
    # Source row — carry it through so the projection layer can compute
    # fantasy points on demand without rejoining tables.
    season_row: Optional[HistoricalPlayerSeason] = None


@dataclass
class CorpusProfiles:
    """Bundle of profiles + corpus-wide normalization statistics."""
    profiles: list[Profile]
    feature_means: np.ndarray
    feature_stds: np.ndarray

    def normalize(self, raw_vec: np.ndarray) -> np.ndarray:
        """Z-score a single raw vector using the corpus statistics.

        Used to project NEW players (e.g. current-season Cooper Flagg)
        into the same normalized space as the historical corpus.
        """
        std = np.where(self.feature_stds > 1e-9, self.feature_stds, 1.0)
        return (raw_vec - self.feature_means) / std


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _per36(stat: float, mpg: float) -> float:
    if mpg is None or mpg <= 0:
        return 0.0
    return stat * (36.0 / mpg)


def _ts_pct(pts: float, fga: float, fta: float) -> float:
    """True shooting percentage.

    TS% = PTS / (2 * (FGA + 0.44 * FTA)). Returns 0 if denominator
    is zero (skips garbage rows).
    """
    denom = 2.0 * (fga + 0.44 * fta)
    if denom <= 0:
        return 0.0
    return pts / denom


def build_profile(
    row: HistoricalPlayerSeason,
    position_bucket: Optional[str] = None,
) -> Profile:
    """Turn one HistoricalPlayerSeason into an un-normalized Profile."""
    mpg = row.minutes  # already per-game from LeagueDashPlayerStats
    raw_vec = np.array([
        _per36(row.pts, mpg),
        _per36(row.reb, mpg),
        _per36(row.ast, mpg),
        _per36(row.stl, mpg),
        _per36(row.blk, mpg),
        _per36(row.tpm, mpg),
        _per36(row.tov, mpg),
        _per36(row.fga, mpg),
        _per36(row.fta, mpg),
        _ts_pct(row.pts, row.fga, row.fta),
        min(1.0, row.gp / 82.0),
        mpg,
    ], dtype=np.float64)
    bucket = position_bucket or derive_position_bucket(row)
    return Profile(
        nba_id=row.nba_id,
        name=row.name,
        season=row.season,
        season_end_year=row.season_end_year,
        age=row.age,
        team=row.team,
        position_bucket=bucket,
        raw_vec=raw_vec,
        season_row=row,
    )


# ---------------------------------------------------------------------------
# Position bucket derivation
# ---------------------------------------------------------------------------

def derive_position_bucket(row: HistoricalPlayerSeason) -> str:
    """Bucket a player into PG/SG/SF/PF/C from their stat profile.

    Heuristic — calibrated against modern (post-2010) NBA. Two signals:

      * **Floor sense:** AST/36 — primary distributors are 1s, role
        guards 2s, wings/bigs lower.
      * **Frontcourt sense:** (REB + 1.5 * BLK)/36 — bigs rebound
        and block, wings less so, guards not at all.

    The thresholds were eyeballed from the actual stat distributions
    of the 1980-present corpus and tightened until they roughly
    matched listed positions for a sanity-check sample of stars.

    Misclassifications are FINE — the bucket is a soft filter, not
    a hard gate. A wing who looks like a big in one season just gets
    compared against bigs that year, and the cosine similarity score
    will still surface his real comps.
    """
    mpg = row.minutes if row.minutes > 0 else 24.0
    ast36 = row.ast * (36.0 / mpg)
    big36 = (row.reb + 1.5 * row.blk) * (36.0 / mpg)

    # Big-man axis dominates: very high frontcourt production → big.
    if big36 >= 14.0 and ast36 < 3.0:
        return "C"
    if big36 >= 11.0 and ast36 < 4.0:
        return "PF"
    # Guard axis: high assists → PG, moderate → SG.
    if ast36 >= 6.5:
        return "PG"
    if ast36 >= 4.0 and big36 < 8.0:
        return "SG"
    # Wing default.
    if big36 < 9.0:
        return "SF"
    # Fallback for combo bigs.
    return "PF"


# ---------------------------------------------------------------------------
# Corpus normalization
# ---------------------------------------------------------------------------

def build_corpus_profiles(
    rows: Iterable[HistoricalPlayerSeason],
) -> CorpusProfiles:
    """Build the full corpus: profile every player-season + z-score."""
    profiles = [build_profile(r) for r in rows]
    if not profiles:
        return CorpusProfiles(
            profiles=[],
            feature_means=np.zeros(len(FEATURE_NAMES)),
            feature_stds=np.ones(len(FEATURE_NAMES)),
        )
    raw = np.vstack([p.raw_vec for p in profiles])
    means = raw.mean(axis=0)
    stds = raw.std(axis=0)
    # Replace zero stds (constant features) with 1.0 to avoid divide-
    # by-zero. Constant features contribute 0 distance afterward.
    stds_safe = np.where(stds > 1e-9, stds, 1.0)
    normed = (raw - means) / stds_safe
    for prof, nv in zip(profiles, normed):
        prof.norm_vec = nv
    return CorpusProfiles(
        profiles=profiles,
        feature_means=means,
        feature_stds=stds_safe,
    )


def zscore_normalize(
    raw_vecs: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Standalone z-score for batch operations. Stable for divide-by-zero."""
    stds_safe = np.where(stds > 1e-9, stds, 1.0)
    return (raw_vecs - means) / stds_safe


# ---------------------------------------------------------------------------
# Current player → Profile (no historical row available)
# ---------------------------------------------------------------------------

def build_profile_from_stats(
    *,
    nba_id: str,
    name: str,
    season: str,
    season_end_year: int,
    age: float,
    team: Optional[str],
    gp: int,
    mpg: float,
    pts: float,
    reb: float,
    ast: float,
    stl: float,
    blk: float,
    tov: float,
    tpm: float,
    fga: float,
    fta: float,
    position_bucket: Optional[str] = None,
) -> Profile:
    """Build a Profile for a current player (no HistoricalPlayerSeason needed).

    Used to project CURRENT players (the ones we're ranking) into the
    corpus's normalized space. The caller is expected to z-score the
    raw_vec against the corpus statistics afterward.
    """
    raw_vec = np.array([
        _per36(pts, mpg),
        _per36(reb, mpg),
        _per36(ast, mpg),
        _per36(stl, mpg),
        _per36(blk, mpg),
        _per36(tpm, mpg),
        _per36(tov, mpg),
        _per36(fga, mpg),
        _per36(fta, mpg),
        _ts_pct(pts, fga, fta),
        min(1.0, gp / 82.0),
        mpg,
    ], dtype=np.float64)
    if position_bucket is None:
        # Reuse derive_position_bucket via a tiny shim.
        from ..sources.historical_nba import HistoricalPlayerSeason as _H
        shim = _H(
            nba_id=nba_id, name=name, season=season, season_end_year=season_end_year,
            age=age, team=team, gp=gp, minutes=mpg, pts=pts, reb=reb, ast=ast,
            stl=stl, blk=blk, tov=tov, tpm=tpm, fga=fga, fta=fta,
            fgm=0.0, ftm=0.0, fg_pct=0.0, ft_pct=0.0,
        )
        position_bucket = derive_position_bucket(shim)
    return Profile(
        nba_id=nba_id,
        name=name,
        season=season,
        season_end_year=season_end_year,
        age=age,
        team=team,
        position_bucket=position_bucket,
        raw_vec=raw_vec,
        season_row=None,
    )


# ---------------------------------------------------------------------------
# College vectorization (PR #7 — rookie chain)
# ---------------------------------------------------------------------------

# The college feature space mirrors the NBA feature space dimension-
# for-dimension so the two corpora can be compared directly. We do NOT
# share the same z-score normalization with the NBA corpus, however —
# college and NBA stats live in genuinely different distributions
# (a 20-PPG college freshman ≠ a 20-PPG NBA veteran). Each corpus
# z-scores against ITSELF, and the rookie comp lookup happens within
# the college corpus. The NBA careers attached to the college comps
# via the bridge supply the projection (longevity + fantasy points).

COLLEGE_FEATURE_NAMES: tuple[str, ...] = (
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "stl_per36",
    "blk_per36",
    "tpm_per36",
    "tov_per36_proxy",
    "fga_per36",
    "fta_per36",
    "ts_pct",
    "gp_pct",          # GP / 35 (full NCAA D1 reg+conf tournament)
    "mpg",
    # NCAA-specific signals — these don't have NBA analogues but they
    # SHARPEN the within-college comp search (age-relative production,
    # conference strength, class indicator).
    "usg_pct",
    "age_rel_class",   # age minus typical age for that class
    "conf_strength",   # 1.00 P5 / 0.92 HM / 0.83 MM / 0.75 LM
    "class_progress",  # 0 Fr / 1 So / 2 Jr / 3 Sr (linear)
)

# Typical age-at-season for each NCAA class (approximate, used as the
# baseline against which we compute age_rel_class).
_TYPICAL_AGE_BY_CLASS = {
    "Fr": 18.5, "So": 19.5, "Jr": 20.5, "Sr": 21.5,
    "Gr": 22.5, "R-Fr": 19.5, "R-So": 20.5, "R-Jr": 21.5, "R-Sr": 22.5,
}
_CLASS_PROGRESS = {
    "Fr": 0, "So": 1, "Jr": 2, "Sr": 3, "Gr": 4,
    "R-Fr": 0, "R-So": 1, "R-Jr": 2, "R-Sr": 3,
}


def _derive_college_bucket(
    ast_pg: float, reb_pg: float, blk_pg: float, mpg: float,
    height: Optional[str] = None,
) -> str:
    """Bucket an NCAA player into PG/SG/SF/PF/C from their stat profile.

    Mirrors derive_position_bucket but tuned for NCAA per-game ranges
    (college pace is slower, scoring totals smaller). Optionally uses
    height when available as a tiebreaker — barttorvik's listed height
    is usually accurate.
    """
    if mpg <= 0:
        mpg = 24.0
    ast36 = ast_pg * (36.0 / mpg)
    big36 = (reb_pg + 1.5 * blk_pg) * (36.0 / mpg)
    inches = _height_to_inches(height)
    # Honest 7-footer who rebounds/blocks → C. 6-10 (82) and below are
    # too often combo bigs / stretch 4s to lock as C purely from height.
    if inches and inches >= 83 and big36 >= 9.5 and ast36 < 4.5:
        return "C"
    if big36 >= 13.0 and ast36 < 3.0:
        return "C"
    if big36 >= 9.0 and ast36 < 3.5:
        return "PF"
    if ast36 >= 5.5:
        return "PG"
    if ast36 >= 3.5 and big36 < 8.0:
        return "SG"
    if big36 < 9.0:
        return "SF"
    # 6-9+ with playmaking → SF/PF combo; default to PF.
    if inches and inches >= 80:
        return "PF"
    return "SF"


def _height_to_inches(h: Optional[str]) -> Optional[int]:
    if not h or not isinstance(h, str) or "-" not in h:
        return None
    try:
        ft, inch = h.split("-", 1)
        return int(ft) * 12 + int(inch)
    except Exception:
        return None


def vectorize_college_season(
    row,
    conference_multiplier: float = 1.0,
) -> Profile:
    """Turn one HistoricalNCAASeason into a Profile in college space.

    Implements the PR #7 college side of the similarity engine:
      * Per-36 production rates (PTS/REB/AST/STL/BLK/3PM/FGA/FTA)
      * TS%, USG%
      * Class progress (0..3) + age-relative-to-class
      * Conference strength multiplier applied to the production
        vector (a 20 PPG SEC freshman is more impressive than a
        20 PPG Sun Belt freshman of the same age).
      * Position bucket derived from per-36 stats + height.

    The returned Profile's ``raw_vec`` is in the COLLEGE feature space
    (length == ``len(COLLEGE_FEATURE_NAMES)``), NOT the NBA feature
    space. Pass a list of these to ``build_college_corpus_profiles``
    to z-score them against the NCAA corpus.
    """
    mpg = row.mpg if row.mpg and row.mpg > 0 else 24.0
    # Apply conference strength to the production dimensions so the
    # KNN naturally prefers comps from the right competitive tier.
    cm = conference_multiplier
    raw_vec = np.array([
        _per36(row.pts_pg, mpg) * cm,
        _per36(row.reb_pg, mpg) * cm,
        _per36(row.ast_pg, mpg) * cm,
        _per36(row.stl_pg, mpg) * cm,
        _per36(row.blk_pg, mpg) * cm,
        _per36(row.tpm_pg, mpg) * cm,
        # TO/G isn't directly in the NCAA row — TO% × USG% is a coarse
        # proxy. Multiplied by 0.01 to roughly normalize to per-36 scale.
        (row.to_pct or 0.0) * (row.usg_pct or 0.0) * 0.01,
        _per36(row.fga_pg, mpg) * cm,
        _per36(row.fta_pg, mpg) * cm,
        row.ts_pct or 0.0,
        min(1.0, row.gp / 35.0),
        mpg,
        row.usg_pct or 0.0,
        _age_rel_class(row.age_at_season, row.class_year),
        cm,
        _CLASS_PROGRESS.get((row.class_year or ""), 1),
    ], dtype=np.float64)
    bucket = _derive_college_bucket(
        ast_pg=row.ast_pg, reb_pg=row.reb_pg, blk_pg=row.blk_pg,
        mpg=mpg, height=row.height,
    )
    return Profile(
        nba_id=row.sr_player_id,   # piggyback the nba_id slot for ncaa pid
        name=row.name,
        season=row.season,
        season_end_year=row.season_end_year,
        age=row.age_at_season if row.age_at_season is not None else _TYPICAL_AGE_BY_CLASS.get(row.class_year or "", 19.5),
        team=row.school,
        position_bucket=bucket,
        raw_vec=raw_vec,
        season_row=None,
    )


def _age_rel_class(age: Optional[float], class_year: Optional[str]) -> float:
    """Age relative to the typical age for the player's class.

    Positive = older than peers (often less NBA upside), negative =
    younger than peers (often higher NBA upside — early-entry).
    """
    if age is None or class_year is None:
        return 0.0
    typ = _TYPICAL_AGE_BY_CLASS.get(class_year, 19.5)
    return float(age) - typ


def build_college_corpus_profiles(
    rows: Iterable,
    conference_lookup=None,
) -> CorpusProfiles:
    """Build the college corpus: profile every NCAA player-season + z-score.

    ``conference_lookup`` is a callable ``conf_code -> multiplier``. If
    omitted, defaults to the standard 4-tier lookup in historical_ncaa.
    """
    if conference_lookup is None:
        from ..sources.historical_ncaa import conference_strength_multiplier
        conference_lookup = conference_strength_multiplier

    profiles = [
        vectorize_college_season(r, conference_multiplier=conference_lookup(r.conference))
        for r in rows
    ]
    if not profiles:
        return CorpusProfiles(
            profiles=[],
            feature_means=np.zeros(len(COLLEGE_FEATURE_NAMES)),
            feature_stds=np.ones(len(COLLEGE_FEATURE_NAMES)),
        )
    raw = np.vstack([p.raw_vec for p in profiles])
    means = raw.mean(axis=0)
    stds = raw.std(axis=0)
    stds_safe = np.where(stds > 1e-9, stds, 1.0)
    normed = (raw - means) / stds_safe
    for prof, nv in zip(profiles, normed):
        prof.norm_vec = nv
    return CorpusProfiles(
        profiles=profiles,
        feature_means=means,
        feature_stds=stds_safe,
    )
