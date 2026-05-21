"""KNN comparables — find each player's most similar historical seasons.

Workflow
--------
For a current player P at age A, we want the top-K historical player-
seasons whose age was within ±1 of A, with position-bucket compatible
(same or adjacent on the PG → C axis), and whose profile vector is
nearest in cosine distance to P's. For each comp we also need:

  * What the comp did the REST of his career — total fantasy points,
    games played, seasons played — so the projection layer can build
    a career arc from him.
  * Per-year survival flag: did the comp still play at A+1, A+2, ...?

The "rest-of-career" rollup is precomputed once per corpus build via
``build_career_index`` and indexed by ``nba_id``. Comp lookup is then
a numpy dot-product against the corpus matrix — fast even at 22K rows.

Output
------
A ``Comparable`` dataclass per match, carrying both the identity of
the comp (name, season, age) and the actionable forward-looking
fields the projection layer consumes.

Parameterizable for PR #5
-------------------------
The functions here are deliberately agnostic to "current player" vs
"college prospect". ``find_comparables`` accepts an arbitrary profile
+ age + bucket and a corpus to search — so the college engine can
build a parallel college→NBA bridge corpus and call the same code path.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from ..sources.historical_nba import HistoricalPlayerSeason
from .vectorize import (
    Profile,
    CorpusProfiles,
    ADJACENT_BUCKETS,
)


# ---------------------------------------------------------------------------
# Speed: the corpus has ~10K player-seasons and we search it once per
# current player (~570). Naively iterating per call is too slow in
# Python. ``prepare_corpus_for_search`` builds a stacked numpy matrix
# of normalized vectors + per-(age_bucket, position_bucket) indices
# so each search is dominated by a single matmul.
# ---------------------------------------------------------------------------

@dataclass
class CorpusSearchIndex:
    """Searchable matrix view of a CorpusProfiles.

    Built once per corpus; reused across all current players. Holds:
      * ``norm_matrix``: (N, D) float64 array of z-scored vectors.
      * ``norm_row_norms``: (N,) precomputed L2 norms (used by cosine).
      * ``eligible_index``: {(age_bin, bucket): np.ndarray of corpus row
        indices} so we can look up candidates without iterating the
        full corpus.
      * ``has_forward``: (N,) bool array of which corpus rows have at
        least one later season for the same player. Pre-evaluated so
        the search loop doesn't re-check it.
    """
    corpus: CorpusProfiles
    norm_matrix: np.ndarray
    norm_row_norms: np.ndarray
    eligible_index: dict
    has_forward: np.ndarray


def prepare_corpus_for_search(
    corpus: CorpusProfiles,
    career_index: "CareerIndex",
) -> CorpusSearchIndex:
    """Precompute the search structures for fast repeated KNN lookups."""
    if not corpus.profiles:
        return CorpusSearchIndex(
            corpus=corpus,
            norm_matrix=np.zeros((0, 1)),
            norm_row_norms=np.zeros(0),
            eligible_index={},
            has_forward=np.zeros(0, dtype=bool),
        )
    norm_matrix = np.vstack([p.norm_vec for p in corpus.profiles])
    norm_row_norms = np.linalg.norm(norm_matrix, axis=1)
    has_forward = np.zeros(len(corpus.profiles), dtype=bool)
    eligible_index: dict = {}
    for i, p in enumerate(corpus.profiles):
        rollup = career_index.forward_rollup(p.nba_id, from_age=p.age)
        has_forward[i] = rollup["remaining_seasons"] > 0
        if not has_forward[i]:
            continue
        age_bin = int(round(p.age))
        key = (age_bin, p.position_bucket)
        eligible_index.setdefault(key, []).append(i)
    # Convert to arrays for fast filtering.
    eligible_arrays = {k: np.array(v, dtype=np.int64) for k, v in eligible_index.items()}
    return CorpusSearchIndex(
        corpus=corpus,
        norm_matrix=norm_matrix,
        norm_row_norms=norm_row_norms,
        eligible_index=eligible_arrays,
        has_forward=has_forward,
    )


# Default KNN size. 20 is enough for stable medians without diluting
# the signal with mediocre matches.
DEFAULT_K = 20

# Default age window (±1 year). 18yo Cooper Flagg can match 17 / 18
# / 19 yo historical seasons.
DEFAULT_AGE_WINDOW = 1.0


@dataclass
class CareerIndex:
    """Per-player rollup of "rest-of-career" stats keyed by nba_id.

    Built once per corpus. For each nba_id, ``forward_rollup_from_age(a)``
    returns the stats from age ``a+1`` through the player's last
    cached season — i.e. how the rest of their career played out
    after season at age ``a``.

    Forward rollups are memoized on ``(nba_id, rounded_age)`` because
    they're called once per candidate during the KNN eligibility scan;
    without memoization we re-do the same fantasy_ppg loop hundreds
    of times per cohort player. Cache keys use ``round(age, 1)`` so
    near-identical ages still hit.
    """
    # nba_id -> sorted list of HistoricalPlayerSeason (by season_end_year)
    seasons_by_player: dict[str, list[HistoricalPlayerSeason]]
    # max season_end_year present in corpus — used to detect "still active"
    corpus_max_season_end_year: int
    # Memoization cache. Populated lazily by forward_rollup().
    _rollup_cache: dict = field(default_factory=dict)

    def forward_rollup(
        self,
        nba_id: str,
        from_age: float,
    ) -> dict:
        """Aggregate stats from age (from_age + 1) onward for the given player.

        Returns:
            {
              "remaining_seasons": int,
              "remaining_games": int,
              "fantasy_ppg_dhk": float | None,
              "fantasy_ppg_default": float | None,
              "seasons_after_age": list[(age, season_str)],
              "last_age": float | None,
              "censored": bool   # True if player's last corpus season is
                                  # the corpus_max_season_end_year — i.e.
                                  # he was still active when our window
                                  # ended; longevity here is a lower bound.
            }
        """
        cache_key = (nba_id, round(float(from_age), 1))
        cached = self._rollup_cache.get(cache_key)
        if cached is not None:
            return cached

        from ..sources.basketball_reference import (
            fantasy_ppg as _fppg,
            _PlayerProduction as _Prod,
        )

        seasons = self.seasons_by_player.get(nba_id, [])
        forward = [s for s in seasons if s.age > from_age + 0.5]
        if not forward:
            empty = {
                "remaining_seasons": 0,
                "remaining_games": 0,
                "fantasy_ppg_dhk": None,
                "fantasy_ppg_default": None,
                "seasons_after_age": [],
                "last_age": None,
                "censored": False,
            }
            self._rollup_cache[cache_key] = empty
            return empty

        # Translate HistoricalPlayerSeason → _PlayerProduction so we
        # can reuse the existing fantasy_ppg helper (single source of
        # truth for scoring math).
        def _to_prod(s: HistoricalPlayerSeason) -> _Prod:
            return _Prod(
                nba_id=s.nba_id, name=s.name, team=s.team, age=s.age,
                gp=s.gp, minutes=s.minutes,
                pts=s.pts, reb=s.reb, ast=s.ast, stl=s.stl, blk=s.blk,
                tov=s.tov, tpm=s.tpm,
            )

        total_games = sum(s.gp for s in forward)
        # Games-weighted fantasy PPG across remaining career.
        if total_games > 0:
            fp_dhk = sum(_fppg(_to_prod(s), "points_dhk") * s.gp for s in forward) / total_games
            fp_def = sum(_fppg(_to_prod(s), "points_default") * s.gp for s in forward) / total_games
        else:
            fp_dhk = None
            fp_def = None

        last_age = max(s.age for s in forward)
        last_season_year = max(s.season_end_year for s in forward)
        censored = last_season_year >= self.corpus_max_season_end_year

        result = {
            "remaining_seasons": len(forward),
            "remaining_games": total_games,
            "fantasy_ppg_dhk": fp_dhk,
            "fantasy_ppg_default": fp_def,
            "seasons_after_age": [(s.age, s.season) for s in forward],
            "last_age": last_age,
            "censored": censored,
        }
        self._rollup_cache[cache_key] = result
        return result


def build_career_index(
    corpus_rows: Iterable[HistoricalPlayerSeason],
) -> CareerIndex:
    """Group historical player-seasons by nba_id for rollup queries."""
    seasons_by_player: dict[str, list[HistoricalPlayerSeason]] = {}
    max_season_end = 0
    for r in corpus_rows:
        seasons_by_player.setdefault(r.nba_id, []).append(r)
        if r.season_end_year > max_season_end:
            max_season_end = r.season_end_year
    for k in seasons_by_player:
        seasons_by_player[k].sort(key=lambda s: s.season_end_year)
    return CareerIndex(
        seasons_by_player=seasons_by_player,
        corpus_max_season_end_year=max_season_end,
    )


# ---------------------------------------------------------------------------
# Comparable record
# ---------------------------------------------------------------------------

@dataclass
class Comparable:
    """One historical comp + similarity score + forward-career rollup."""
    comp_nba_id: str
    comp_name: str
    comp_season: str
    comp_age_when_compared: float
    similarity: float  # cosine similarity in normalized vector space, 0..1
    position_bucket: str
    bucket_match: bool   # same bucket vs adjacent
    # Forward-career fields (filled in via CareerIndex.forward_rollup).
    remaining_seasons: int = 0
    remaining_games: int = 0
    remaining_fantasy_ppg_dhk: Optional[float] = None
    remaining_fantasy_ppg_default: Optional[float] = None
    last_age: Optional[float] = None
    censored: bool = False
    # Which ages the comp still played at, for survival probability.
    ages_after: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core KNN
# ---------------------------------------------------------------------------

def _cosine_similarity_matrix(
    target_vec: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between target_vec (D,) and matrix (N, D).

    Returns array of length N. Stable for zero-vectors (returns 0).
    """
    t_norm = np.linalg.norm(target_vec)
    if t_norm < 1e-9:
        return np.zeros(matrix.shape[0])
    row_norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(row_norms > 1e-9, row_norms, 1.0)
    sims = (matrix @ target_vec) / (safe * t_norm)
    sims = np.where(row_norms > 1e-9, sims, 0.0)
    # Clip to [-1, 1] then shift to [0, 1] for cleaner aggregation.
    sims = np.clip(sims, -1.0, 1.0)
    return (sims + 1.0) / 2.0


def find_comparables(
    target_profile: Profile,
    corpus: CorpusProfiles,
    career_index: CareerIndex,
    k: int = DEFAULT_K,
    age_window: float = DEFAULT_AGE_WINDOW,
    exclude_nba_id: Optional[str] = None,
    bucket_penalty: float = 0.05,
    search_index: Optional[CorpusSearchIndex] = None,
) -> list[Comparable]:
    """Find the top-K historical comps for a single target profile.

    Filtering:
      * Age within ``target.age ± age_window``
      * Same or adjacent position bucket
      * Excludes the target player's own seasons (when ``exclude_nba_id``
        is set, even at different ages — prevents self-matching)
      * Comp must have at least one later season in the corpus
        (otherwise "remaining career" is undefined).

    Scoring:
      * cosine similarity in normalized space, with a small subtraction
        ``bucket_penalty`` for adjacent (vs same) bucket matches. Without
        this, a 7-foot center routinely matches a power forward who
        rebounds + blocks shots — true, but we want the same-bucket
        match preferred when available.

    Performance:
      * Pass a precomputed ``search_index`` (via ``prepare_corpus_for_search``)
        when running KNN repeatedly across a cohort. Without it we
        build a fresh index every call, which is fine for a single
        lookup but quadratic across a 500-player cohort.

    Returns Comparables sorted by similarity desc.
    """
    if target_profile.norm_vec is None:
        normed = corpus.normalize(target_profile.raw_vec)
        target_profile = Profile(
            nba_id=target_profile.nba_id,
            name=target_profile.name,
            season=target_profile.season,
            season_end_year=target_profile.season_end_year,
            age=target_profile.age,
            team=target_profile.team,
            position_bucket=target_profile.position_bucket,
            raw_vec=target_profile.raw_vec,
            norm_vec=normed,
            season_row=target_profile.season_row,
        )

    if search_index is None:
        search_index = prepare_corpus_for_search(corpus, career_index)

    tbucket = target_profile.position_bucket
    allowed_buckets = ADJACENT_BUCKETS.get(tbucket, {tbucket})
    age_low = target_profile.age - age_window
    age_high = target_profile.age + age_window
    age_bin_low = int(np.floor(age_low))
    age_bin_high = int(np.ceil(age_high))

    # Collect candidate corpus row indices via the precomputed bucket
    # lookup. This is O(small) vs O(N) iteration.
    cand_idx_parts: list[np.ndarray] = []
    for age_bin in range(age_bin_low, age_bin_high + 1):
        for bucket in allowed_buckets:
            arr = search_index.eligible_index.get((age_bin, bucket))
            if arr is not None and arr.size > 0:
                cand_idx_parts.append(arr)
    if not cand_idx_parts:
        return []
    cand_idx = np.unique(np.concatenate(cand_idx_parts))

    # Finer age filter (±age_window is fractional, age bins are int).
    profiles = corpus.profiles
    ages = np.array([profiles[i].age for i in cand_idx])
    age_ok = (ages >= age_low) & (ages <= age_high)
    cand_idx = cand_idx[age_ok]
    if cand_idx.size == 0:
        return []

    # Exclude same-nba_id rows.
    if exclude_nba_id is not None:
        keep_mask = np.array([profiles[i].nba_id != exclude_nba_id for i in cand_idx])
        cand_idx = cand_idx[keep_mask]
        if cand_idx.size == 0:
            return []

    # Cosine similarity (matrix slice).
    cand_matrix = search_index.norm_matrix[cand_idx]
    cand_norms = search_index.norm_row_norms[cand_idx]
    t_norm = np.linalg.norm(target_profile.norm_vec)
    if t_norm < 1e-9:
        return []
    safe = np.where(cand_norms > 1e-9, cand_norms, 1.0)
    raw_sims = (cand_matrix @ target_profile.norm_vec) / (safe * t_norm)
    raw_sims = np.where(cand_norms > 1e-9, raw_sims, -1.0)
    raw_sims = np.clip(raw_sims, -1.0, 1.0)
    sims = (raw_sims + 1.0) / 2.0

    # Bucket penalty for adjacent (non-same) bucket.
    for j, i in enumerate(cand_idx):
        if profiles[int(i)].position_bucket != tbucket:
            sims[j] = max(0.0, sims[j] - bucket_penalty)

    # Top-K.
    top_k_local = np.argsort(-sims)[:k]
    out: list[Comparable] = []
    for j in top_k_local:
        i = int(cand_idx[j])
        p = profiles[i]
        rollup = career_index.forward_rollup(p.nba_id, from_age=p.age)
        out.append(Comparable(
            comp_nba_id=p.nba_id,
            comp_name=p.name,
            comp_season=p.season,
            comp_age_when_compared=p.age,
            similarity=float(sims[j]),
            position_bucket=p.position_bucket,
            bucket_match=(p.position_bucket == tbucket),
            remaining_seasons=rollup["remaining_seasons"],
            remaining_games=rollup["remaining_games"],
            remaining_fantasy_ppg_dhk=rollup["fantasy_ppg_dhk"],
            remaining_fantasy_ppg_default=rollup["fantasy_ppg_default"],
            last_age=rollup["last_age"],
            censored=rollup["censored"],
            ages_after=[a for a, _ in rollup["seasons_after_age"]],
        ))
    return out
