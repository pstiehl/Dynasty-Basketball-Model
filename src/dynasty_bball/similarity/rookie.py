"""Rookie similarity engine -- college->NBA career-arc projection.

This is the PR #7 chain that extends the PR #4/#5 NBA-similarity engine
to rookies and incoming draft prospects. Workflow:

  1. Vectorize a target NCAA player (a rookie, a current draft prospect,
     or any player with <1 NBA season) in the college feature space.
  2. KNN over the NCAA corpus to find top-K college comparables, filtered
     by same/adjacent position bucket and matching class (Fr/So/Jr/Sr).
  3. For each comp, look up their realized NBA career through the
     ncaa->nba bridge. Some comps will have ZERO NBA seasons (didn't
     make the league / cup of coffee) -- those still count toward the
     similarity-weighted aggregation but with zero remaining NBA
     production, anchoring the projection toward realistic outcomes.
  4. Aggregate with similarity weights, time-discount 5%/year, output
     a CareerProjection that slots into the same composite pipeline as
     PR #4's NBA-only projections.

This produces ``rookie_dynasty_value_dhk``, ``rookie_dynasty_value_default``
and the comparables list for site rendering.

The aggregation uses the same ``project_career`` machinery as the NBA
engine -- we just supply a list of Comparable records whose
remaining-career fields come from the bridge lookup instead of being
"the comp's own later seasons in the NBA corpus."
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..sources.historical_ncaa import HistoricalNCAASeason, conference_strength_multiplier
from .vectorize import (
    Profile,
    CorpusProfiles,
    ADJACENT_BUCKETS,
    POSITION_BUCKETS,
    vectorize_college_season,
    build_college_corpus_profiles,
)

# Integer-encode position buckets so the eligibility mask can use
# fast numpy comparisons instead of object-dtype np.isin (which is
# 30x slower).
_BUCKET_CODES = {b: i for i, b in enumerate(POSITION_BUCKETS)}
_ADJACENT_CODES = {
    _BUCKET_CODES[b]: frozenset(_BUCKET_CODES[x] for x in adj if x in _BUCKET_CODES)
    for b, adj in ADJACENT_BUCKETS.items() if b in _BUCKET_CODES
}
_CLASS_LIST = ("", "Fr", "So", "Jr", "Sr", "Gr", "R-Fr", "R-So", "R-Jr", "R-Sr")
_CLASS_CODES = {c: i for i, c in enumerate(_CLASS_LIST)}
from .comparables import Comparable, CareerIndex
from .projection import (
    CareerProjection,
    project_career,
    MIN_PROJECTED_REMAINING_YEARS,
)


log = logging.getLogger(__name__)


DEFAULT_K = 20
DEFAULT_AGE_WINDOW = 1.5


@dataclass
class NCAACorpusSearchIndex:
    """Searchable matrix view of the NCAA corpus.

    Holds the normalized vector matrix plus vectorized per-row
    metadata (age, class code, bucket code, sr_player_id) so the
    eligibility mask can be built with numpy comparisons instead of
    a Python loop over 55K rows. Buckets and classes are integer-
    coded so the comparisons are O(N) tight numpy loops.
    """
    corpus: CorpusProfiles
    norm_matrix: np.ndarray
    norm_row_norms: np.ndarray
    # Vectorized metadata.
    ages: np.ndarray             # float (N,)
    class_codes: np.ndarray      # int8 (N,)
    bucket_codes: np.ndarray     # int8 (N,)
    buckets: np.ndarray          # object (N,) -- kept for output
    pids: np.ndarray             # object (N,) -- sr_player_id


def prepare_ncaa_search_index(corpus: CorpusProfiles, rows: list[HistoricalNCAASeason]) -> NCAACorpusSearchIndex:
    """Precompute the search structures for fast repeated KNN over NCAA."""
    if not corpus.profiles:
        return NCAACorpusSearchIndex(
            corpus=corpus,
            norm_matrix=np.zeros((0, 1)),
            norm_row_norms=np.zeros(0),
            ages=np.zeros(0),
            classes=np.array([], dtype=object),
            buckets=np.array([], dtype=object),
            pids=np.array([], dtype=object),
        )
    norm_matrix = np.vstack([p.norm_vec for p in corpus.profiles])
    norm_row_norms = np.linalg.norm(norm_matrix, axis=1)
    ages = np.array([p.age for p in corpus.profiles], dtype=np.float64)
    class_codes = np.array(
        [_CLASS_CODES.get(r.class_year or "", 0) for r in rows],
        dtype=np.int8,
    )
    bucket_codes = np.array(
        [_BUCKET_CODES.get(p.position_bucket, -1) for p in corpus.profiles],
        dtype=np.int8,
    )
    buckets = np.array([p.position_bucket for p in corpus.profiles], dtype=object)
    pids = np.array([r.sr_player_id for r in rows], dtype=object)
    return NCAACorpusSearchIndex(
        corpus=corpus,
        norm_matrix=norm_matrix,
        norm_row_norms=norm_row_norms,
        ages=ages,
        class_codes=class_codes,
        bucket_codes=bucket_codes,
        buckets=buckets,
        pids=pids,
    )


# Memoize bridge lookups by nba_id -- the full-career rollup is
# independent of the target. ~800 entries max.
_BRIDGE_ROLLUP_CACHE: dict[tuple[int, str], dict] = {}


def _bridge_lookup(
    btv_pid: str,
    bridge_by_pid: dict,
    nba_career_index: CareerIndex,
) -> dict:
    """For a college player (btv_pid), return their realized NBA career.

    Returns a dict with the SAME shape as
    ``CareerIndex.forward_rollup(...)``, so the comparable's
    remaining-career fields plug right in.

    If the bridge has no NBA match for this btv_pid, returns a "no NBA
    career" rollup -- remaining_seasons=0, fantasy_ppg=None. This is
    NOT skipped from the comp list -- it correctly anchors the
    projection by counting "comp who washed out" as a possible outcome.
    """
    nba_id = bridge_by_pid.get(btv_pid)
    if not nba_id:
        return {
            "remaining_seasons": 0,
            "remaining_games": 0,
            "fantasy_ppg_dhk": None,
            "fantasy_ppg_default": None,
            "seasons_after_age": [],
            "last_age": None,
            "censored": False,
            "nba_id": None,
        }
    cache_key = (id(nba_career_index), nba_id)
    cached = _BRIDGE_ROLLUP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    # Sum the comp's entire NBA career (not "after age X" -- the
    # college comp's full pro career is the relevant projection.)
    nba_seasons = nba_career_index.seasons_by_player.get(nba_id, [])
    if not nba_seasons:
        result = {
            "remaining_seasons": 0,
            "remaining_games": 0,
            "fantasy_ppg_dhk": None,
            "fantasy_ppg_default": None,
            "seasons_after_age": [],
            "last_age": None,
            "censored": False,
            "nba_id": nba_id,
        }
        _BRIDGE_ROLLUP_CACHE[cache_key] = result
        return result
    from ..sources.basketball_reference import (
        fantasy_ppg as _fppg,
        _PlayerProduction as _Prod,
    )

    def _to_prod(s):
        return _Prod(
            nba_id=s.nba_id, name=s.name, team=s.team, age=s.age,
            gp=s.gp, minutes=s.minutes,
            pts=s.pts, reb=s.reb, ast=s.ast, stl=s.stl, blk=s.blk,
            tov=s.tov, tpm=s.tpm,
        )

    total_games = sum(s.gp for s in nba_seasons)
    if total_games > 0:
        fp_dhk = sum(_fppg(_to_prod(s), "points_dhk") * s.gp for s in nba_seasons) / total_games
        fp_def = sum(_fppg(_to_prod(s), "points_default") * s.gp for s in nba_seasons) / total_games
    else:
        fp_dhk = None
        fp_def = None
    last_age = max(s.age for s in nba_seasons)
    last_season_year = max(s.season_end_year for s in nba_seasons)
    censored = last_season_year >= nba_career_index.corpus_max_season_end_year
    result = {
        "remaining_seasons": len(nba_seasons),
        "remaining_games": total_games,
        "fantasy_ppg_dhk": fp_dhk,
        "fantasy_ppg_default": fp_def,
        "seasons_after_age": [(s.age, s.season) for s in nba_seasons],
        "last_age": last_age,
        "censored": censored,
        "nba_id": nba_id,
    }
    _BRIDGE_ROLLUP_CACHE[cache_key] = result
    return result


def find_college_comparables_batch(
    targets: list[tuple],
    ncaa_corpus: CorpusProfiles,
    ncaa_index: NCAACorpusSearchIndex,
    bridge_by_pid: dict,
    nba_career_index: CareerIndex,
    k: int = DEFAULT_K,
    age_window: float = DEFAULT_AGE_WINDOW,
    bucket_penalty: float = 0.05,
) -> dict[str, list[Comparable]]:
    """Run KNN for a batch of targets in one matmul.

    ``targets`` is a list of ``(btv_pid, Profile, class_year)``. Returns
    ``{btv_pid: [Comparable, ...]}``. This is ~50x faster than calling
    ``find_college_comparables`` per target because the matmul cost
    dominates and a single (T, D) x (D, N) matmul replaces T separate
    (cand, D) x (D,) matvecs.
    """
    if not targets or ncaa_index.norm_matrix.shape[0] == 0:
        return {}
    # Normalize profiles that haven't been z-scored yet.
    norm_targets = []
    for pid, prof, klass in targets:
        if prof.norm_vec is None:
            normed = ncaa_corpus.normalize(prof.raw_vec)
            norm_targets.append((pid, normed, prof.position_bucket, prof.age, klass))
        else:
            norm_targets.append((pid, prof.norm_vec, prof.position_bucket, prof.age, klass))

    T_mat = np.vstack([t[1] for t in norm_targets]).astype(np.float32)
    T_norms = np.linalg.norm(T_mat, axis=1)
    T_norms_safe = np.where(T_norms > 1e-9, T_norms, 1.0)

    # Process targets in CHUNKS to bound memory. A (chunk, N) matrix
    # for N=55K, chunk=200, float32 is ~44 MB -- comfortable.
    cand_norms = ncaa_index.norm_row_norms
    cand_norms_safe = np.where(cand_norms > 1e-9, cand_norms, 1.0).astype(np.float32)
    M_T = ncaa_index.norm_matrix.astype(np.float32).T  # (D, N)

    CHUNK = 200
    n_targets = len(norm_targets)
    # Pre-extract pids and codes for cheap masking.
    pids_arr = ncaa_index.pids
    bucket_codes = ncaa_index.bucket_codes
    class_codes = ncaa_index.class_codes
    ages = ncaa_index.ages

    out: dict[str, list[Comparable]] = {}

    for chunk_start in range(0, n_targets, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n_targets)
        T_chunk = T_mat[chunk_start:chunk_end]   # (c, D)
        T_chunk_norms = T_norms_safe[chunk_start:chunk_end]
        # Cosine similarity matrix for this chunk.
        raw = T_chunk @ M_T  # (c, N)
        sims_chunk = raw / (T_chunk_norms[:, None] * cand_norms_safe[None, :])
        np.clip(sims_chunk, -1.0, 1.0, out=sims_chunk)
        sims_chunk = (sims_chunk + 1.0) / 2.0

        for local_i, target_i in enumerate(range(chunk_start, chunk_end)):
            pid, _norm, tbucket, tage, tclass = norm_targets[target_i]
            _process_one_target(
                pid=pid, tbucket=tbucket, tage=tage, tclass=tclass,
                sims_row=sims_chunk[local_i],
                ncaa_corpus=ncaa_corpus, ncaa_index=ncaa_index,
                bridge_by_pid=bridge_by_pid, nba_career_index=nba_career_index,
                k=k, age_window=age_window, bucket_penalty=bucket_penalty,
                out=out,
            )
    return out


def _process_one_target(
    *, pid, tbucket, tage, tclass, sims_row,
    ncaa_corpus, ncaa_index,
    bridge_by_pid, nba_career_index,
    k, age_window, bucket_penalty, out,
) -> None:
    """Mask + top-K for a single target's pre-computed similarities row."""
    tbucket_code = _BUCKET_CODES.get(tbucket, -1)
    allowed_codes = list(_ADJACENT_CODES.get(tbucket_code, frozenset([tbucket_code])))
    # Eligibility mask
    age_ok = np.abs(ncaa_index.ages - tage) <= float(age_window)
    if len(allowed_codes) == 1:
        bucket_ok = (ncaa_index.bucket_codes == allowed_codes[0])
    else:
        bucket_ok = np.zeros_like(age_ok)
        for c in allowed_codes:
            bucket_ok |= (ncaa_index.bucket_codes == c)
    eligible = age_ok & bucket_ok
    eligible[ncaa_index.pids == pid] = False
    if tclass:
        tclass_code = _CLASS_CODES.get(tclass, 0)
        eligible_strict = eligible & (ncaa_index.class_codes == tclass_code)
        if eligible_strict.any():
            eligible = eligible_strict
    if not eligible.any():
        out[pid] = []
        return
    sims = sims_row.copy()
    # Bucket penalty for adjacent bucket.
    adj = (ncaa_index.bucket_codes != tbucket_code)
    sims[adj] = np.maximum(0.0, sims[adj] - bucket_penalty)
    # Mask ineligible.
    sims[~eligible] = -1.0
    # Top-K
    top_k = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    top_k = top_k[np.argsort(-sims[top_k])]

    comps = []
    for idx_i in top_k:
        ii = int(idx_i)
        if sims[ii] < 0:
            continue
        prof = ncaa_corpus.profiles[ii]
        btv_pid = ncaa_index.pids[ii]
        bucket = ncaa_index.buckets[ii]
        rollup = _bridge_lookup(btv_pid, bridge_by_pid, nba_career_index)
        comps.append(Comparable(
            comp_nba_id=rollup["nba_id"] or f"ncaa:{btv_pid}",
            comp_name=prof.name,
            comp_season=prof.season,
            comp_age_when_compared=prof.age,
            similarity=float(sims[ii]),
            position_bucket=bucket,
            bucket_match=(bucket == tbucket),
            remaining_seasons=rollup["remaining_seasons"],
            remaining_games=rollup["remaining_games"],
            remaining_fantasy_ppg_dhk=rollup["fantasy_ppg_dhk"],
            remaining_fantasy_ppg_default=rollup["fantasy_ppg_default"],
            last_age=rollup["last_age"],
            censored=rollup["censored"],
            ages_after=[a for a, _ in rollup["seasons_after_age"]],
        ))
    out[pid] = comps


def find_college_comparables(
    target_profile: Profile,
    target_class: Optional[str],
    ncaa_corpus: CorpusProfiles,
    ncaa_index: NCAACorpusSearchIndex,
    bridge_by_pid: dict,
    nba_career_index: CareerIndex,
    k: int = DEFAULT_K,
    age_window: float = DEFAULT_AGE_WINDOW,
    exclude_btv_pid: Optional[str] = None,
    bucket_penalty: float = 0.05,
    require_same_class: bool = True,
) -> list[Comparable]:
    """Find top-K NCAA player-seasons most similar to the target.

    For each match, look up the player's realized NBA career via the
    bridge and attach it to the Comparable. Comps with no NBA career
    are kept (remaining_seasons=0) so the projection naturally accounts
    for the "didn't make it" outcome.

    Class match (``require_same_class=True``) is the rookie engine's
    analogue to age-windowing -- a Fr should match other Fr seasons
    (and adjacent classes via age_window). Strict class match catches
    the most meaningful pattern: 18-year-old Cooper Flagg should not
    be compared to 22-year-old Sr seniors who happen to have similar
    counting stats.
    """
    if target_profile.norm_vec is None:
        normed = ncaa_corpus.normalize(target_profile.raw_vec)
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

    tbucket = target_profile.position_bucket
    tbucket_code = _BUCKET_CODES.get(tbucket, -1)
    allowed_codes = list(_ADJACENT_CODES.get(tbucket_code, frozenset([tbucket_code])))

    n = len(ncaa_index.ages)
    if n == 0:
        return []

    # Eligibility mask -- vectorized, integer-coded for speed.
    age_ok = np.abs(ncaa_index.ages - float(target_profile.age)) <= float(age_window)
    if len(allowed_codes) == 1:
        bucket_ok = (ncaa_index.bucket_codes == allowed_codes[0])
    else:
        bucket_ok = np.zeros(n, dtype=bool)
        for c in allowed_codes:
            bucket_ok |= (ncaa_index.bucket_codes == c)
    eligible = age_ok & bucket_ok
    if exclude_btv_pid is not None:
        eligible &= (ncaa_index.pids != exclude_btv_pid)
    if require_same_class and target_class:
        target_class_code = _CLASS_CODES.get(target_class, 0)
        eligible &= (ncaa_index.class_codes == target_class_code)

    cand_idx = np.where(eligible)[0]
    if cand_idx.size == 0:
        # Relax: drop class requirement first, then bucket.
        if require_same_class:
            return find_college_comparables(
                target_profile, target_class, ncaa_corpus, ncaa_index,
                bridge_by_pid, nba_career_index, k=k, age_window=age_window,
                exclude_btv_pid=exclude_btv_pid, bucket_penalty=bucket_penalty,
                require_same_class=False,
            )
        return []

    cand_matrix = ncaa_index.norm_matrix[cand_idx]
    cand_norms = ncaa_index.norm_row_norms[cand_idx]
    t_norm = np.linalg.norm(target_profile.norm_vec)
    if t_norm < 1e-9:
        return []
    safe = np.where(cand_norms > 1e-9, cand_norms, 1.0)
    raw_sims = (cand_matrix @ target_profile.norm_vec) / (safe * t_norm)
    raw_sims = np.where(cand_norms > 1e-9, raw_sims, -1.0)
    raw_sims = np.clip(raw_sims, -1.0, 1.0)
    sims = (raw_sims + 1.0) / 2.0

    # Bucket penalty for adjacent (non-same) bucket -- vectorized.
    cand_buckets = ncaa_index.buckets[cand_idx]
    adjacency_penalty = np.where(cand_buckets != tbucket, bucket_penalty, 0.0)
    sims = np.maximum(0.0, sims - adjacency_penalty)

    # Top-K
    top_k_local = np.argsort(-sims)[:k]
    out: list[Comparable] = []
    for j in top_k_local:
        i = int(cand_idx[j])
        prof = ncaa_corpus.profiles[i]
        btv_pid = ncaa_index.pids[i]
        bucket = ncaa_index.buckets[i]
        rollup = _bridge_lookup(btv_pid, bridge_by_pid, nba_career_index)
        out.append(Comparable(
            comp_nba_id=rollup["nba_id"] or f"ncaa:{btv_pid}",
            comp_name=prof.name,
            comp_season=prof.season,
            comp_age_when_compared=prof.age,
            similarity=float(sims[j]),
            position_bucket=bucket,
            bucket_match=(bucket == tbucket),
            remaining_seasons=rollup["remaining_seasons"],
            remaining_games=rollup["remaining_games"],
            remaining_fantasy_ppg_dhk=rollup["fantasy_ppg_dhk"],
            remaining_fantasy_ppg_default=rollup["fantasy_ppg_default"],
            last_age=rollup["last_age"],
            censored=rollup["censored"],
            ages_after=[a for a, _ in rollup["seasons_after_age"]],
        ))
    return out


# When a college comp's NBA career is censored (player still active),
# their realized seasons are a LOWER bound on their career length. We
# extrapolate to a typical NBA career length given (a) their realized
# seasons so far, (b) their last observed age. The extrapolation
# assumes the average peak-NBA exit age is 32 for stars (high fppg)
# and 28 for role players (low fppg). For each censored comp we set
# their effective ``remaining_seasons`` to MAX(realized, ext_age - debut_age).
DEFAULT_EXIT_AGE_STAR = 34.0
DEFAULT_EXIT_AGE_ROLE = 30.0
DEFAULT_EXIT_AGE_BENCH = 27.0
STAR_FPPG_DHK = 18.0     # above which we treat the comp as a star
ROLE_FPPG_DHK = 8.0      # above which we treat the comp as a rotation player


def _extrapolate_censored(comp: Comparable) -> Comparable:
    """For censored comps, replace remaining_seasons with an extrapolation.

    A college comp who's a 25-year-old NBA star with 4 censored seasons
    realistically has ~7 more seasons in them. The rookie engine needs
    to credit Flagg's projection with the comp's expected FULL career,
    not just what's been observed so far.
    """
    if not comp.censored or comp.last_age is None:
        return comp
    fppg = max(
        comp.remaining_fantasy_ppg_dhk or 0,
        comp.remaining_fantasy_ppg_default or 0,
    )
    if fppg >= STAR_FPPG_DHK:
        exit_age = DEFAULT_EXIT_AGE_STAR
    elif fppg >= ROLE_FPPG_DHK:
        exit_age = DEFAULT_EXIT_AGE_ROLE
    else:
        exit_age = DEFAULT_EXIT_AGE_BENCH
    # NBA debut age = last_age - (remaining_seasons - 1).
    debut_age = comp.last_age - max(0, comp.remaining_seasons - 1)
    extrapolated = max(comp.remaining_seasons, int(round(exit_age - debut_age + 1)))
    if extrapolated == comp.remaining_seasons:
        return comp
    # Build a new Comparable with the bumped count. Games are scaled
    # proportionally so projection PV math stays consistent.
    if comp.remaining_seasons > 0:
        games_per_season = comp.remaining_games / comp.remaining_seasons
    else:
        games_per_season = 65  # league average for an NBA regular
    new_games = int(round(games_per_season * extrapolated))
    new_ages = list(comp.ages_after)
    # Extend ages_after forward year-by-year up to exit_age (used for
    # survival probability calculation).
    cur = max(new_ages) if new_ages else (comp.last_age or debut_age)
    while cur < exit_age:
        cur = round(cur + 1.0, 1)
        new_ages.append(cur)
    return Comparable(
        comp_nba_id=comp.comp_nba_id,
        comp_name=comp.comp_name,
        comp_season=comp.comp_season,
        comp_age_when_compared=comp.comp_age_when_compared,
        similarity=comp.similarity,
        position_bucket=comp.position_bucket,
        bucket_match=comp.bucket_match,
        remaining_seasons=extrapolated,
        remaining_games=new_games,
        remaining_fantasy_ppg_dhk=comp.remaining_fantasy_ppg_dhk,
        remaining_fantasy_ppg_default=comp.remaining_fantasy_ppg_default,
        last_age=comp.last_age,
        censored=comp.censored,
        ages_after=new_ages,
    )


def project_rookie(
    *,
    btv_pid: str,
    name: str,
    age: float,
    class_year: Optional[str],
    ncaa_profile: Profile,
    ncaa_corpus: CorpusProfiles,
    ncaa_index: NCAACorpusSearchIndex,
    bridge_by_pid: dict,
    nba_career_index: CareerIndex,
    league_format: str = "points_dhk",
    k: int = DEFAULT_K,
    age_window: float = DEFAULT_AGE_WINDOW,
) -> tuple[CareerProjection, list[Comparable]]:
    """Find college comps, chain to NBA careers, project rookie value.

    Returns ``(CareerProjection, comparables_list)``. The Comparable
    objects' ``comp_nba_id`` is either the bridged NBA id (when the
    comp made the league) or ``"ncaa:<btv_pid>"`` (when they didn't).
    Either way the projection layer handles them correctly because
    ``remaining_seasons=0`` zeros their contribution to longevity.

    Censored comps (still-active NBA players) get their remaining
    seasons extrapolated to a typical exit age (32 for stars, 28 for
    role players) so a comp like Jayson Tatum doesn't get capped at
    his realized 8 seasons.
    """
    comps = find_college_comparables(
        target_profile=ncaa_profile,
        target_class=class_year,
        ncaa_corpus=ncaa_corpus,
        ncaa_index=ncaa_index,
        bridge_by_pid=bridge_by_pid,
        nba_career_index=nba_career_index,
        k=k,
        age_window=age_window,
        exclude_btv_pid=btv_pid,
    )
    # Extrapolate censored comps (still-active careers).
    comps = [_extrapolate_censored(c) for c in comps]

    # Longevity median is computed over comps WITH an NBA career.
    # College comps who washed out should NOT drag down the median --
    # the spec is explicit: "weight their contribution to longevity
    # as zero". We split the projection:
    #   * Longevity (projected_remaining_years) -> only NBA-having comps
    #   * Fantasy points + survival probability -> ALL comps (a comp
    #     who didn't make the league correctly contributes zero PV
    #     points and zero probability of being in the NBA at age+N).
    nba_comps = [c for c in comps if c.remaining_seasons > 0]
    if nba_comps:
        # Project years over the NBA-having subset, then transplant
        # the result onto the full-comp projection.
        proj_years_only = project_career(
            nba_id=btv_pid, name=name, age=age,
            comparables=nba_comps, league_format=league_format,
        )
        proj = project_career(
            nba_id=btv_pid, name=name, age=age,
            comparables=comps, league_format=league_format,
        )
        proj.projected_remaining_years = proj_years_only.projected_remaining_years
    else:
        proj = project_career(
            nba_id=btv_pid, name=name, age=age,
            comparables=comps, league_format=league_format,
        )
    return proj, comps


def blended_dynasty_value(
    *,
    rookie_dv: Optional[float],
    nba_dv: Optional[float],
    n_nba_seasons: int,
) -> Optional[float]:
    """Blend rookie projection with NBA projection based on NBA sample size.

    Logic from PR #7 spec:
      * 0 NBA seasons -> rookie_dv only
      * 1 NBA season  -> 0.5 * rookie_dv + 0.5 * nba_dv (NBA sample noisy,
                          college comps still valuable)
      * 2+ NBA seasons -> nba_dv only (PR #4 behavior)

    Returns None when both inputs are None, otherwise prefers whichever
    is available.
    """
    if n_nba_seasons <= 0:
        return rookie_dv if rookie_dv is not None else None
    if n_nba_seasons == 1:
        if rookie_dv is None:
            return nba_dv
        if nba_dv is None:
            return rookie_dv
        return 0.5 * rookie_dv + 0.5 * nba_dv
    # n_nba_seasons >= 2
    return nba_dv if nba_dv is not None else rookie_dv
