"""Career-arc similarity engine.

Submodules:

  * ``vectorize`` — turn a player-season into a profile vector
    (per-36 production rates + usage + efficiency + durability).
  * ``comparables`` — KNN over the historical corpus, age-windowed,
    position-bucketed.
  * ``projection`` — aggregate top-k comps into a dynasty value +
    per-year survival probabilities.

See ``docs/SIMILARITY-METHODOLOGY.md`` for the full design.
"""
from .vectorize import (
    Profile,
    build_profile,
    build_corpus_profiles,
    zscore_normalize,
    feature_names,
    derive_position_bucket,
    POSITION_BUCKETS,
)
from .comparables import (
    Comparable,
    find_comparables,
    build_career_index,
    prepare_corpus_for_search,
    CorpusSearchIndex,
)
from .projection import (
    CareerProjection,
    project_career,
    project_all_current_players,
    rescale_to_0_100,
)

__all__ = [
    "Profile",
    "build_profile",
    "build_corpus_profiles",
    "zscore_normalize",
    "feature_names",
    "derive_position_bucket",
    "POSITION_BUCKETS",
    "Comparable",
    "find_comparables",
    "build_career_index",
    "prepare_corpus_for_search",
    "CorpusSearchIndex",
    "CareerProjection",
    "project_career",
    "project_all_current_players",
    "rescale_to_0_100",
]
