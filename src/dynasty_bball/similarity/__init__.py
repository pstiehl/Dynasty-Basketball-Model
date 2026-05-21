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
    vectorize_college_season,
    build_college_corpus_profiles,
    COLLEGE_FEATURE_NAMES,
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
from .rookie import (
    NCAACorpusSearchIndex,
    prepare_ncaa_search_index,
    find_college_comparables,
    project_rookie,
    blended_dynasty_value,
)
from .bridge import (
    build_bridge,
    load_bridge,
    save_bridge,
    coverage_excluding_pre_corpus,
)

__all__ = [
    "Profile",
    "build_profile",
    "build_corpus_profiles",
    "zscore_normalize",
    "feature_names",
    "derive_position_bucket",
    "POSITION_BUCKETS",
    "vectorize_college_season",
    "build_college_corpus_profiles",
    "COLLEGE_FEATURE_NAMES",
    "Comparable",
    "find_comparables",
    "build_career_index",
    "prepare_corpus_for_search",
    "CorpusSearchIndex",
    "CareerProjection",
    "project_career",
    "project_all_current_players",
    "rescale_to_0_100",
    "NCAACorpusSearchIndex",
    "prepare_ncaa_search_index",
    "find_college_comparables",
    "project_rookie",
    "blended_dynasty_value",
    "build_bridge",
    "load_bridge",
    "save_bridge",
    "coverage_excluding_pre_corpus",
]
