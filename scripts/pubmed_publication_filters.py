"""Shared PubMed publication type filtering and mapping helpers."""

from __future__ import annotations

from typing import List, Optional, Set

# Target publication types (case-insensitive matching)
# These are mapped to article_type values compatible with reranker.py tiers.
TARGET_PUBLICATION_TYPES = {
    "practice guideline": "practice_guideline",
    "meta-analysis": "meta_analysis",
    "systematic review": "systematic_review",
    "randomized controlled trial": "randomized_controlled_trial",
    "clinical trial": "clinical_trial",
    "clinical trial, phase i": "clinical_trial_phase_i",
    "clinical trial, phase ii": "clinical_trial_phase_ii",
    "clinical trial, phase iii": "clinical_trial_phase_iii",
    "clinical trial, phase iv": "clinical_trial_phase_iv",
    "controlled clinical trial": "controlled_clinical_trial",
    "multicenter study": "multicenter_study",
    "review": "review",
}

# Priority order ensures highest-value type is returned first.
PUBLICATION_TYPE_PRIORITY = [
    "practice guideline",
    "meta-analysis",
    "systematic review",
    "randomized controlled trial",
    "clinical trial, phase iii",
    "clinical trial, phase ii",
    "clinical trial, phase iv",
    "clinical trial, phase i",
    "controlled clinical trial",
    "clinical trial",
    "multicenter study",
    "review",
]


def _normalize_pub_types(pub_types: List[str]) -> Set[str]:
    return {pt.lower().strip() for pt in pub_types if pt}


def is_target_article(pub_types: List[str]) -> bool:
    """Check if an article has at least one target publication type."""
    pub_types_lower = _normalize_pub_types(pub_types)
    return any(target in pub_types_lower for target in TARGET_PUBLICATION_TYPES.keys())


def map_publication_type(pub_types: List[str]) -> Optional[str]:
    """
    Map PubMed publication types to reranker-compatible article_type.

    Returns:
        The highest-priority mapped article_type when matched, else None.
    """
    pub_types_lower = _normalize_pub_types(pub_types)
    for pub_type in PUBLICATION_TYPE_PRIORITY:
        if pub_type in pub_types_lower:
            return TARGET_PUBLICATION_TYPES[pub_type]
    return None
