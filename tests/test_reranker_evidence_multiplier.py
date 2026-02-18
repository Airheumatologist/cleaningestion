from src.reranker import (
    TIER_1_BOOST,
    TIER_3_BOOST,
    apply_evidence_boosts,
    get_evidence_multiplier,
)


def test_publication_type_practice_guideline_is_tier1():
    multiplier = get_evidence_multiplier(
        article_type="research_article",
        title="Long-term outcomes in CKD",
        publication_types=["Practice Guideline"],
    )
    assert multiplier == TIER_1_BOOST


def test_society_guideline_title_is_tier1():
    multiplier = get_evidence_multiplier(
        article_type="research_article",
        title="2024 ESC Guidelines for acute coronary syndrome",
        publication_types=[],
        journal="European Heart Journal",
    )
    assert multiplier == TIER_1_BOOST


def test_society_mention_without_guideline_language_not_tier1():
    multiplier = get_evidence_multiplier(
        article_type="research_article",
        title="ESC registry outcomes after PCI",
        publication_types=[],
        journal="European Heart Journal",
    )
    assert multiplier == TIER_3_BOOST


def test_apply_evidence_boosts_adds_society_debug_fields():
    docs = [{
        "title": "KDIGO clinical practice guideline for CKD",
        "journal": "Kidney International",
        "article_type": "research_article",
        "publication_type": [],
        "combined_score": 0.8,
        "year": 2024,
    }]

    boosted = apply_evidence_boosts(docs, current_year=2025)

    assert boosted[0]["guideline_society_match"] is True
    assert "KDIGO" in boosted[0]["matched_guideline_societies"]
