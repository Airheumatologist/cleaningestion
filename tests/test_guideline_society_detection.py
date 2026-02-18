from src.specialty_journals import detect_guideline_society_signal


def test_detects_society_guideline_signal_from_title():
    signal = detect_guideline_society_signal(
        title="2024 ESC Guidelines for the management of heart failure",
        journal="European Heart Journal",
    )

    assert signal["is_match"] is True
    assert "ESC" in signal["matched_societies"]
    assert signal["has_guideline_context"] is True


def test_detects_kdigo_clinical_practice_guideline():
    signal = detect_guideline_society_signal(
        title="KDIGO clinical practice guideline for chronic kidney disease",
        journal="Kidney International",
    )

    assert signal["is_match"] is True
    assert "KDIGO" in signal["matched_societies"]


def test_society_without_guideline_context_does_not_match():
    signal = detect_guideline_society_signal(
        title="ESC registry outcomes in acute heart failure",
        journal="European Heart Journal",
    )

    assert signal["is_match"] is False
    assert signal["has_guideline_context"] is False


def test_ambiguous_acronym_without_society_cue_is_rejected():
    signal = detect_guideline_society_signal(
        title="ADA deficiency management guidelines in pediatric patients",
        journal="Pediatrics",
    )

    assert signal["is_match"] is False
    assert "ADA" not in signal["matched_societies"]
