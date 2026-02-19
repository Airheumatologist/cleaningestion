"""Shared helpers for DailyMed RX-only update and ingestion filtering."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Optional, Sequence, Set, Tuple

HL7_NS = {"hl7": "urn:hl7-org:v3"}

RX_LABEL_TYPE_CODE = "34391-3"
RX_LABEL_DISPLAY_TOKEN = "human prescription drug label"
INCREMENTAL_ALLOWED_NESTED_ROOTS = {"prescription"}


def _normalize_space(value: str) -> str:
    return " ".join((value or "").split()).strip()


def normalize_label_display(value: str) -> str:
    """Normalize label display string for robust matching."""
    return _normalize_space(value).lower()


def is_human_prescription_label(label_type_code: str, label_type_display: str) -> bool:
    """Return True when the SPL document represents a human RX label."""
    code = _normalize_space(label_type_code)
    display = normalize_label_display(label_type_display)
    return code == RX_LABEL_TYPE_CODE or RX_LABEL_DISPLAY_TOKEN in display


def get_nested_member_root(member_name: str) -> str:
    """Return normalized top-level folder/root for a ZIP member name."""
    normalized = (member_name or "").replace("\\", "/").strip("/")
    if not normalized:
        return ""
    return normalized.split("/", 1)[0].strip().lower()


def summarize_nested_member_roots(member_names: Iterable[str]) -> Counter[str]:
    """Count nested ZIP members grouped by top-level root."""
    counts: Counter[str] = Counter()
    for member_name in member_names:
        root = get_nested_member_root(member_name) or "(root)"
        counts[root] += 1
    return counts


def select_nested_zip_members(
    member_names: Sequence[str],
    allowed_roots: Optional[Set[str]] = None,
) -> Tuple[list[str], Counter[str], bool]:
    """Filter nested ZIP members by root with fallback for structure drift.

    Returns:
        (selected_members, root_counts, used_fallback)
    """
    root_counts = summarize_nested_member_roots(member_names)
    normalized_allowed = {
        root.strip().lower()
        for root in (allowed_roots or set())
        if root and root.strip()
    }
    if not normalized_allowed:
        return list(member_names), root_counts, False

    selected = [
        member_name
        for member_name in member_names
        if get_nested_member_root(member_name) in normalized_allowed
    ]
    used_fallback = bool(member_names) and not selected
    if used_fallback:
        selected = list(member_names)
    return selected, root_counts, used_fallback


def extract_document_label_type(
    root: object,
    namespaces: Optional[Mapping[str, str]] = None,
) -> Tuple[str, str]:
    """Extract document-level label type code/display from SPL root element."""
    ns = dict(namespaces or HL7_NS)

    # lxml path
    if hasattr(root, "xpath"):
        try:
            code_nodes = root.xpath("./hl7:code", namespaces=ns)
            if not code_nodes:
                code_nodes = root.xpath("./code")
            if code_nodes:
                node = code_nodes[0]
                return (node.get("code", "") or "").strip(), (node.get("displayName", "") or "").strip()
        except Exception:
            return "", ""

    # xml.etree path
    if hasattr(root, "find"):
        try:
            node = root.find("hl7:code", ns)
            if node is None:
                node = root.find("code")
            if node is None:
                return "", ""
            return (node.attrib.get("code", "") or "").strip(), (node.attrib.get("displayName", "") or "").strip()
        except Exception:
            return "", ""

    return "", ""
