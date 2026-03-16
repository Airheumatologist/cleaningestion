#!/usr/bin/env python3
"""Phase 2 schema parity validator for PubMed, PMC, and DailyMed LanceDB rows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "schema" / "lancedb_schema_contract.json"
SPEC_PATH = REPO_ROOT / "lancedb_schema_parity_spec.md"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "schema" / "lancedb_index_profiles.json"

FIELD_DESCRIPTIONS = {
    "point_id": "Original Qdrant point ID",
    "vector": "Dense embedding vector (Qwen3 expected dimension)",
    "sparse_indices": "BM25 sparse vector indices",
    "sparse_values": "BM25 sparse vector values",
}


def _prepare_import_paths() -> None:
    for path in (REPO_ROOT, REPO_ROOT / "scripts"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEmbeddingProvider:
    """Deterministic embedding provider with ingestion-compatible output dimension."""

    DIMENSION = 1024

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for idx, text in enumerate(texts):
            seed = (len(text) + idx) % 97
            vectors.append([float((seed + offset) % 97) / 97.0 for offset in range(self.DIMENSION)])
        return vectors


class _FakeSparseEncoder:
    def encode_batch(self, texts: list[str]) -> list[Any]:
        return [self.encode_document(text) for text in texts]

    def encode_document(self, text: str) -> Any:
        from qdrant_client.models import SparseVector

        return SparseVector(indices=[1, 2 + (len(text) % 13)], values=[0.8, 0.2])


def _signature(value: Any, *, depth: int = 0) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        if not value:
            return "list<empty>"
        item_types = sorted({_signature(item, depth=depth + 1) for item in value})
        return f"list<{ '|'.join(item_types) }>"
    if isinstance(value, dict):
        if depth >= 2:
            return "dict"
        if not value:
            return "dict{}"
        items = ",".join(f"{k}:{_signature(v, depth=depth + 1)}" for k, v in sorted(value.items()))
        return f"dict{{{items}}}"
    return type(value).__name__


def _build_pubmed_points(pubmed_module: Any) -> list[Any]:
    article_full = {
        "pmid": "12345678",
        "pmc": "PMC1234567",
        "doi": "10.1000/pubmed.full",
        "pii": "S1234-5678(26)00001-0",
        "other_ids": {"mid": "MID-100", "pmcpid": "PMCPID-1"},
        "title": "Aspirin dosing and contraindications in adults",
        "abstract": " ".join(["Structured abstract content."] * 120),
        "is_gov_affiliated": True,
        "gov_agencies": ["NIH", "CDC"],
        "abstract_structured": [
            {"label": "BACKGROUND", "text": "Background text"},
            {"label": "METHODS", "text": "Methods text"},
        ],
        "has_structured_abstract": True,
        "journal": "The Lancet",
        "journal_full": {
            "title": "The Lancet",
            "abbreviation": "Lancet",
            "publisher": "Elsevier",
            "issn_print": "0140-6736",
            "issn_electronic": "1474-547X",
            "nlm_unique_id": "2985213R",
            "volume": "410",
            "issue": "10001",
        },
        "publication_date": {"year": 2024, "month": 11, "day": 4},
        "year": 2024,
        "article_type": "review",
        "publication_types": [{"type": "Review", "ui": "D016454"}],
        "mesh_terms": [{"descriptor": "Aspirin", "qualifiers": ["therapeutic use"]}],
        "keywords": [{"keyword": "aspirin", "major_yn": "Y"}],
    }
    article_sparse = {
        "pmid": "999",
        "title": "Short title",
        "abstract": "Short abstract",
        "article_type": "other",
        "is_gov_affiliated": False,
        "gov_agencies": [],
    }
    points, _ids = pubmed_module.build_points(
        batch=[article_full, article_sparse],
        embedding_provider=_FakeEmbeddingProvider(),
        sparse_encoder=_FakeSparseEncoder(),
        validate_chunks=False,
        dedup_chunks=False,
    )
    return points


def _build_pmc_points(pmc_module: Any) -> list[Any]:
    oa_article = {
        "_source_type": "pmc_oa",
        "metadata": {
            "identifiers": {"pmcid": "PMC100", "pmid": "100", "doi": "10.1000/pmc.oa"},
            "publication": {
                "year": 2023,
                "country": "US",
                "journal": {"title": "NEJM", "nlm_unique_id": "0255562"},
            },
            "content_flags": {"has_full_text": True, "is_open_access": True, "license": "cc-by"},
            "classification": {"keywords": ["aspirin"], "mesh_terms": ["Aspirin"]},
            "article_type": "review",
        },
        "content": {
            "title": "PMC OA sample",
            "sections": [{"title": "Methods", "type": "methods", "content": " ".join(["method"] * 100)}],
            "tables": [{"content": "Dose | Outcome", "caption_title": "Dose table", "label": "Table 1", "id": "t1"}],
        },
        "abstract": " ".join(["oa abstract"] * 60),
        "publication_type_list": ["Review"],
        "evidence_grade": "B",
        "evidence_level": 2,
        "evidence_term": "review",
        "evidence_source": "article_type",
    }
    author_article = {
        "_source_type": "pmc_author_manuscript",
        "metadata": {
            "identifiers": {
                "pmcid": "PMC200",
                "pmid": "200",
                "doi": "10.1000/pmc.author",
                "nihms_id": "NIHMS12345",
            },
            "publication": {
                "year": 2021,
                "country": "US",
                "journal": {"title": "JAMA", "nlm_unique_id": "0375267"},
            },
            "content_flags": {"has_full_text": True, "is_open_access": False, "license": "author-manuscript"},
            "classification": {"keywords": ["metformin"], "mesh_terms": ["Diabetes Mellitus"]},
            "article_type": "trial",
        },
        "content": {
            "title": "PMC author manuscript sample",
            "sections": [{"title": "Results", "type": "results", "content": " ".join(["result"] * 100)}],
            "tables": [],
        },
        "abstract": " ".join(["author abstract"] * 60),
        "publication_type_list": ["Clinical Trial"],
        "evidence_grade": "A",
        "evidence_level": 1,
        "evidence_term": "trial",
        "evidence_source": "article_type",
    }
    points, _chunk_ids = pmc_module.build_points(
        batch=[oa_article, author_article],
        embedding_provider=_FakeEmbeddingProvider(),
        sparse_encoder=_FakeSparseEncoder(),
        validate_chunks=False,
        dedup_chunks=False,
    )
    return points


def _build_dailymed_points(dailymed_module: Any) -> list[Any]:
    drug = {
        "set_id": "set-123",
        "drug_name": "Acetaminophen",
        "title": "Acetaminophen Label",
        "active_ingredients": ["Acetaminophen"],
        "manufacturer": "Example Pharma",
        "label_type_code": "HUMAN_PRESCRIPTION_DRUG_LABEL",
        "label_type_display": "Human Prescription Drug Label",
        "sections": {
            "indications": {
                "title": "Indications and Usage",
                "text": " ".join(["Use for pain relief."] * 40),
                "has_tables": True,
                "tables": [{"index": 0, "content": "Age | Dose\nAdult | 500mg"}],
            },
            "warnings": {
                "title": "Warnings and Precautions",
                "text": " ".join(["Risk of liver injury at high dose."] * 35),
                "has_tables": False,
                "tables": [],
            },
        },
    }
    chunker = dailymed_module.get_shared_chunker(chunker_class=dailymed_module.CHUNKER_CLASS)
    chunks = dailymed_module.create_chunks(drug, chunker, validate_chunks=False)
    points, _chunk_ids = dailymed_module.build_points(
        chunks=chunks,
        embedding_provider=_FakeEmbeddingProvider(),
        sparse_encoder=_FakeSparseEncoder(),
        validate_chunks=False,
        dedup_chunks=False,
    )
    return points


def collect_rows_by_source() -> dict[str, list[dict[str, Any]]]:
    _prepare_import_paths()
    from lancedb_ingestion_sink import points_to_lancedb_rows

    pubmed_module = _load_module("ingest_pubmed", REPO_ROOT / "scripts" / "21_ingest_pubmed_abstracts.py")
    pmc_module = _load_module("ingest_pmc", REPO_ROOT / "scripts" / "06_ingest_pmc.py")
    dailymed_module = _load_module("ingest_dailymed", REPO_ROOT / "scripts" / "07_ingest_dailymed.py")

    all_points: list[Any] = []
    all_points.extend(_build_pubmed_points(pubmed_module))
    all_points.extend(_build_pmc_points(pmc_module))
    all_points.extend(_build_dailymed_points(dailymed_module))

    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in points_to_lancedb_rows(all_points):
        source = str(row.get("source", "")).strip()
        if source:
            rows_by_source[source].append(row)

    return dict(sorted(rows_by_source.items()))


def build_contract(rows_by_source: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema_basis": "lancedb_rows_via_ingestion_sink",
        "sources": {},
    }
    for source, rows in rows_by_source.items():
        field_obs: dict[str, dict[str, Any]] = {}
        for row in rows:
            for key, value in row.items():
                if key not in field_obs:
                    field_obs[key] = {"count": 0, "types": set()}
                field_obs[key]["count"] += 1
                field_obs[key]["types"].add(_signature(value))

        total_rows = len(rows)
        fields: dict[str, Any] = {}
        for key in sorted(field_obs):
            types_sorted = sorted(field_obs[key]["types"])
            field_info: dict[str, Any] = {
                "required": field_obs[key]["count"] == total_rows,
                "nullable": "null" in types_sorted,
                "types": types_sorted,
            }
            if key in FIELD_DESCRIPTIONS:
                field_info["description"] = FIELD_DESCRIPTIONS[key]
            fields[key] = field_info

        contract["sources"][source] = {
            "sample_payload_count": total_rows,
            "field_count": len(fields),
            "fields": fields,
        }

    return contract


def _load_vector_runtime_defaults() -> tuple[int, int]:
    _prepare_import_paths()
    from config_ingestion import IngestionConfig
    from src.config import EMBEDDING_DIMENSION

    return int(IngestionConfig.get_vector_size()), int(EMBEDDING_DIMENSION)


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_vector_configuration(
    rows_by_source: dict[str, list[dict[str, Any]]],
    expected_vector_dim: int,
    runtime_vector_dim: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    observed_dims_by_source: dict[str, list[int]] = {}

    if runtime_vector_dim != expected_vector_dim:
        errors.append(
            f"runtime vector dimension mismatch expected={expected_vector_dim} runtime={runtime_vector_dim}"
        )

    for source, rows in sorted(rows_by_source.items()):
        dims: set[int] = set()
        for row in rows:
            vector = row.get("vector")
            if not isinstance(vector, list) or not vector:
                errors.append(f"{source}: missing/empty vector field")
                continue
            if not all(isinstance(item, (int, float)) for item in vector):
                errors.append(f"{source}: vector contains non-numeric values")
                continue
            dims.add(len(vector))

        observed_dims = sorted(dims)
        observed_dims_by_source[source] = observed_dims
        for dim in observed_dims:
            if dim != expected_vector_dim:
                errors.append(
                    f"{source}: vector dimension mismatch expected={expected_vector_dim} observed={dim}"
                )

    manifest_vector_column = str(manifest.get("vector_column", ""))
    if manifest_vector_column != "vector":
        errors.append(
            f"manifest vector_column mismatch expected='vector' observed='{manifest_vector_column}'"
        )

    profile_checks: list[dict[str, Any]] = []
    profiles = manifest.get("profiles", {})
    for profile_name, profile in sorted(profiles.items()):
        num_sub_vectors = profile.get("num_sub_vectors")
        compatible = True
        if isinstance(num_sub_vectors, int) and num_sub_vectors > 0:
            compatible = expected_vector_dim % num_sub_vectors == 0
            if not compatible:
                errors.append(
                    f"profile {profile_name}: num_sub_vectors={num_sub_vectors} incompatible with vector_dim={expected_vector_dim}"
                )
        profile_checks.append(
            {
                "profile": profile_name,
                "num_sub_vectors": num_sub_vectors,
                "compatible": compatible,
            }
        )

    return {
        "status": "ok" if not errors else "failed",
        "expected_vector_dim": expected_vector_dim,
        "runtime_vector_dim": runtime_vector_dim,
        "runtime_dim_matches_ingestion": runtime_vector_dim == expected_vector_dim,
        "manifest_vector_column": manifest_vector_column,
        "observed_vector_dims_by_source": observed_dims_by_source,
        "profile_compatibility": profile_checks,
        "errors": errors,
    }


def compare_contracts(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    exp_sources = set(expected.get("sources", {}).keys())
    obs_sources = set(observed.get("sources", {}).keys())
    if exp_sources != obs_sources:
        errors.append(f"source mismatch expected={sorted(exp_sources)} observed={sorted(obs_sources)}")

    for source in sorted(exp_sources & obs_sources):
        exp_fields = expected["sources"][source]["fields"]
        obs_fields = observed["sources"][source]["fields"]
        exp_keys = set(exp_fields.keys())
        obs_keys = set(obs_fields.keys())
        if exp_keys != obs_keys:
            errors.append(
                f"{source}: field mismatch expected_only={sorted(exp_keys - obs_keys)} observed_only={sorted(obs_keys - exp_keys)}"
            )
            continue

        for key in sorted(exp_keys):
            if exp_fields[key]["types"] != obs_fields[key]["types"]:
                errors.append(
                    f"{source}.{key}: type drift expected={exp_fields[key]['types']} observed={obs_fields[key]['types']}"
                )
            if exp_fields[key]["nullable"] != obs_fields[key]["nullable"]:
                errors.append(
                    f"{source}.{key}: nullable drift expected={exp_fields[key]['nullable']} observed={obs_fields[key]['nullable']}"
                )
            if exp_fields[key]["required"] != obs_fields[key]["required"]:
                errors.append(
                    f"{source}.{key}: required drift expected={exp_fields[key]['required']} observed={obs_fields[key]['required']}"
                )

    return errors


def render_markdown_spec(contract: dict[str, Any]) -> str:
    sources = sorted(contract.get("sources", {}).keys())
    all_fields: set[str] = set()
    for source in sources:
        all_fields.update(contract["sources"][source]["fields"].keys())

    lines: list[str] = []
    lines.append("# LanceDB Schema Parity Spec (Phase 2)")
    lines.append("")
    lines.append("This spec is generated from `schema/lancedb_schema_contract.json` and enforced by CI.")
    lines.append("")
    lines.append("## Source-specific constraints")
    lines.append("")
    for source in sources:
        info = contract["sources"][source]
        lines.append(f"- `{source}`: `{info['field_count']}` fields, `{info['sample_payload_count']}` sampled payloads")
    lines.append("")

    vector_expectations = contract.get("vector_expectations", {})
    if vector_expectations:
        lines.append("## Vector Expectations")
        lines.append("")
        lines.append(f"- Expected vector dimension: `{vector_expectations.get('expected_vector_dim')}`")
        lines.append(f"- Runtime vector dimension: `{vector_expectations.get('runtime_vector_dim')}`")
        lines.append(f"- Manifest vector column: `{vector_expectations.get('manifest_vector_column')}`")
        lines.append("")

    lines.append("## Field-by-field parity table")
    lines.append("")
    header = "| Field | " + " | ".join(sources) + " |"
    divider = "|---|" + "|".join(["---"] * len(sources)) + "|"
    lines.append(header)
    lines.append(divider)
    for field in sorted(all_fields):
        cells = [field]
        for source in sources:
            src_fields = contract["sources"][source]["fields"]
            if field not in src_fields:
                cells.append("-")
                continue
            rule = src_fields[field]
            nullability = "nullable" if rule["nullable"] else "non-null"
            required = "required" if rule["required"] else "optional"
            types = ", ".join(rule["types"])
            cells.append(f"`{types}` ({required}, {nullability})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="LanceDB schema parity validator.")
    parser.add_argument("--write-contract", action="store_true", help="Write observed contract snapshot to disk.")
    parser.add_argument("--write-spec", action="store_true", help="Write Markdown parity spec from observed contract.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Path to index profile manifest.")
    parser.add_argument("--expected-vector-dim", type=int, default=None, help="Override expected vector dimension.")
    parser.add_argument("--runtime-vector-dim", type=int, default=None, help="Override runtime vector dimension.")
    args = parser.parse_args()

    rows_by_source = collect_rows_by_source()
    observed_contract = build_contract(rows_by_source)
    manifest = _load_manifest(Path(args.manifest))

    default_expected, default_runtime = _load_vector_runtime_defaults()
    expected_vector_dim = args.expected_vector_dim if args.expected_vector_dim is not None else default_expected
    runtime_vector_dim = args.runtime_vector_dim if args.runtime_vector_dim is not None else default_runtime

    vector_validation = validate_vector_configuration(
        rows_by_source=rows_by_source,
        expected_vector_dim=expected_vector_dim,
        runtime_vector_dim=runtime_vector_dim,
        manifest=manifest,
    )
    observed_contract["vector_expectations"] = {
        "expected_vector_dim": expected_vector_dim,
        "runtime_vector_dim": runtime_vector_dim,
        "manifest_vector_column": manifest.get("vector_column"),
        "validation_status": vector_validation["status"],
    }

    if args.write_contract:
        if vector_validation["errors"]:
            raise RuntimeError(
                "Refusing to write schema contract while vector validation fails: "
                + "; ".join(vector_validation["errors"])
            )
        CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONTRACT_PATH.write_text(json.dumps(observed_contract, indent=2, sort_keys=True), encoding="utf-8")
        output = {
            "status": "written",
            "contract_path": str(CONTRACT_PATH),
            "source_counts": {source: len(rows) for source, rows in rows_by_source.items()},
            "vector_validation": vector_validation,
        }
        print(json.dumps(output, indent=2, sort_keys=True) if args.json else output)
        return

    if args.write_spec:
        markdown = render_markdown_spec(observed_contract)
        SPEC_PATH.write_text(markdown, encoding="utf-8")
        output = {
            "status": "spec_written",
            "spec_path": str(SPEC_PATH),
            "contract_path": str(CONTRACT_PATH),
            "vector_validation": vector_validation,
        }
        print(json.dumps(output, indent=2, sort_keys=True) if args.json else output)
        return

    if not CONTRACT_PATH.exists():
        raise RuntimeError(f"Schema contract not found at {CONTRACT_PATH}. Run with --write-contract first.")

    expected_contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    errors = compare_contracts(expected_contract, observed_contract)
    errors.extend(f"vector_validation: {err}" for err in vector_validation["errors"])

    result = {
        "status": "ok" if not errors else "schema_drift",
        "contract_path": str(CONTRACT_PATH),
        "source_counts": {source: len(rows) for source, rows in rows_by_source.items()},
        "vector_validation": vector_validation,
        "errors": errors,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
