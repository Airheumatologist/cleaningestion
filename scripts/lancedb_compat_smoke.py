#!/usr/bin/env python3
"""Phase 1 LanceDB compatibility smoke checks (sync API, pinned version)."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import timedelta
from typing import Any

import lancedb
from lancedb.rerankers import RRFReranker

EXPECTED_LANCEDB_VERSION = "0.29.2"


def _sample_rows() -> list[dict[str, Any]]:
    return [
        {
            "vector": [0.11, 0.22, 0.33, 0.44],
            "page_content": "Aspirin dosing guidance for adults with fever",
            "source": "pubmed_abstract",
            "source_family": "pubmed",
            "year": 2024,
            "article_type": "review",
            "doc_id": "pmid-1",
            "chunk_id": "pmid-1-0",
        },
        {
            "vector": [0.12, 0.19, 0.36, 0.41],
            "page_content": "Aspirin contraindications from DailyMed label",
            "source": "dailymed",
            "source_family": "dailymed",
            "year": 2023,
            "article_type": "label",
            "doc_id": "setid-2",
            "chunk_id": "setid-2-0",
        },
        {
            "vector": [0.81, 0.74, 0.63, 0.59],
            "page_content": "Immunotherapy outcomes in metastatic melanoma",
            "source": "pmc_oa",
            "source_family": "pmc",
            "year": 2022,
            "article_type": "trial",
            "doc_id": "pmc-3",
            "chunk_id": "pmc-3-0",
        },
    ]


def _wait_for_all_indices(table: Any) -> list[str]:
    names = [index.name for index in table.list_indices()]
    if names:
        table.wait_for_index(names, timeout=timedelta(seconds=60))
    return names


def _create_common_indexes(table: Any) -> None:
    table.create_fts_index("page_content", replace=True)
    table.create_scalar_index("source", replace=True)
    table.create_scalar_index("source_family", replace=True)
    table.create_scalar_index("year", replace=True)
    table.create_scalar_index("article_type", replace=True)


def run_smoke_checks() -> dict[str, Any]:
    installed_version = getattr(lancedb, "__version__", "unknown")
    if installed_version != EXPECTED_LANCEDB_VERSION:
        raise RuntimeError(
            f"Expected lancedb=={EXPECTED_LANCEDB_VERSION}, found {installed_version}. "
            "Pin/update dependencies before migration work."
        )

    checks: dict[str, Any] = {"installed_version": installed_version}

    with tempfile.TemporaryDirectory(prefix="lancedb-smoke-") as db_dir:
        db = lancedb.connect(db_dir)

        table_rq = db.create_table("medical_rq", data=_sample_rows(), mode="overwrite")
        _create_common_indexes(table_rq)
        table_rq.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_RQ",
            num_partitions=2,
            num_sub_vectors=2,
            replace=True,
        )
        rq_index_names = _wait_for_all_indices(table_rq)
        checks["rq_index_names"] = rq_index_names

        vector_results = (
            table_rq.search([0.11, 0.22, 0.33, 0.44], query_type="vector")
            .where("source_family = 'pubmed'", prefilter=True)
            .limit(3)
            .to_list()
        )
        if not vector_results:
            raise RuntimeError("Vector prefilter query returned no rows.")

        fts_results = (
            table_rq.search("aspirin contraindications", query_type="fts")
            .where("source = 'dailymed'", prefilter=True)
            .limit(3)
            .to_list()
        )
        if not fts_results:
            raise RuntimeError("FTS prefilter query returned no rows.")

        hybrid_results = (
            table_rq.search(query_type="hybrid")
            .vector([0.11, 0.22, 0.33, 0.44])
            .text("aspirin adults")
            .where("year >= 2023", prefilter=True)
            .rerank(reranker=RRFReranker())
            .limit(5)
            .to_list()
        )
        if not hybrid_results or "_relevance_score" not in hybrid_results[0]:
            raise RuntimeError("Hybrid + RRF query did not return expected ranking metadata.")

        plan_text = (
            table_rq.search(query_type="hybrid")
            .vector([0.11, 0.22, 0.33, 0.44])
            .text("aspirin adults")
            .where("source_family = 'pubmed'", prefilter=True)
            .limit(5)
            .explain_plan()
        )
        if "ANNSubIndex" not in plan_text or "MatchQuery" not in plan_text:
            raise RuntimeError("Query plan did not show ANN and FTS operators.")
        checks["planner_has_ann_and_fts"] = True

        table_hnsw = db.create_table("medical_hnsw_sq", data=_sample_rows(), mode="overwrite")
        table_hnsw.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_HNSW_SQ",
            num_partitions=1,
            m=8,
            ef_construction=80,
            replace=True,
        )
        hnsw_index_names = _wait_for_all_indices(table_hnsw)
        checks["hnsw_index_names"] = hnsw_index_names

        hnsw_results = (
            table_hnsw.search([0.11, 0.22, 0.33, 0.44], query_type="vector")
            .limit(1)
            .to_list()
        )
        if not hnsw_results:
            raise RuntimeError("IVF_HNSW_SQ vector query returned no rows.")

        checks["vector_prefilter_ok"] = True
        checks["fts_prefilter_ok"] = True
        checks["hybrid_rrf_ok"] = True
        checks["ivf_hnsw_sq_ok"] = True
        checks["sample_top_doc_ids"] = {
            "vector": vector_results[0]["doc_id"],
            "fts": fts_results[0]["doc_id"],
            "hybrid": hybrid_results[0]["doc_id"],
            "hnsw": hnsw_results[0]["doc_id"],
        }

    checks["status"] = "ok"
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LanceDB API compatibility smoke checks.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    args = parser.parse_args()

    checks = run_smoke_checks()
    if args.json:
        print(json.dumps(checks, indent=2, sort_keys=True))
    else:
        print(f"LanceDB compatibility smoke passed for lancedb=={checks['installed_version']}")
        print(f"IVF_RQ indexes: {', '.join(checks['rq_index_names'])}")
        print(f"IVF_HNSW_SQ indexes: {', '.join(checks['hnsw_index_names'])}")
        print(f"Planner check (ANN + FTS): {checks['planner_has_ann_and_fts']}")
        print(f"Top docs: {checks['sample_top_doc_ids']}")


if __name__ == "__main__":
    main()
