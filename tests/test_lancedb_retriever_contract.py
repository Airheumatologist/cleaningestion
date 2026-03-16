import tempfile
import unittest

import lancedb

from src.retriever_lancedb import LanceDBRetriever


class LanceDBRetrieverContractTests(unittest.TestCase):
    def _make_retriever(self):
        tmp = tempfile.TemporaryDirectory(prefix="lancedb-retriever-test-")
        db = lancedb.connect(tmp.name)
        rows = [
            {
                "point_id": "1",
                "doc_id": "doc-1",
                "pmcid": "PMC1",
                "chunk_id": "chunk-1",
                "chunk_index": 0,
                "title": "Aspirin trial",
                "page_content": "aspirin trial outcomes",
                "abstract": "aspirin abstract",
                "source": "pmc_oa",
                "source_family": "pmc",
                "year": 2024,
                "article_type": "trial",
                "gov_agencies": ["NIH"],
                "vector": [0.1, 0.2, 0.3, 0.4],
            },
            {
                "point_id": "2",
                "doc_id": "doc-1",
                "pmcid": "PMC1",
                "chunk_id": "chunk-1b",
                "chunk_index": 1,
                "title": "Aspirin trial (section 2)",
                "page_content": "aspirin trial follow-up details",
                "abstract": "aspirin abstract",
                "source": "pmc_oa",
                "source_family": "pmc",
                "year": 2024,
                "article_type": "trial",
                "gov_agencies": ["CDC"],
                "vector": [0.12, 0.2, 0.29, 0.41],
            },
            {
                "point_id": "3",
                "doc_id": "doc-2",
                "pmcid": "PMC2",
                "chunk_id": "chunk-2",
                "chunk_index": 0,
                "title": "Aspirin review",
                "page_content": "aspirin review overview",
                "abstract": "review abstract",
                "source": "pubmed_abstract",
                "source_family": "pubmed",
                "year": 2023,
                "article_type": "review",
                "gov_agencies": ["FDA"],
                "vector": [0.11, 0.19, 0.31, 0.39],
            },
            {
                "point_id": "4",
                "doc_id": "set-1",
                "pmcid": "dailymed_set-1",
                "chunk_id": "dm-1",
                "set_id": "set-1",
                "drug_name": "Aspirin",
                "page_content": "dailymed aspirin contraindications",
                "source": "dailymed",
                "source_family": "dailymed",
                "year": 2022,
                "article_type": "drug_label",
                "vector": [0.9, 0.8, 0.7, 0.6],
            },
        ]
        table = db.create_table("medical_docs", data=rows, mode="overwrite")
        table.create_fts_index("page_content", replace=True)
        table.wait_for_index([idx.name for idx in table.list_indices()])

        retriever = LanceDBRetriever.__new__(LanceDBRetriever)
        retriever.n_retrieval = 10
        retriever.n_keyword_search = 5
        retriever.score_threshold = 0.0
        retriever.table = table
        retriever._tmp = tmp
        retriever._embed_query = lambda query, use_instruction=True: [0.1, 0.2, 0.3, 0.4]
        return retriever

    def test_retrieve_passages_contract_keys(self):
        retriever = self._make_retriever()
        passages = retriever.retrieve_passages("aspirin", use_hybrid=True)
        self.assertGreater(len(passages), 0)
        sample = passages[0]
        required_keys = {"corpus_id", "title", "text", "score", "stype", "article_type", "source"}
        self.assertTrue(required_keys.issubset(set(sample.keys())))
        self.assertNotEqual(sample.get("source"), "dailymed")

    def test_source_family_filter_semantics(self):
        retriever = self._make_retriever()
        passages = retriever.retrieve_passages("aspirin", source_family="pubmed")
        self.assertTrue(passages)
        for passage in passages:
            self.assertEqual(passage.get("source"), "pubmed_abstract")

    def test_gov_agency_filter_semantics(self):
        retriever = self._make_retriever()
        passages = retriever.retrieve_passages("aspirin", gov_agency="NIH")
        self.assertTrue(passages)
        for passage in passages:
            agencies = passage.get("gov_agencies") or []
            self.assertIn("NIH", agencies)

    def test_get_all_chunks_for_doc_filter_path(self):
        retriever = self._make_retriever()
        chunks = retriever.get_all_chunks_for_doc("doc-1")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].get("chunk_index"), 0)
        self.assertEqual(chunks[1].get("chunk_index"), 1)

    def test_dailymed_search_returns_dailymed_payload(self):
        retriever = self._make_retriever()
        results = retriever.search_dailymed_by_drug(["aspirin"], limit=3)
        self.assertTrue(results)
        self.assertEqual(results[0].get("source"), "dailymed")
        self.assertTrue(results[0].get("set_id") or results[0].get("corpus_id"))

    def test_dense_only_uses_similarity_ordering(self):
        tmp = tempfile.TemporaryDirectory(prefix="lancedb-retriever-dense-test-")
        db = lancedb.connect(tmp.name)
        rows = [
            {
                "point_id": "near",
                "doc_id": "near",
                "pmcid": "PMC_NEAR",
                "chunk_id": "near-1",
                "title": "near",
                "page_content": "near vector",
                "source": "pmc_oa",
                "source_family": "pmc",
                "year": 2024,
                "article_type": "trial",
                "vector": [0.1, 0.2, 0.3, 0.4],
            },
            {
                "point_id": "far",
                "doc_id": "far",
                "pmcid": "PMC_FAR",
                "chunk_id": "far-1",
                "title": "far",
                "page_content": "far vector",
                "source": "pmc_oa",
                "source_family": "pmc",
                "year": 2024,
                "article_type": "trial",
                "vector": [0.9, 0.9, 0.9, 0.9],
            },
        ]
        table = db.create_table("medical_docs", data=rows, mode="overwrite")
        retriever = LanceDBRetriever.__new__(LanceDBRetriever)
        retriever.n_retrieval = 10
        retriever.n_keyword_search = 5
        retriever.score_threshold = 0.0
        retriever.table = table
        retriever._tmp = tmp
        retriever._embed_query = lambda query, use_instruction=True: [0.1, 0.2, 0.3, 0.4]

        passages = retriever.retrieve_passages("vector", use_hybrid=False)
        self.assertGreaterEqual(len(passages), 2)
        self.assertEqual(passages[0].get("pmcid"), "PMC_NEAR")
        self.assertGreater(float(passages[0].get("score", 0.0)), float(passages[1].get("score", 0.0)))


if __name__ == "__main__":
    unittest.main()
