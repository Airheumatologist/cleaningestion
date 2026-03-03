import unittest
from datetime import datetime

import pandas as pd

from src.rag_pipeline import MedicalRAGPipeline


class DummyRetriever:
    def __init__(self, full_text_docs=None):
        self.full_text_docs = full_text_docs or {}

    def get_all_chunks_for_doc(self, doc_id):
        text = self.full_text_docs.get(doc_id, "")
        if not text:
            return []
        return [{"full_section_text": text}]


def make_pipeline_for_test() -> MedicalRAGPipeline:
    pipeline = MedicalRAGPipeline.__new__(MedicalRAGPipeline)
    pipeline.final_recency_policy_mode = "hybrid"
    pipeline.final_recency_window_years = 5
    pipeline.final_recency_backfill_max_evidence_level = 2
    pipeline.final_recency_exclude_unknown_non_dailymed = True
    pipeline.final_top_articles = 4
    pipeline.pmc_fulltext_recent_only = True
    pipeline.retriever = DummyRetriever()
    pipeline._last_recency_stats = {}
    pipeline._last_context_stats = {}
    return pipeline


class RecencySelectionTests(unittest.TestCase):
    def test_hybrid_recency_selection_with_backfill_and_dailymed_exemption(self):
        pipeline = make_pipeline_for_test()
        current_year = datetime.now().year

        papers_df = pd.DataFrame(
            [
                {
                    "title": "Recent Journal Paper",
                    "year": current_year,
                    "article_type": "review",
                    "relevance_score": 0.90,
                    "evidence_level": 2,
                },
                {
                    "title": "Unknown-Year Non-DailyMed",
                    "year": None,
                    "article_type": "review",
                    "relevance_score": 0.85,
                    "evidence_level": 1,
                },
                {
                    "title": "DailyMed Label",
                    "year": None,
                    "article_type": "drug_label",
                    "source": "dailymed",
                    "pmcid": "dailymed_abc",
                    "relevance_score": 0.80,
                    "evidence_level": None,
                },
                {
                    "title": "Older High-Evidence Backfill",
                    "year": current_year - 8,
                    "article_type": "guideline",
                    "relevance_score": 0.70,
                    "evidence_level": 1,
                },
                {
                    "title": "Older Low-Evidence Excluded",
                    "year": current_year - 8,
                    "article_type": "review",
                    "relevance_score": 0.60,
                    "evidence_level": 3,
                },
            ]
        )

        selected_df, stats = pipeline._apply_final_recency_policy(papers_df)
        selected_titles = selected_df["title"].tolist()

        self.assertEqual(selected_titles, ["Recent Journal Paper", "DailyMed Label", "Older High-Evidence Backfill"])
        self.assertEqual(stats["recent_kept_non_dailymed"], 1)
        self.assertEqual(stats["dailymed_kept"], 1)
        self.assertEqual(stats["older_backfilled"], 1)
        self.assertEqual(stats["unknown_non_dailymed_excluded"], 1)

    def test_unknown_year_non_dailymed_excluded_but_dailymed_retained(self):
        pipeline = make_pipeline_for_test()
        papers_df = pd.DataFrame(
            [
                {
                    "title": "Unknown Non-DailyMed",
                    "year": None,
                    "article_type": "review",
                    "relevance_score": 0.90,
                },
                {
                    "title": "Unknown DailyMed",
                    "year": None,
                    "article_type": "drug_label",
                    "source": "dailymed",
                    "pmcid": "dailymed_xyz",
                    "relevance_score": 0.80,
                },
            ]
        )

        selected_df, stats = pipeline._apply_final_recency_policy(papers_df)
        self.assertEqual(selected_df["title"].tolist(), ["Unknown DailyMed"])
        self.assertEqual(stats["unknown_non_dailymed_excluded"], 1)


class PmcFullTextRecencyTests(unittest.TestCase):
    def test_recent_pmc_full_text_caps_at_two(self):
        pipeline = make_pipeline_for_test()
        current_year = datetime.now().year
        pipeline.retriever = DummyRetriever(
            {
                "PMC_A": "full text A",
                "PMC_B": "full text B",
                "PMC_C": "full text C",
            }
        )

        papers_df = pd.DataFrame(
            [
                {"title": "A", "pmcid": "PMC_A", "corpus_id": "PMC_A", "doc_id": "PMC_A", "year": current_year, "article_type": "other", "abstract": "abs A"},
                {"title": "B", "pmcid": "PMC_B", "corpus_id": "PMC_B", "doc_id": "PMC_B", "year": current_year - 1, "article_type": "other", "abstract": "abs B"},
                {"title": "C", "pmcid": "PMC_C", "corpus_id": "PMC_C", "doc_id": "PMC_C", "year": current_year - 2, "article_type": "other", "abstract": "abs C"},
            ]
        )

        pipeline._get_papers_for_context(papers_df, query="small cell lung cancer")
        self.assertEqual(pipeline._last_context_stats["pmc_recent_fulltext_used"], 2)

    def test_single_recent_pmc_full_text_uses_one(self):
        pipeline = make_pipeline_for_test()
        current_year = datetime.now().year
        pipeline.retriever = DummyRetriever(
            {
                "PMC_RECENT": "recent full text",
                "PMC_OLD1": "old full text 1",
                "PMC_OLD2": "old full text 2",
            }
        )

        papers_df = pd.DataFrame(
            [
                {"title": "Recent", "pmcid": "PMC_RECENT", "corpus_id": "PMC_RECENT", "doc_id": "PMC_RECENT", "year": current_year, "article_type": "other", "abstract": "abs recent"},
                {"title": "Old1", "pmcid": "PMC_OLD1", "corpus_id": "PMC_OLD1", "doc_id": "PMC_OLD1", "year": current_year - 8, "article_type": "other", "abstract": "abs old1"},
                {"title": "Old2", "pmcid": "PMC_OLD2", "corpus_id": "PMC_OLD2", "doc_id": "PMC_OLD2", "year": current_year - 9, "article_type": "other", "abstract": "abs old2"},
            ]
        )

        pipeline._get_papers_for_context(papers_df, query="small cell lung cancer")
        self.assertEqual(pipeline._last_context_stats["pmc_recent_fulltext_used"], 1)

    def test_no_recent_pmc_full_text_uses_zero(self):
        pipeline = make_pipeline_for_test()
        current_year = datetime.now().year
        pipeline.retriever = DummyRetriever(
            {
                "PMC_OLD1": "old full text 1",
                "PMC_OLD2": "old full text 2",
                "PMC_OLD3": "old full text 3",
            }
        )

        papers_df = pd.DataFrame(
            [
                {"title": "Old1", "pmcid": "PMC_OLD1", "corpus_id": "PMC_OLD1", "doc_id": "PMC_OLD1", "year": current_year - 10, "article_type": "other", "abstract": "abs old1"},
                {"title": "Old2", "pmcid": "PMC_OLD2", "corpus_id": "PMC_OLD2", "doc_id": "PMC_OLD2", "year": current_year - 11, "article_type": "other", "abstract": "abs old2"},
                {"title": "Old3", "pmcid": "PMC_OLD3", "corpus_id": "PMC_OLD3", "doc_id": "PMC_OLD3", "year": current_year - 12, "article_type": "other", "abstract": "abs old3"},
            ]
        )

        pipeline._get_papers_for_context(papers_df, query="small cell lung cancer")
        self.assertEqual(pipeline._last_context_stats["pmc_recent_fulltext_used"], 0)


if __name__ == "__main__":
    unittest.main()
