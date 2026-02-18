import src.retriever_qdrant as retriever_module
from src.retriever_qdrant import QdrantRetriever


class _DummyPoint:
    def __init__(self, payload, score=0.9, point_id="p1"):
        self.payload = payload
        self.score = score
        self.id = point_id


class _DummyQueryResult:
    def __init__(self, points):
        self.points = points


class _DummyClient:
    def __init__(self, points):
        self._points = points

    def query_points(self, **kwargs):
        return _DummyQueryResult(self._points)


def test_normalize_publication_types_handles_missing_and_mixed_formats():
    retriever = QdrantRetriever.__new__(QdrantRetriever)

    assert retriever._normalize_publication_type_list(None) == []
    assert retriever._normalize_publication_type_list(" Practice Guideline ") == ["Practice Guideline"]
    assert retriever._normalize_publication_type_list(
        [{"type": "Guideline"}, {"name": "Consensus Statement"}, "Guideline"]
    ) == ["Guideline", "Consensus Statement"]


def test_dense_search_maps_publication_type_for_standard_payload(monkeypatch):
    monkeypatch.setattr(retriever_module, "USE_HYBRID_SEARCH", False)

    payload = {
        "pmcid": "PMC123",
        "pmid": "123",
        "title": "Sample guideline paper",
        "page_content": "Text",
        "abstract": "Abstract",
        "journal": "Journal",
        "article_type": "research_article",
        "publication_type": "Practice Guideline",
        "authors": ["Author A"],
    }

    retriever = QdrantRetriever.__new__(QdrantRetriever)
    retriever.collection_name = "test"
    retriever.n_retrieval = 5
    retriever.score_threshold = 0.0
    retriever.client = _DummyClient([_DummyPoint(payload)])
    retriever._build_filter = lambda **kwargs: None
    retriever._embed_query = lambda query: [0.1, 0.2]
    retriever.bm25_sparse_encoder = None

    passages = retriever.retrieve_passages("guideline query")

    assert len(passages) == 1
    assert passages[0]["publication_type"] == ["Practice Guideline"]


def test_transform_dailymed_payload_includes_publication_type():
    retriever = QdrantRetriever.__new__(QdrantRetriever)
    payload = {
        "set_id": "abc",
        "drug_name": "ExampleDrug",
        "manufacturer": "Example Pharma",
        "publication_type": ["Drug Label"],
    }

    transformed = retriever._transform_dailymed_payload(payload, score=0.8)

    assert transformed["publication_type"] == ["Drug Label"]
