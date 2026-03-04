"""
Qdrant Cloud Client for Medical RAG Pipeline.

Handles connection to Qdrant Cloud, collection management,
and vector operations with scalar quantization.
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    PointStruct,
    Distance,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    Filter,
    FieldCondition,
    MatchValue,
)

from .config import (
    QDRANT_URL, 
    QDRANT_API_KEY, 
    QDRANT_PREFER_GRPC,
    QDRANT_GRPC_PORT,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    TOP_K_RESULTS,
    SCORE_THRESHOLD,
    BULK_RETRIEVAL_LIMIT
)


class MedicalQdrantClient:
    """
    Qdrant Cloud client for medical document storage and retrieval.
    
    Features:
    - Scalar quantization for memory reduction
    - 768-dimensional ClinicalBERT vector support
    - Medical metadata payload storage
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern for connection reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize Qdrant Cloud connection."""
        if self._initialized:
            return
        
        print(f"🔗 Connecting to Qdrant Cloud...")
        
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=QDRANT_PREFER_GRPC,
            grpc_port=QDRANT_GRPC_PORT,
        )
        self.collection_name = COLLECTION_NAME
        self._initialized = True
        
        print("✅ Qdrant connection established!")
    
    def test_connection(self) -> bool:
        """Test the Qdrant Cloud connection."""
        try:
            collections = self.client.get_collections()
            print(f"✅ Connection successful! Found {len(collections.collections)} collections.")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def collection_exists(self) -> bool:
        """Check if the medical collection exists."""
        try:
            collections = self.client.get_collections()
            return any(c.name == self.collection_name for c in collections.collections)
        except Exception:
            return False
    
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create the medical documents collection with scalar quantization.
        
        Args:
            recreate: If True, delete existing collection first
            
        Returns:
            True if collection was created successfully
        """
        # Check if collection exists
        if self.collection_exists():
            if recreate:
                print(f"🗑️  Deleting existing collection '{self.collection_name}'...")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"ℹ️  Collection '{self.collection_name}' already exists.")
                return True
        
        print(f"📦 Creating collection '{self.collection_name}'...")
        print(f"   - Dimension: {EMBEDDING_DIMENSION} (ClinicalBERT)")
        print(f"   - Distance: Cosine")
        print(f"   - Quantization: Scalar (int8, 75% memory reduction)")
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
            on_disk_payload=True,  # Store payloads on disk to save RAM
        )
        
        print(f"✅ Collection '{self.collection_name}' created successfully!")
        return True
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self.collection_exists():
            return {"error": "Collection does not exist"}
        
        info = self.client.get_collection(self.collection_name)
        # Handle both old and new Qdrant API
        vectors_count = getattr(info, 'vectors_count', None) or getattr(info, 'points_count', 0)
        return {
            "name": self.collection_name,
            "vectors_count": vectors_count,
            "points_count": info.points_count,
            "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
        }
    
    def upsert_articles(
        self, 
        articles: List[Dict[str, Any]], 
        embeddings: List[List[float]]
    ) -> int:
        """
        Upsert medical articles with their embeddings to Qdrant.
        
        Args:
            articles: List of article dictionaries with metadata
            embeddings: List of 768-d embeddings (same order as articles)
            
        Returns:
            Number of points upserted
        """
        if len(articles) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(articles)} articles but {len(embeddings)} embeddings"
            )
        
        points = []
        for idx, (article, embedding) in enumerate(zip(articles, embeddings)):
            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "pmcid": article.get("pmcid", f"SAMPLE-{idx}"),
                    "pmid": article.get("pmid"),
                    "title": article.get("title", "")[:200],
                    "abstract": article.get("abstract", "")[:500],
                    "journal": article.get("journal", "Unknown"),
                    "year": article.get("year"),
                    "article_type": article.get("article_type", "research"),
                    "keywords": article.get("keywords", [])[:5],
                }
            )
            points.append(point)
        
        print(f"📤 Upserting {len(points)} vectors to Qdrant...")
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        
        print(f"✅ Successfully upserted {len(points)} vectors!")
        return len(points)
    
    def search(
        self, 
        query_embedding: List[float],
        top_k: int = TOP_K_RESULTS,
        score_threshold: float = SCORE_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Search for similar medical documents.
        
        Args:
            query_embedding: 768-d query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching documents with scores
        """
        # Use query_points which is the current Qdrant API
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        
        documents = []
        for hit in results.points:
            documents.append({
                "id": hit.id,
                "score": hit.score,
                "pmcid": hit.payload.get("pmcid"),
                "pmid": hit.payload.get("pmid"),
                "title": hit.payload.get("title"),
                "abstract": hit.payload.get("abstract"),
                "journal": hit.payload.get("journal"),
                "year": hit.payload.get("year"),
                "article_type": hit.payload.get("article_type"),
                "keywords": hit.payload.get("keywords", []),
            })
        
        return documents
    
    def search_bulk(
        self,
        query_embeddings: List[List[float]],
        top_k_per_query: int = 50,
        total_limit: int = 200,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search with multiple query embeddings and deduplicate results.
        
        Args:
            query_embeddings: List of query vectors (from expanded queries)
            top_k_per_query: Candidates to retrieve per query
            total_limit: Maximum total unique articles to return
            score_threshold: Minimum similarity score (uses default if None)
            
        Returns:
            List of unique articles up to total_limit
        """
        score_threshold = score_threshold or SCORE_THRESHOLD
        
        # Track seen articles by pmcid to deduplicate
        seen_pmcids = set()
        all_articles = []
        
        print(f"🔎 Bulk search with {len(query_embeddings)} queries, "
              f"{top_k_per_query} per query...")
        
        for i, embedding in enumerate(query_embeddings):
            try:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=embedding,
                    limit=top_k_per_query,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
                
                for hit in results.points:
                    pmcid = hit.payload.get("pmcid")
                    
                    # Skip duplicates
                    if pmcid in seen_pmcids:
                        continue
                    
                    seen_pmcids.add(pmcid)
                    all_articles.append({
                        "id": hit.id,
                        "score": hit.score,
                        "pmcid": pmcid,
                        "pmid": hit.payload.get("pmid"),
                        "title": hit.payload.get("title"),
                        "abstract": hit.payload.get("abstract"),
                        "full_text": hit.payload.get("full_text", ""),
                        "journal": hit.payload.get("journal"),
                        "year": hit.payload.get("year"),
                        "article_type": hit.payload.get("article_type"),
                        "keywords": hit.payload.get("keywords", []),
                    })
                    
                    # Stop if we have enough
                    if len(all_articles) >= total_limit:
                        break
                        
            except Exception as e:
                print(f"⚠️ Query {i+1} failed: {e}")
                continue
            
            if len(all_articles) >= total_limit:
                break
        
        print(f"✅ Retrieved {len(all_articles)} unique articles")
        return all_articles

    def get_all_chunks_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunk payloads for a given doc_id using scroll pagination.
        """
        normalized_doc_id = str(doc_id or "").strip()
        if not normalized_doc_id:
            return []

        def _as_int(value: Any, default: int = 10**9) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _scroll_all(field_name: str, value: str) -> List[Dict[str, Any]]:
            query_filter = Filter(
                must=[FieldCondition(key=field_name, match=MatchValue(value=value))]
            )
            payloads: List[Dict[str, Any]] = []
            next_offset = None

            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    offset=next_offset,
                    limit=256,
                    with_payload=True,
                )

                if not results:
                    break

                for point in results:
                    payload = point.payload or {}
                    if payload:
                        payloads.append(payload)

                if next_offset is None:
                    break

            return payloads

        all_chunks = _scroll_all("doc_id", normalized_doc_id)
        if not all_chunks and normalized_doc_id.upper().startswith("PMC"):
            all_chunks = _scroll_all("pmcid", normalized_doc_id.upper())

        all_chunks.sort(
            key=lambda payload: (
                _as_int(payload.get("chunk_index")),
                _as_int(
                    payload.get("char_offset")
                    or payload.get("char_start_offset")
                    or payload.get("start_offset")
                    or payload.get("offset")
                ),
                str(payload.get("section_id") or ""),
                str(payload.get("chunk_id") or ""),
            )
        )
        return all_chunks


if __name__ == "__main__":
    # Test Qdrant connection
    print("🧪 Testing Qdrant Client")
    print("=" * 50)
    
    client = MedicalQdrantClient()
    
    # Test connection
    client.test_connection()
    
    # Get collection info if it exists
    if client.collection_exists():
        info = client.get_collection_info()
        print(f"\n📊 Collection Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
