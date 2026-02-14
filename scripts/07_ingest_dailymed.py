#!/usr/bin/env python3
"""Ingest DailyMed XML files into self-hosted Qdrant."""

from __future__ import annotations

import argparse
import logging
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Document, PointStruct

from config_ingestion import IngestionConfig, ensure_data_dirs
from src.bm25_sparse import BM25SparseEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "dailymed_ingested_ids.txt"
NS = {"hl7": "urn:hl7-org:v3"}

SECTION_CODES = {
    "34067-9": "indications",
    "34070-3": "contraindications",
    "34068-7": "dosage",
    "34084-4": "adverse_reactions",
    "34073-7": "interactions",
    "34071-1": "warnings",
    "43685-7": "warnings_precautions",
    "34089-3": "description",
}


class EmbeddingProvider:
    def __init__(self) -> None:
        self.provider = IngestionConfig.EMBEDDING_PROVIDER.lower().strip()
        self.model = IngestionConfig.EMBEDDING_MODEL
        self.local_encoder = None

        if self.provider == "local":
            from sentence_transformers import SentenceTransformer

            logger.info("Loading local embedding model: %s", self.model)
            self.local_encoder = SentenceTransformer(self.model)

    def use_cloud(self) -> bool:
        return self.provider == "qdrant_cloud_inference"

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.local_encoder is None:
            raise RuntimeError("Local encoder not initialized")
        vectors = self.local_encoder.encode(texts, normalize_embeddings=True, batch_size=IngestionConfig.EMBEDDING_BATCH_SIZE)
        return [v.tolist() for v in vectors]



def get_text(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""
    return " ".join(element.itertext()).strip()



def parse_spl_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        set_id_elem = root.find(".//hl7:setId", NS)
        set_id = set_id_elem.get("root", "") if set_id_elem is not None else ""
        if not set_id:
            set_id = xml_path.stem

        title = get_text(root.find(".//hl7:title", NS))

        drug_name = title
        name_elem = root.find(".//hl7:manufacturedProduct/hl7:manufacturedProduct/hl7:name", NS)
        if name_elem is not None and name_elem.text:
            drug_name = name_elem.text.strip()

        active_ingredients: List[str] = []
        for ingredient in root.findall('.//hl7:ingredient[@classCode="ACTIB"]', NS):
            ing_name = ingredient.find('.//hl7:ingredientSubstance/hl7:name', NS)
            if ing_name is not None and ing_name.text:
                active_ingredients.append(ing_name.text.strip())

        sections: Dict[str, str] = {}
        for section in root.findall(".//hl7:section", NS):
            code_elem = section.find("hl7:code", NS)
            if code_elem is None:
                continue
            code = code_elem.get("code", "")
            if code not in SECTION_CODES:
                continue
            text_elem = section.find("hl7:text", NS)
            sections[SECTION_CODES[code]] = get_text(text_elem)[:5000]

        manufacturer = ""
        org_name = root.find(".//hl7:representedOrganization/hl7:name", NS)
        if org_name is not None and org_name.text:
            manufacturer = org_name.text.strip()

        return {
            "set_id": set_id,
            "drug_name": drug_name,
            "title": title,
            "active_ingredients": active_ingredients,
            "manufacturer": manufacturer,
            "indications": sections.get("indications", ""),
            "contraindications": sections.get("contraindications", ""),
            "dosage": sections.get("dosage", ""),
            "adverse_reactions": sections.get("adverse_reactions", ""),
            "interactions": sections.get("interactions", ""),
            "warnings": sections.get("warnings", "") or sections.get("warnings_precautions", ""),
            "description": sections.get("description", ""),
            "source": "dailymed",
            "article_type": "drug_label",
        }
    except Exception:
        return None



def create_embedding_text(drug: Dict[str, Any]) -> str:
    parts = [
        drug.get("drug_name", ""),
        f"Active ingredients: {', '.join(drug.get('active_ingredients', []))}",
        drug.get("indications", ""),
        drug.get("contraindications", ""),
        drug.get("warnings", ""),
        drug.get("dosage", "")[:800],
    ]
    return ". ".join(p for p in parts if p)[:2000]



def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return {line.strip() for line in CHECKPOINT_FILE.read_text().splitlines() if line.strip()}
    return set()



def append_checkpoint(ids: Iterable[str]) -> None:
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
        for value in ids:
            f.write(f"{value}\n")



def build_points(batch: List[Dict[str, Any]], embedding_provider: EmbeddingProvider) -> tuple[List[PointStruct], List[str]]:
    set_ids: List[str] = []
    texts: List[str] = []
    sparse_vectors = []
    payloads: List[Dict[str, Any]] = []
    sparse_encoder = None
    if IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25":
        sparse_encoder = BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
            min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )

    for drug in batch:
        set_id = (drug.get("set_id") or "").strip()
        if not set_id:
            continue
        text = create_embedding_text(drug)
        if len(text) < 20:
            continue

        set_ids.append(set_id)
        texts.append(text)
        sparse_vectors.append(sparse_encoder.encode_document(text) if sparse_encoder is not None else None)
        payloads.append(
            {
                "set_id": set_id,
                "drug_name": (drug.get("drug_name") or "")[:200],
                "title": (drug.get("title") or "")[:300],
                "active_ingredients": (drug.get("active_ingredients") or [])[:10],
                "manufacturer": drug.get("manufacturer", ""),
                "indications": (drug.get("indications") or "")[:5000],
                "contraindications": (drug.get("contraindications") or "")[:3000],
                "warnings": (drug.get("warnings") or "")[:3000],
                "adverse_reactions": (drug.get("adverse_reactions") or "")[:3000],
                "dosage": (drug.get("dosage") or "")[:5000],
                "source": "dailymed",
                "article_type": "drug_label",
            }
        )

    points: List[PointStruct] = []
    if not texts:
        return points, []

    if embedding_provider.use_cloud():
        for set_id, text, sparse_vector, payload in zip(set_ids, texts, sparse_vectors, payloads):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"dailymed:{set_id}"))
            vector_data: Any = Document(text=text, model=embedding_provider.model)
            if sparse_vector is not None:
                vector_data = {"": vector_data, "sparse": sparse_vector}
            points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))
    else:
        vectors = embedding_provider.embed_batch(texts)
        for set_id, vector, sparse_vector, payload in zip(set_ids, vectors, sparse_vectors, payloads):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"dailymed:{set_id}"))
            vector_data: Any = vector
            if sparse_vector is not None:
                vector_data = {"": vector, "sparse": sparse_vector}
            points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))

    return points, set_ids



def upsert_with_retry(client: QdrantClient, points: List[PointStruct]) -> None:
    for attempt in range(IngestionConfig.MAX_RETRIES):
        try:
            client.upsert(collection_name=IngestionConfig.COLLECTION_NAME, points=points, wait=False)
            return
        except Exception as exc:
            if attempt == IngestionConfig.MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)
            logger.warning("Upsert retry after error: %s", str(exc)[:200])



def run_ingestion(xml_dir: Path, embedding_provider: EmbeddingProvider) -> None:
    ensure_data_dirs()
    xml_files = sorted(xml_dir.rglob("*.xml"))

    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
        cloud_inference=embedding_provider.use_cloud(),
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    info = client.get_collection(IngestionConfig.COLLECTION_NAME)
    logger.info("Connected to %s points=%s", IngestionConfig.COLLECTION_NAME, info.points_count)

    seen = load_checkpoint()
    logger.info("DailyMed checkpoint size=%s", len(seen))

    batch: List[Dict[str, Any]] = []
    inserted = 0
    skipped = 0
    started = time.time()

    for xml_file in xml_files:
        drug = parse_spl_xml(xml_file)
        if not drug:
            continue

        set_id = (drug.get("set_id") or "").strip()
        if not set_id or set_id in seen:
            skipped += 1
            continue

        batch.append(drug)
        if len(batch) < IngestionConfig.BATCH_SIZE:
            continue

        points, set_ids = build_points(batch, embedding_provider)
        if points:
            upsert_with_retry(client, points)
            append_checkpoint(set_ids)
            seen.update(set_ids)
            inserted += len(points)
        batch.clear()

    if batch:
        points, set_ids = build_points(batch, embedding_provider)
        if points:
            upsert_with_retry(client, points)
            append_checkpoint(set_ids)
            inserted += len(points)

    logger.info(
        "DailyMed ingestion complete inserted=%s skipped=%s elapsed=%.1fs",
        inserted,
        skipped,
        time.time() - started,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest DailyMed into self-hosted Qdrant")
    parser.add_argument("--xml-dir", type=Path, default=IngestionConfig.DAILYMED_XML_DIR)
    args = parser.parse_args()

    if not args.xml_dir.exists():
        raise SystemExit(f"Directory not found: {args.xml_dir}")

    provider = EmbeddingProvider()
    run_ingestion(args.xml_dir, provider)
