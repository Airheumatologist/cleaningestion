"""
ScholarQA-Style Reranker with Paper Aggregation.

Implements ai2-scholarqa-lib's exact reranking methodology:
- Passage-level reranking using Cohere (substitute for CrossEncoder)
- Paper aggregation using max rerank_score
- DataFrame output with ScholarQA reference_string format

No custom criteria (evidence hierarchy, article count limits, etc.)
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import cohere
from anyascii import anyascii
import yaml

from .config import COHERE_API_KEY

# Try to import medical entity expander for entity matching
try:
    from .medical_entity_expander import MedicalEntityExpander
    ENTITY_EXPANDER_AVAILABLE = True
except ImportError:
    ENTITY_EXPANDER_AVAILABLE = False
    MedicalEntityExpander = None

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions (from ScholarQA)
# =============================================================================

def make_int(value) -> int:
    """Convert value to int, returning 0 if not possible."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def get_ref_author_str(authors) -> str:
    """
    Format authors for reference string (ScholarQA format).
    
    Returns: "LastName, LastName2 et al." or "LastName"
    """
    if not authors:
        return "Unknown"
    
    if isinstance(authors, str):
        # Simple string format
        names = [a.strip() for a in authors.split(",")]
        if len(names) >= 2:
            return f"{names[0].split()[-1]}, {names[1].split()[-1]} et al."
        return names[0].split()[-1] if names else "Unknown"
    
    if isinstance(authors, list):
        author_names = []
        for a in authors[:2]:  # First 2 authors
            if isinstance(a, dict):
                name = a.get("name", "Unknown")
            else:
                name = str(a)
            # Get last name
            author_names.append(name.split()[-1] if name else "Unknown")
        
        if len(author_names) >= 2:
            return f"{author_names[0]}, {author_names[1]} et al."
        elif author_names:
            return author_names[0]
    
    return "Unknown"


# =============================================================================
# Abstract Reranker Interface (matches ScholarQA)
# =============================================================================

# Evidence hierarchy boost multipliers
EVIDENCE_HIERARCHY = {
    # Highest evidence level
    "systematic_review": 1.25,
    "meta_analysis": 1.25,
    "guideline": 1.30,
    # Strong evidence - official drug labeling
    "drug_label": 1.25,  # FDA-approved drug labels are authoritative
    "clinical_trial": 1.20,
    "randomized_controlled_trial": 1.20,
    # Good evidence
    "review_article": 1.10,
    "cohort_study": 1.05,
    # Standard evidence
    "research_article": 1.00,
    "original_research": 1.00,
    "research": 1.00,
    "other": 1.00,
    # Lower generalizability
    "case_report": 0.90,
    "case_series": 0.92,
    "letter": 0.85,
    "editorial": 0.85,
    "comment": 0.80,
}

# High-impact medical journals and authoritative sources
HIGH_IMPACT_JOURNALS = {
    "new england journal of medicine", "nejm",
    "the lancet", "lancet",
    "jama", "journal of the american medical association",
    "bmj", "british medical journal",
    "nature medicine",
    "annals of internal medicine",
    "circulation",
    "blood",
    "jama internal medicine",
    "the lancet oncology",
    "lancet oncology",
    "journal of clinical oncology",
    "european heart journal",
    "gastroenterology",
    "hepatology",
    "diabetes care",
    "chest",
    # Official drug information sources
    "dailymed",  # FDA drug labels
    "fda drug label",
}


def apply_evidence_boosts(
    documents: list,
    current_year: int = 2025
) -> list:
    """
    Apply evidence hierarchy boosts to reranked documents.
    
    Boosts based on:
    1. Article type (evidence hierarchy)
    2. Recency (newer papers slightly preferred)
    3. Journal prestige (high-impact journals)
    
    Adds 'boosted_score' field to each document.
    """
    for doc in documents:
        base_score = doc.get("rerank_score", doc.get("score", 0.5))
        multiplier = 1.0
        
        # 1. Article type boost
        article_type = doc.get("article_type", "other").lower().replace("-", "_")
        type_boost = EVIDENCE_HIERARCHY.get(article_type, 1.0)
        multiplier *= type_boost
        
        # 2. Recency boost
        year = doc.get("year")
        if year:
            try:
                year = int(year)
                if year >= current_year - 1:  # 2024-2025
                    multiplier *= 1.05
                elif year >= current_year - 3:  # 2022-2023
                    multiplier *= 1.02
            except (ValueError, TypeError):
                pass
        
        # 3. High-impact journal boost
        journal = doc.get("journal", doc.get("venue", "")).lower()
        if any(j in journal for j in HIGH_IMPACT_JOURNALS):
            multiplier *= 1.08
        
        # Calculate boosted score
        doc["boosted_score"] = base_score * multiplier
        doc["boost_multiplier"] = multiplier
    
    return documents


class AbstractReranker:
    """Abstract base class for rerankers."""
    
    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        raise NotImplementedError


# =============================================================================
# Cohere Reranker (substitute for CrossEncoder)
# =============================================================================

class CohereReranker(AbstractReranker):
    """
    Cohere-based reranker as substitute for ScholarQA's CrossEncoder.
    
    Uses Cohere rerank-v4.0-pro (upgraded from v3.5) for better healthcare performance.
    Formats documents as YAML for optimal reranking performance.
    """
    
    def __init__(self, model: str = "rerank-v4.0-pro"):
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not set")
        
        self.client = cohere.ClientV2(api_key=COHERE_API_KEY)
        self.model = model
        logger.info(f"✅ Cohere Reranker initialized with {model}")
    
    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Get relevance scores for documents.
        
        Formats documents as YAML for optimal performance (Cohere best practice).
        
        Args:
            query: Search query
            documents: List of document strings (can be plain text or YAML)
            
        Returns:
            List of relevance scores
        """
        if not documents:
            return []
        
        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=len(documents),  # Return all scores (Cohere best practice)
            )
            
            # Create score array indexed by original position
            scores = [0.0] * len(documents)
            for result in response.results:
                scores[result.index] = result.relevance_score
            
            return scores
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            # Return default scores
            return [0.5] * len(documents)
    
    def format_document_as_yaml(self, title: str, content: str, abstract: str = "") -> str:
        """
        Format document as YAML string for optimal Cohere reranking.
        
        Args:
            title: Document title
            content: Document content (abstract or full text)
            abstract: Optional abstract (if content is full text)
            
        Returns:
            YAML-formatted string
        """
        import yaml
        
        doc_dict = {
            "Title": title,
        }
        
        if abstract:
            doc_dict["Abstract"] = abstract
            doc_dict["Content"] = content
        else:
            doc_dict["Content"] = content
        
        return yaml.dump(doc_dict, sort_keys=False, allow_unicode=True)


# =============================================================================
# Paper Finder with Reranker (matches ScholarQA's PaperFinderWithReranker)
# =============================================================================

class PaperFinderWithReranker:
    """
    ScholarQA-style paper finder with reranking.
    
    Flow:
    1. Rerank passages using Cohere
    2. Aggregate passages to paper level (max score)
    3. Format into DataFrame with reference strings
    """
    
    def __init__(
        self,
        reranker: AbstractReranker = None,
        n_rerank: int = -1,
        context_threshold: float = 0.0
    ):
        """
        Initialize paper finder.
        
        Args:
            reranker: Reranker engine (defaults to Cohere)
            n_rerank: Max passages to keep after rerank (-1 = all)
            context_threshold: Min score threshold
        """
        self.reranker_engine = reranker or CohereReranker()
        self.n_rerank = n_rerank
        self.context_threshold = context_threshold
        
        # Initialize medical entity expander for entity matching
        self.entity_expander = None
        if ENTITY_EXPANDER_AVAILABLE:
            try:
                self.entity_expander = MedicalEntityExpander()
                logger.info("✅ Medical entity expander initialized for reranking")
            except Exception as e:
                logger.warning(f"Medical entity expander not available for reranking: {e}")
    
    def rerank(
        self,
        query: str,
        retrieved_ctxs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank passages using Cohere + evidence hierarchy boosts.
        
        1. Cohere reranking for semantic relevance
        2. Apply evidence hierarchy boosts (article type, recency, journal)
        3. Sort by boosted_score
        """
        if not retrieved_ctxs:
            return []
        
        logger.info(f"Reranking {len(retrieved_ctxs)} passages...")
        
        # Format documents as YAML for optimal Cohere reranking (best practice)
        passages = []
        for doc in retrieved_ctxs:
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            text = doc.get("text", "")
            full_text = doc.get("full_text", "")
            
            # Use full text if available, otherwise use abstract/text
            content = full_text if full_text else (text if text else abstract)
            
            # Format as YAML for better performance
            if isinstance(self.reranker_engine, CohereReranker):
                yaml_doc = self.reranker_engine.format_document_as_yaml(
                    title=title,
                    content=content[:2000],  # Limit content length
                    abstract=abstract[:500] if abstract else ""
                )
                passages.append(yaml_doc)
            else:
                # Fallback to plain text
                passages.append((title + " " + content).strip()[:2000])
        
        # Get rerank scores from Cohere
        rerank_scores = self.reranker_engine.get_scores(query, passages)
        
        # Calculate entity matching scores
        entity_scores = self._calculate_entity_scores(query, retrieved_ctxs)
        
        # Combine Cohere rerank scores with entity matching scores
        # Weight: 70% rerank score, 30% entity score
        combined_scores = []
        for i, (rerank_score, entity_score) in enumerate(zip(rerank_scores, entity_scores)):
            combined_score = 0.7 * rerank_score + 0.3 * entity_score
            combined_scores.append(combined_score)
            retrieved_ctxs[i]["rerank_score"] = rerank_score
            retrieved_ctxs[i]["entity_score"] = entity_score
            retrieved_ctxs[i]["combined_score"] = combined_score
        
        logger.info(f"Entity scores: {[round(s, 3) for s in entity_scores[:5]]}")
        logger.info(f"Combined scores: {[round(s, 3) for s in combined_scores[:5]]}")
        
        # Apply evidence hierarchy boosts
        retrieved_ctxs = apply_evidence_boosts(retrieved_ctxs)
        
        # Log boost effects
        boosted = [(d.get("article_type", "?"), d.get("boost_multiplier", 1.0)) for d in retrieved_ctxs[:5]]
        logger.info(f"Evidence boosts applied: {boosted}")
        
        # Sort by boosted_score (includes evidence hierarchy and entity matching)
        # Prefer combined_score if available, otherwise use boosted_score
        sorted_ctxs = sorted(
            retrieved_ctxs,
            key=lambda x: x.get("boosted_score", x.get("combined_score", x.get("rerank_score", 0))),
            reverse=True
        )
        
        # Apply n_rerank limit if set
        if self.n_rerank > 0:
            sorted_ctxs = sorted_ctxs[:self.n_rerank]
        
        top_scores = [round(d.get("boosted_score", 0), 3) for d in sorted_ctxs[:5]]
        logger.info(f"Done reranking: {len(sorted_ctxs)} passages (top scores: {top_scores})")
        return sorted_ctxs
    
    def _calculate_entity_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate entity matching scores for documents.
        
        Extracts medical entities from query and scores documents based on entity overlap.
        
        Args:
            query: Search query
            documents: List of document dictionaries
            
        Returns:
            List of entity scores (0.0 to 1.0)
        """
        if not self.entity_expander or not documents:
            return [0.0] * len(documents)
        
        # Extract medical entities from query
        query_entities = self._extract_query_entities(query)
        
        if not query_entities:
            return [0.0] * len(documents)
        
        scores = []
        for doc in documents:
            title = str(doc.get("title", "")).lower()
            abstract = str(doc.get("abstract", "")).lower()
            text = str(doc.get("text", "")).lower()
            full_text = str(doc.get("full_text", "")).lower()
            
            # Combine text for searching
            doc_text = f"{title} {abstract} {text} {full_text}".lower()
            
            # Count entity matches
            matches = 0
            for entity in query_entities:
                entity_lower = entity.lower()
                # Check for entity in document (word boundary or exact match)
                if (entity_lower in doc_text or 
                    f" {entity_lower} " in doc_text or
                    entity_lower in title):
                    matches += 1
            
            # Score: proportion of query entities found in document
            score = matches / len(query_entities) if query_entities else 0.0
            scores.append(score)
        
        return scores
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract medical entities from query.
        
        Args:
            query: Search query
            
        Returns:
            List of medical entity terms
        """
        entities = []
        
        if not self.entity_expander:
            return entities
        
        # Extract acronyms and their expansions
        import re
        words = query.split()
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check if it's a known acronym
            if self.entity_expander._is_likely_acronym(clean_word):
                expansions = self.entity_expander.expand_acronym(clean_word)
                if expansions:
                    # Add both acronym and full term
                    entities.append(clean_word.upper())
                    entities.append(expansions[0])
        
        # Also look for common medical condition patterns
        medical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Syndrome\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Disease\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Disorder\b',
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in seen:
                seen.add(entity_lower)
                unique_entities.append(entity)
        
        return unique_entities
    
    def aggregate_snippets_to_papers(
        self,
        snippets_list: List[Dict[str, Any]],
        paper_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Aggregate passages to paper level (ScholarQA methodology).
        
        Uses max rerank_score for each paper.
        """
        logger.info(f"Aggregating {len(snippets_list)} passages at paper level")
        
        paper_snippets = {}
        
        for snippet in snippets_list:
            # Use pmcid as corpus_id for our Qdrant data
            corpus_id = snippet.get("pmcid") or snippet.get("corpus_id", "")
            
            if not corpus_id:
                continue
            
            if corpus_id not in paper_snippets:
                paper_snippets[corpus_id] = {
                    "corpus_id": corpus_id,
                    "pmcid": corpus_id,
                    "pmid": snippet.get("pmid"),
                    "title": snippet.get("title", ""),
                    "abstract": snippet.get("abstract", ""),
                    "venue": snippet.get("journal") or snippet.get("venue", ""),
                    "year": snippet.get("year"),
                    "authors": snippet.get("authors", []),
                    "article_type": snippet.get("article_type", "other"),
                    "sentences": [],
                    "relevance_judgement": -1,
                    "citation_count": snippet.get("citation_count", 0),
                }
            
            # Add sentence/snippet
            paper_snippets[corpus_id]["sentences"].append({
                "text": snippet.get("text", ""),
                "section_title": snippet.get("section_title", "abstract"),
                "char_start_offset": snippet.get("char_offset", 0),
            })
            
            # Update relevance using max boosted_score (evidence hierarchy aware)
            current_score = paper_snippets[corpus_id]["relevance_judgement"]
            # Prefer boosted_score (with evidence hierarchy) over raw rerank_score
            new_score = snippet.get("boosted_score", snippet.get("rerank_score", snippet.get("score", 0)))
            paper_snippets[corpus_id]["relevance_judgement"] = max(current_score, new_score)
            
            # Update abstract if from abstract section
            if snippet.get("section_title") == "abstract" and not paper_snippets[corpus_id]["abstract"]:
                paper_snippets[corpus_id]["abstract"] = snippet.get("text", "")
        
        # Sort by relevance
        sorted_papers = sorted(
            paper_snippets.values(),
            key=lambda x: x["relevance_judgement"],
            reverse=True
        )
        
        # Apply context threshold
        sorted_papers = [
            p for p in sorted_papers
            if p["relevance_judgement"] >= self.context_threshold
        ]
        
        logger.info(f"Aggregated to {len(sorted_papers)} papers")
        logger.info(f"Scores: {[p['relevance_judgement'] for p in sorted_papers[:10]]}")
        
        return sorted_papers
    
    def format_retrieval_response(
        self,
        agg_candidates: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Format into DataFrame with ScholarQA reference strings.
        
        Creates:
        - reference_string: "[corpus_id | author | year | Citations: N]"
        - relevance_judgment_input_expanded: Formatted paper content
        """
        if not agg_candidates:
            return pd.DataFrame()
        
        def format_sections_to_markdown(sentences: List[Dict]) -> str:
            """Format paper sections to markdown."""
            if not sentences:
                return ""
            
            df = pd.DataFrame(sentences)
            if df.empty:
                return ""
            
            # Sort by offset
            if "char_start_offset" in df.columns:
                df = df.sort_values(by="char_start_offset")
            
            # Group by section
            grouped = df.groupby("section_title", sort=False)["text"].apply("\n...\n".join)
            
            # Exclude abstract (already in prepend)
            grouped = grouped[~grouped.index.isin(["abstract", "title"])]
            
            return "\n\n".join(f"## {title}\n{text}" for title, text in grouped.items())
        
        df = pd.DataFrame(agg_candidates)
        
        if df.empty:
            return df
        
        # Format authors
        df["authors"] = df["authors"].fillna("").apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        # Create prepend text (title, venue, authors, abstract)
        prepend_text = df.apply(
            lambda row: (
                f"# Title: {row['title']}\n"
                f"# Venue: {row['venue']}\n"
                f"# Authors: {', '.join([a.get('name', a) if isinstance(a, dict) else str(a) for a in row['authors']])}\n"
                f"## Abstract\n{row['abstract']}\n"
            ),
            axis=1
        )
        
        # Format sections
        section_text = df["sentences"].apply(format_sections_to_markdown)
        
        # Create expanded input
        df["relevance_judgment_input_expanded"] = prepend_text + section_text
        
        # Create reference string (ScholarQA format with PMCID)
        df["reference_string"] = df.apply(
            lambda row: anyascii(
                f"[{row['pmcid']} | {get_ref_author_str(row['authors'])} | "
                f"{make_int(row['year'])} | Citations: {make_int(row.get('citation_count', 0))}]"
            ),
            axis=1
        )
        
        logger.info(f"Formatted {len(df)} papers into DataFrame")
        return df
    
    def aggregate_into_dataframe(
        self,
        snippets_list: List[Dict[str, Any]],
        paper_metadata: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Full pipeline: aggregate then format (ScholarQA interface).
        """
        aggregated = self.aggregate_snippets_to_papers(snippets_list, paper_metadata)
        return self.format_retrieval_response(aggregated)


# For backward compatibility
ArticleReranker = PaperFinderWithReranker


if __name__ == "__main__":
    print("🧪 Testing ScholarQA-style Reranker")
    print("=" * 60)
    
    # Test reranker
    reranker = CohereReranker()
    
    sample_passages = [
        {"pmcid": "PMC001", "title": "COPD Treatment", "text": "COPD management...", "year": 2023, "journal": "Lancet", "authors": [{"name": "Smith J"}]},
        {"pmcid": "PMC002", "title": "Diabetes Care", "text": "Type 2 diabetes...", "year": 2024, "journal": "NEJM", "authors": [{"name": "Doe A"}, {"name": "Lee B"}]},
    ]
    
    finder = PaperFinderWithReranker()
    reranked = finder.rerank("COPD management", sample_passages)
    print(f"Reranked {len(reranked)} passages")
    
    df = finder.aggregate_into_dataframe(reranked)
    print(f"DataFrame: {len(df)} rows")
    print(f"Reference strings: {df['reference_string'].tolist()}")
