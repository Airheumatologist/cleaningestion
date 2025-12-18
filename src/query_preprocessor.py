"""
Enhanced Query Preprocessor for Medical RAG Pipeline.

Adapts ScholarQA's query decomposition approach to:
- Extract metadata filters (year, venue, field of study)
- Generate rewritten query for semantic search
- Generate keyword-optimized query for hybrid search
- Support structured Pydantic output parsing
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, NamedTuple, Union
from openai import OpenAI
from pydantic import BaseModel, Field

from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, QUERY_EXPANSION_COUNT
from .medical_entity_expander import MedicalEntityExpander

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class DecomposedQuery(BaseModel):
    """Structured query decomposition output from LLM."""
    earliest_search_year: str = Field(default="", description="The earliest year to search for papers")
    latest_search_year: str = Field(default="", description="The latest year to search for papers")
    venues: str = Field(default="", description="Comma separated list of venues to search for papers")
    authors: Union[List[str], str] = Field(default=[], description="List of authors to search for papers")
    field_of_study: str = Field(default="", description="Comma separated list of field of study")
    rewritten_query: str = Field(description="The rewritten simplified query for semantic search")
    rewritten_query_for_keyword_search: str = Field(description="The keyword-optimized query")


class LLMProcessedQuery(NamedTuple):
    """Result of LLM query processing."""
    rewritten_query: str
    keyword_query: str
    search_filters: Dict[str, Any]
    original_query: str
    decomposed: Optional[DecomposedQuery] = None


class QueryExpansionResult(NamedTuple):
    """Result of query expansion (backward compatibility)."""
    original_query: str
    expanded_queries: List[str]
    all_queries: List[str]


# =============================================================================
# Prompts (adapted from ScholarQA)
# =============================================================================

QUERY_DECOMPOSER_PROMPT = """
<task>
Your task is to analyze a medical/clinical query and break it down for searching PubMed Central articles.
Create a structured JSON output for academic search.

CRITICAL: Preserve all medical condition names, disease names, and medical acronyms in the rewritten queries.
Do NOT remove or abbreviate disease names - they are essential for accurate retrieval.

Components to extract:
1. Publication years: If "recent" → 2022-2025. If "last 5 years" → 2020-2025. Leave blank if unspecified.
2. Venues: Journal names mentioned (e.g., "NEJM", "Lancet", "JAMA"). Leave blank if none.
3. Authors: Author names mentioned. Leave as empty array if none.
4. Field of study: Map to one of: Medicine, Biology, Chemistry, Pharmacology, Psychology, Nursing, Public Health. 
   Default to "Medicine" for general medical queries.
5. Rewritten query: Simplify for semantic vector search. ALWAYS include full disease/condition names.
   - If query contains acronyms (e.g., "APS"), include both the acronym AND the full term (e.g., "Antiphospholipid Syndrome APS")
   - Preserve medical condition names exactly as they appear
   - Remove only metadata (years, venues) already extracted
6. Keyword query: Extract key medical terms for keyword matching. Include both acronyms and full terms.

Current year is 2025.
</task>

<examples>
<example input>
What are the latest treatments for heart failure with preserved ejection fraction?
</example input>
<example output>
{
    "earliest_search_year": "2022",
    "latest_search_year": "2025",
    "venues": "",
    "authors": [],
    "field_of_study": "Medicine",
    "rewritten_query": "Treatments for heart failure with preserved ejection fraction HFpEF",
    "rewritten_query_for_keyword_search": "HFpEF treatment therapy heart failure preserved ejection fraction"
}
</example output>

<example input>
Systematic reviews on SGLT2 inhibitors for diabetes and heart failure from 2020 onwards
</example input>
<example output>
{
    "earliest_search_year": "2020",
    "latest_search_year": "2025",
    "venues": "",
    "authors": [],
    "field_of_study": "Medicine",
    "rewritten_query": "Systematic reviews SGLT2 inhibitors sodium-glucose cotransporter-2 diabetes heart failure",
    "rewritten_query_for_keyword_search": "SGLT2 inhibitor systematic review meta-analysis diabetes heart failure"
}
</example output>

<example input>
What do recent NEJM and Lancet articles say about mRNA vaccine side effects?
</example input>
<example output>
{
    "earliest_search_year": "2022",
    "latest_search_year": "2025",
    "venues": "New England Journal of Medicine,Lancet",
    "authors": [],
    "field_of_study": "Medicine",
    "rewritten_query": "mRNA vaccine adverse effects side effects safety",
    "rewritten_query_for_keyword_search": "mRNA vaccine side effects adverse events safety"
}
</example output>
</examples>

Output valid JSON only. No markdown formatting.
"""


# =============================================================================
# Query Preprocessor Class
# =============================================================================

class QueryPreprocessor:
    """
    Enhanced query preprocessor using ScholarQA-style decomposition.
    
    Features:
    - Medical entity expansion (acronyms → full terms)
    - LLM-based structured query decomposition
    - Metadata extraction (year, venue, field of study)
    - Dual query generation (semantic + keyword)
    - Fallback to basic expansion on errors
    """
    
    def __init__(self, model: str = OPENROUTER_MODEL, use_entity_expansion: bool = True):
        """
        Initialize with OpenRouter client.
        
        Args:
            model: OpenRouter model to use
            use_entity_expansion: If True, expand medical acronyms before decomposition
        """
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set in environment")

        self.openai_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            timeout=300.0
        )  # 5 minutes timeout
        self.model = model
        self.fallback_model = OPENROUTER_MODEL
        self.expansion_count = QUERY_EXPANSION_COUNT
        self.use_entity_expansion = use_entity_expansion
        
        # Initialize medical entity expander (lazy loading)
        self._entity_expander = None
        if use_entity_expansion:
            try:
                self._entity_expander = MedicalEntityExpander()
                logger.info("✅ Medical entity expander initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize medical entity expander: {e}")
                logger.warning("Continuing without entity expansion")
                self.use_entity_expansion = False
    
    def decompose_query(self, query: str) -> LLMProcessedQuery:
        """
        Decompose query into structured components using LLM.
        
        Adapts ScholarQA's decompose_query() for medical domain with Groq.
        Includes medical entity expansion before decomposition.
        
        Args:
            query: Original user query
            
        Returns:
            LLMProcessedQuery with rewritten query, keyword query, and filters
        """
        # Step 1: Expand medical acronyms if enabled
        expanded_query = query
        if self.use_entity_expansion and self._entity_expander:
            try:
                expanded_query = self._entity_expander.expand_query(query, preserve_original=True)
                if expanded_query != query:
                    logger.info(f"Expanded query: '{query}' → '{expanded_query}'")
            except Exception as e:
                logger.warning(f"Entity expansion failed: {e}, using original query")
                expanded_query = query
        
        search_filters = {}
        
        try:
            response = self.openai_client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": QUERY_DECOMPOSER_PROMPT},
                    {"role": "user", "content": expanded_query}
                ],
                text={
                    "format": {
                        "type": "text"
                    }
                },
                reasoning={
                    "effort": "low"
                },
                store=False
            )

            content = response.output_text.strip()
            
            # Clean up potential markdown formatting
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            # Parse JSON response
            decomposed_dict = json.loads(content)
            decomposed = DecomposedQuery(**decomposed_dict)
            
            logger.info(f"Decomposed query: {decomposed}")
            
            # Extract search filters
            if decomposed.earliest_search_year or decomposed.latest_search_year:
                search_filters["year"] = f"{decomposed.earliest_search_year}-{decomposed.latest_search_year}"
            
            if decomposed.venues:
                search_filters["venue"] = decomposed.venues
            
            if decomposed.field_of_study:
                search_filters["field_of_study"] = decomposed.field_of_study
            
            if decomposed.authors:
                authors = decomposed.authors if isinstance(decomposed.authors, list) else [decomposed.authors]
                if authors and authors[0]:
                    search_filters["authors"] = authors
            
            # Ensure medical entities are preserved in both queries
            rewritten = decomposed.rewritten_query or expanded_query
            keyword = decomposed.rewritten_query_for_keyword_search or expanded_query
            
            # If entity expansion was used, ensure expanded terms are in queries
            if self.use_entity_expansion and self._entity_expander and expanded_query != query:
                # Add expanded terms if not already present
                if query.upper() in rewritten.upper() and expanded_query not in rewritten:
                    # Add expanded terms to rewritten query
                    rewritten = f"{rewritten} {expanded_query}"
                if query.upper() in keyword.upper() and expanded_query not in keyword:
                    # Add expanded terms to keyword query
                    keyword = f"{keyword} {expanded_query}"
            
            return LLMProcessedQuery(
                rewritten_query=rewritten,
                keyword_query=keyword,
                search_filters=search_filters,
                original_query=query,
                decomposed=decomposed
            )
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}, falling back to basic processing")
            # Fallback: return original query with no filters
            return LLMProcessedQuery(
                rewritten_query=query,
                keyword_query=query,
                search_filters={},
                original_query=query,
                decomposed=None
            )
    
    def expand_query(self, query: str) -> QueryExpansionResult:
        """
        Expand query into multiple variations (backward compatibility).
        
        Args:
            query: Original user query
            
        Returns:
            QueryExpansionResult with expanded queries
        """
        system_prompt = f"""You are a medical search query expansion expert. 
Generate exactly {self.expansion_count} alternative search queries for medical literature.

RULES:
1. Each query should approach the topic from a different angle
2. Use medical synonyms and related terminology
3. Include both technical and layman terms
4. Output ONLY the queries, one per line, no numbering or bullets"""

        user_prompt = f"""Generate {self.expansion_count} alternative medical search queries for:

"{query}"

Output only the queries, one per line:"""

        try:
            response = self.openai_client.responses.create(
                model=self.fallback_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                text={
                    "format": {
                        "type": "text"
                    }
                },
                reasoning={
                    "effort": "low"
                },
                store=False
            )

            response_text = response.output_text.strip()
            expanded = []
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line:
                    cleaned = line.lstrip('0123456789.-•*) ').strip()
                    if cleaned and len(cleaned) > 10:
                        expanded.append(cleaned)
            
            expanded = expanded[:self.expansion_count]
            
            return QueryExpansionResult(
                original_query=query,
                expanded_queries=expanded,
                all_queries=[query] + expanded
            )
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return QueryExpansionResult(
                original_query=query,
                expanded_queries=[],
                all_queries=[query]
            )
    
    def preprocess(self, query: str, use_decomposition: bool = True) -> LLMProcessedQuery:
        """
        Full preprocessing pipeline.
        
        Args:
            query: Original user query
            use_decomposition: If True, use structured decomposition; else basic expansion
            
        Returns:
            LLMProcessedQuery with all preprocessing results
        """
        if use_decomposition:
            return self.decompose_query(query)
        else:
            expansion = self.expand_query(query)
            return LLMProcessedQuery(
                rewritten_query=query,
                keyword_query=" ".join(expansion.expanded_queries[:2]),
                search_filters={},
                original_query=query,
                decomposed=None
            )


if __name__ == "__main__":
    # Test query preprocessing
    print("🧪 Testing Enhanced Query Preprocessor")
    print("=" * 60)
    
    preprocessor = QueryPreprocessor()
    
    test_queries = [
        "What are the latest treatments for heart failure?",
        "Systematic reviews on metformin for type 2 diabetes from 2020",
        "COVID-19 vaccine side effects in elderly patients",
    ]
    
    for query in test_queries:
        print(f"\n📝 Original: {query}")
        print("-" * 40)
        
        result = preprocessor.decompose_query(query)
        
        print(f"✅ Rewritten: {result.rewritten_query}")
        print(f"🔑 Keyword: {result.keyword_query}")
        print(f"🔍 Filters: {result.search_filters}")
        
        if result.decomposed:
            print(f"📅 Year: {result.decomposed.earliest_search_year}-{result.decomposed.latest_search_year}")
            print(f"📚 Field: {result.decomposed.field_of_study}")
