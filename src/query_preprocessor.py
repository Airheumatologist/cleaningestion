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

from .config import DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL, DEEPINFRA_MODEL, LLM_TEMPERATURE, LLM_TOP_P, QUERY_EXPANSION_COUNT
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
    drug_names: List[str] = Field(default=[], description="List of drug names (brand and generic) mentioned in query")
    medical_conditions: List[str] = Field(default=[], description="Key medical conditions/diseases mentioned in query")


class LLMProcessedQuery(NamedTuple):
    """Result of LLM query processing."""
    rewritten_query: str
    keyword_query: str
    search_filters: Dict[str, Any]
    original_query: str
    decomposed: Optional[DecomposedQuery] = None
    expanded_queries: List[str] = []


class QueryExpansionResult(NamedTuple):
    """Result of query expansion (backward compatibility)."""
    original_query: str
    expanded_queries: List[str]
    all_queries: List[str]


# =============================================================================
# Prompts
# =============================================================================

QUERY_DECOMPOSER_PROMPT = """
<task>
Your task is to analyze a medical/clinical query and break it down for searching PubMed Central articles and DailyMed drug labels.
Create a structured JSON output for academic and clinical search.

CRITICAL: Preserve all medical condition names, disease names, and medical acronyms in the rewritten queries.
Do NOT remove or abbreviate disease names - they are essential for accurate retrieval.

Components to extract:
1. Publication years: If "recent" → 2022-2025. If "last 5 years" → 2020-2025. Leave blank if unspecified.
2. Venues: Journal names mentioned (e.g., "NEJM", "Lancet", "JAMA"). Leave blank if none.
3. Authors: Author names mentioned. Leave as empty array if none.
4. Field of study: Map to one of: Medicine, Biology, Chemistry, Pharmacology, Psychology, Nursing, Public Health. 
   Default to "Medicine" for general medical queries.
5. Rewritten query: Simplify for semantic vector search. ALWAYS include full disease/condition names.
   - If query contains acronyms (e.g., "APS"), include both the acronym AND the full term
   - Preserve medical condition names exactly as they appear
   - Remove only metadata (years, venues) already extracted
6. Keyword query: Extract key medical terms for keyword matching. Include both acronyms and full terms.
7. Drug names: COMPREHENSIVE drug name extraction. Include:
   - Generic name AND brand name equivalents (e.g., "tofacitinib" → ["tofacitinib", "Xeljanz"])
   - FORMULATION-SPECIFIC BRANDS: Many drugs have different brand names for different routes of administration.
     * If a specific route IS mentioned (infusion, IV, subcutaneous, SC, injection, oral):
       Include ONLY the brand for that route (e.g., "golimumab infusion" → ["golimumab", "SIMPONI ARIA"])
     * If NO specific route is mentioned:
       Include ALL formulation brands for comprehensive coverage (e.g., "golimumab dosing" → ["golimumab", "SIMPONI", "SIMPONI ARIA"])
   - Common examples of multi-formulation drugs:
     * Golimumab: SIMPONI (SC injection) vs SIMPONI ARIA (IV infusion)
     * Rituximab: RITUXAN (IV) vs RITUXAN HYCELA (SC)
     * Trastuzumab: HERCEPTIN (IV) vs HERCEPTIN HYLECTA (SC)
     * Tocilizumab: ACTEMRA (both IV and SC, same brand)
8. Medical conditions: List key diseases or clinical conditions mentioned.

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
    "rewritten_query_for_keyword_search": "HFpEF treatment therapy heart failure preserved ejection fraction",
    "drug_names": [],
    "medical_conditions": ["heart failure", "HFpEF"]
}
</example output>

<example input>
golimumab infusion dosing for rheumatoid arthritis
</example input>
<example output>
{
    "earliest_search_year": "",
    "latest_search_year": "",
    "venues": "",
    "authors": [],
    "field_of_study": "Medicine",
    "rewritten_query": "golimumab IV infusion dosing rheumatoid arthritis",
    "rewritten_query_for_keyword_search": "golimumab SIMPONI ARIA infusion IV dosing rheumatoid arthritis RA",
    "drug_names": ["golimumab", "SIMPONI ARIA"],
    "medical_conditions": ["rheumatoid arthritis"]
}
</example output>

<example input>
golimumab dosing for RA
</example input>
<example output>
{
    "earliest_search_year": "",
    "latest_search_year": "",
    "venues": "",
    "authors": [],
    "field_of_study": "Medicine",
    "rewritten_query": "golimumab dosing rheumatoid arthritis",
    "rewritten_query_for_keyword_search": "golimumab SIMPONI dosing rheumatoid arthritis RA",
    "drug_names": ["golimumab", "SIMPONI", "SIMPONI ARIA"],
    "medical_conditions": ["rheumatoid arthritis"]
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
    "rewritten_query_for_keyword_search": "SGLT2 inhibitor systematic review meta-analysis diabetes heart failure",
    "drug_names": ["SGLT2 inhibitors"],
    "medical_conditions": ["diabetes", "heart failure"]
}
</example output>

<example input>
rituximab subcutaneous vs IV administration
</example input>
<example output>
{
    "earliest_search_year": "",
    "latest_search_year": "",
    "venues": "",
    "authors": [],
    "field_of_study": "Medicine",
    "rewritten_query": "rituximab subcutaneous versus intravenous IV administration comparison",
    "rewritten_query_for_keyword_search": "rituximab RITUXAN RITUXAN HYCELA SC IV subcutaneous intravenous",
    "drug_names": ["rituximab", "RITUXAN", "RITUXAN HYCELA"],
    "medical_conditions": []
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
    
    def __init__(self, model: str = DEEPINFRA_MODEL, use_entity_expansion: bool = True):
        """Initialize with DeepInfra client."""
        if not DEEPINFRA_API_KEY:
            raise ValueError("DEEPINFRA_API_KEY not set")

        self.openai_client = OpenAI(
            api_key=DEEPINFRA_API_KEY,
            base_url=DEEPINFRA_BASE_URL,
            timeout=300.0
        )
        self.model = model
        self.expansion_count = QUERY_EXPANSION_COUNT
        self.use_entity_expansion = use_entity_expansion
        self._entity_expander = None
        
        if use_entity_expansion:
            try:
                self._entity_expander = MedicalEntityExpander()
                logger.info("✅ Medical entity expander initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize medical entity expander: {e}")
                self.use_entity_expansion = False
    
    def _expand_medical_entities(self, query: str) -> str:
        """Expand medical acronyms in query if enabled."""
        if not (self.use_entity_expansion and self._entity_expander):
            return query
        try:
            expanded = self._entity_expander.expand_query(query, preserve_original=True)
            if expanded != query:
                logger.info(f"Expanded query: '{query}' → '{expanded}'")
            return expanded
        except Exception as e:
            logger.warning(f"Entity expansion failed: {e}")
            return query
    
    def _parse_llm_response(self, content: str) -> DecomposedQuery:
        """Parse LLM response, handling markdown formatting."""
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        return DecomposedQuery(**json.loads(content))
    
    def _extract_search_filters(self, decomposed: DecomposedQuery) -> Dict[str, Any]:
        """Extract search filters from decomposed query."""
        filters = {}
        
        if decomposed.earliest_search_year or decomposed.latest_search_year:
            filters["year"] = f"{decomposed.earliest_search_year}-{decomposed.latest_search_year}"
        if decomposed.venues:
            filters["venue"] = decomposed.venues
        if decomposed.field_of_study:
            filters["field_of_study"] = decomposed.field_of_study
        if decomposed.authors:
            authors = decomposed.authors if isinstance(decomposed.authors, list) else [decomposed.authors]
            if authors and authors[0]:
                filters["authors"] = authors
        
        return filters
    
    def _extract_condition(self, query: str, decomposed: Optional[DecomposedQuery] = None) -> str:
        """Extract the medical condition/disease from a query or decomposed data."""
        if decomposed and decomposed.medical_conditions:
            # Use the most specific condition from LLM
            return decomposed.medical_conditions[0]
            
        # Fallback to regex logic
        condition_match = re.search(
            r'((?:[A-Z][a-z]*\d*[-]?)+(?:\s+[a-z]+)*\s*(?:disease|syndrome|disorder|condition)?)',
            query, re.IGNORECASE
        )
        condition = condition_match.group(1).strip() if condition_match else query
        
        acronym_match = re.search(r'\b([A-Z]{2,}(?:\d+)?)\b', query)
        if acronym_match:
            acronym = acronym_match.group(1)
            if acronym.lower() not in condition.lower():
                condition = f"{condition} ({acronym})"
        
        return condition
    
    def _generate_query_variations(self, rewritten: str, keyword: str, expanded: str, decomposed: Optional[DecomposedQuery] = None) -> List[str]:
        """
        Generate semantically diverse query variations for multi-query retrieval.
        
        Creates queries exploring different aspects: treatment, diagnosis, guidelines, etc.
        """
        base_query = rewritten or keyword or expanded
        condition = self._extract_condition(base_query, decomposed)
        logger.info(f"Extracted condition for expansion: '{condition}'")
        
        # Optimization: Detect query intent to avoid redundant variations
        query_lower = base_query.lower()
        is_dosing = any(word in query_lower for word in ["dose", "dosing", "dosage", "amount", "mg", "frequency"])
        is_treatment = any(word in query_lower for word in ["treat", "therapy", "management", "drug", "medication"])
        is_diagnosis = any(word in query_lower for word in ["diagnose", "diagnosis", "criteria", "test", "screening"])
        
        # Define query angles with templates
        query_templates = [base_query]
        
        # Add intent-specific queries first
        if is_dosing:
            query_templates.append(f"{condition} dosing protocol administration")
        if is_treatment or is_dosing:
            query_templates.append(f"treatment therapy management of {condition}")
        if is_diagnosis:
            query_templates.append(f"diagnosis clinical features criteria of {condition}")
            
        # Add general medical angles
        query_templates.extend([
            f"guidelines recommendations for {condition}",
            f"pathogenesis pathophysiology mechanism of {condition}",
            f"clinical trial outcomes {condition}",
        ])
        
        # Deduplicate while preserving order
        seen = set()
        variations = []
        for q in query_templates:
            if q and q.lower() not in seen:
                variations.append(q)
                seen.add(q.lower())
            if len(variations) >= self.expansion_count:
                break
        
        logger.info(f"Generated {len(variations)} diverse query variations")
        for i, v in enumerate(variations, 1):
            logger.info(f"  {i}. {v}")
        
        return variations

    def decompose_query(self, query: str) -> LLMProcessedQuery:
        """
        Decompose query into structured components using LLM.
        
        Returns:
            LLMProcessedQuery with rewritten query, keyword query, filters, and expanded queries
        """
        expanded_query = self._expand_medical_entities(query)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUERY_DECOMPOSER_PROMPT},
                    {"role": "user", "content": expanded_query}
                ],
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P
            )

            decomposed = self._parse_llm_response(response.choices[0].message.content.strip())
            logger.info(f"Decomposed query: {decomposed}")
            
            # Extract filters and queries
            search_filters = self._extract_search_filters(decomposed)
            rewritten = decomposed.rewritten_query or expanded_query
            keyword = decomposed.rewritten_query_for_keyword_search or expanded_query
            
            # Ensure expanded terms are included if entity expansion was used
            if expanded_query != query:
                if query.upper() in rewritten.upper() and expanded_query not in rewritten:
                    rewritten = f"{rewritten} {expanded_query}"
                if query.upper() in keyword.upper() and expanded_query not in keyword:
                    keyword = f"{keyword} {expanded_query}"
            
            expanded_queries = self._generate_query_variations(rewritten, keyword, expanded_query, decomposed)
            
            return LLMProcessedQuery(
                rewritten_query=rewritten,
                keyword_query=keyword,
                search_filters=search_filters,
                original_query=query,
                decomposed=decomposed,
                expanded_queries=expanded_queries
            )
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}, falling back to basic processing")
            return LLMProcessedQuery(
                rewritten_query=query,
                keyword_query=query,
                search_filters={},
                original_query=query,
                decomposed=None,
                expanded_queries=[query]
            )
    
    def expand_query(self, query: str) -> QueryExpansionResult:
        """
        Expand query using LLM (backward compatibility method).
        """
        system_prompt = f"""You are a medical search query expansion expert. 
Generate exactly {self.expansion_count} alternative search queries for medical literature.

RULES:
1. Each query should approach the topic from a different angle
2. Use medical synonyms and related terminology
3. Include both technical and layman terms
4. Output ONLY the queries, one per line, no numbering or bullets"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Generate {self.expansion_count} alternative medical search queries for:\n\n"{query}"\n\nOutput only the queries, one per line:'}
                ],
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P
            )

            expanded = []
            for line in response.choices[0].message.content.strip().split('\n'):
                cleaned = line.strip().lstrip('0123456789.-•*) ').strip()
                if cleaned and len(cleaned) > 10:
                    expanded.append(cleaned)
            
            expanded = expanded[:self.expansion_count]
            return QueryExpansionResult(query, expanded, [query] + expanded)
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return QueryExpansionResult(query, [], [query])
    
    def preprocess(self, query: str, use_decomposition: bool = True) -> LLMProcessedQuery:
        """
        Full preprocessing pipeline.
        
        Args:
            query: Original user query
            use_decomposition: If True, use structured decomposition; else basic expansion
        """
        if use_decomposition:
            return self.decompose_query(query)
        
        expansion = self.expand_query(query)
        return LLMProcessedQuery(
            rewritten_query=query,
            keyword_query=" ".join(expansion.expanded_queries[:2]),
            search_filters={},
            original_query=query,
            decomposed=None,
            expanded_queries=expansion.all_queries
        )


if __name__ == "__main__":
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
