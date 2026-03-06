"""
Query preprocessor for Medical RAG retrieval.

Responsibilities:
- Correct likely medical typos
- Distill verbose prompts (including long clinical vignettes) into compact retrieval queries
- Extract key entities for downstream entity filtering
- Detect explicit drug-information intent for DailyMed routing
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, NamedTuple

from groq import Groq
from openai import OpenAI
from pydantic import BaseModel, Field

from .config import (
    DEEPINFRA_API_KEY,
    DEEPINFRA_BASE_URL,
    GROQ_API_KEY,
    LLM_PROVIDER,
    LLM_RETRY_COUNT,
    LLM_RETRY_DELAY,
    LLM_CHAT_TIMEOUT_SECONDS,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_REASONING_EFFORT,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    QUERY_EXPANSION_COUNT,
)
from .medical_entity_expander import MedicalEntityExpander
from .retry_utils import retry_with_exponential_backoff

logger = logging.getLogger(__name__)


class DecomposedQuery(BaseModel):
    """Minimal query decomposition output from LLM."""

    corrected_query: str = Field(
        default="", description="Typo-corrected query text (empty if no correction needed)."
    )
    primary_query: str = Field(
        description="Primary semantic retrieval query. Keep compact and clinically specific."
    )
    keyword_query: str = Field(
        description="Compact keyword retrieval query for sparse/BM25 retrieval."
    )
    key_entities: List[str] = Field(
        default=[],
        description="Key diseases/conditions/findings from the user query (original spelling).",
    )
    corrected_entities: List[str] = Field(
        default=[],
        description="Typo-corrected key entities for matching and recall.",
    )
    is_drug_query: bool = Field(
        default=False,
        description="True only when user intent is medication information (dose/adverse effects/interactions/contraindications/MOA).",
    )
    drug_names: List[str] = Field(
        default=[], description="Pharmaceutical drug names only (generic and applicable brand names)."
    )


class LLMProcessedQuery(NamedTuple):
    """Result of LLM query processing for retrieval."""

    primary_query: str
    keyword_query: str
    original_query: str
    decomposed: Optional[DecomposedQuery] = None
    retrieval_queries: List[str] = []


class QueryExpansionResult(NamedTuple):
    """Result of query expansion (backward compatibility)."""

    original_query: str
    expanded_queries: List[str]
    all_queries: List[str]


QUERY_DECOMPOSER_PROMPT = """
<task>
You are a medical retrieval query preprocessor.
Analyze the user query and output compact retrieval-focused JSON.
If the input is a long clinical vignette (USMLE style), distill it into a concise 15-25 word primary_query that keeps only high-yield findings and question intent; never copy the full vignette.
Set is_drug_query=true ONLY when the user explicitly asks medication information (dose, adverse effects, interactions, contraindications, mechanism, administration).
If a drug is only background history in a case vignette, set is_drug_query=false and drug_names=[].
Current year: __CURRENT_YEAR__
</task>

<fields>
corrected_query: Full typo-corrected text only when you are confident there is a misspelling; otherwise "".
primary_query: Main dense retrieval query. Compact and specific. Remove answer option blocks like (A)...(E).
keyword_query: Compact sparse retrieval query. May include condensed option terms for differential diagnosis.
key_entities: Key conditions/findings from user query (original spelling).
corrected_entities: Corrected spellings of key_entities when applicable.
is_drug_query: boolean with strict intent rule from <task>.
drug_names: Pharmaceutical names only (generic + applicable brand names). Never output conditions, anatomy, procedures, symptoms, labs, or demographics here.
</fields>

<examples>
<example input>
What are the contraindications and adverse effects of lisinopril?
</example input>
<example output>
{
  "corrected_query": "",
  "primary_query": "lisinopril contraindications adverse effects",
  "keyword_query": "lisinopril contraindication adverse effects safety",
  "key_entities": ["hypertension"],
  "corrected_entities": [],
  "is_drug_query": true,
  "drug_names": ["lisinopril"]
}
</example output>

<example input>
A 68-year-old female presents with 5 days of fever, chills, and painful swelling in the right groin with leukocytosis and neutrophilia. She has cat exposure. CT shows enlarged inflamed inguinal lymph node without abscess. What is the most likely diagnosis? (A) Lymphogranuloma venereum (B) Cat scratch disease (C) Pyogenic lymphadenitis (D) Inguinal hernia (E) Necrotizing fasciitis
</example input>
<example output>
{
  "corrected_query": "",
  "primary_query": "inguinal lymphadenopathy tender groin mass fever cat exposure leukocytosis neutrophilia differential diagnosis",
  "keyword_query": "inguinal lymphadenopathy fever cat scratch disease pyogenic lymphadenitis neutrophilia diagnosis",
  "key_entities": ["inguinal lymphadenopathy", "fever"],
  "corrected_entities": [],
  "is_drug_query": false,
  "drug_names": []
}
</example output>
</examples>

Output valid JSON only. No markdown.
"""


class QueryPreprocessor:
    """LLM-powered preprocessing for retrieval-focused medical queries."""

    _DRUG_INTENT_TERMS = (
        "dose",
        "dosing",
        "dosage",
        "contraindication",
        "contraindications",
        "adverse",
        "side effect",
        "side effects",
        "interaction",
        "interactions",
        "mechanism",
        "moa",
        "pharmacokinet",
        "administration",
    )

    _FALLBACK_STOPWORDS = {
        "the",
        "and",
        "with",
        "from",
        "that",
        "this",
        "have",
        "been",
        "into",
        "what",
        "which",
        "when",
        "where",
        "does",
        "about",
        "without",
        "normal",
        "reports",
        "patient",
        "female",
        "male",
        "year",
        "years",
        "old",
    }

    def __init__(self, model: str = LLM_MODEL, use_entity_expansion: bool = True):
        """Initialize query decomposition LLM client."""
        if LLM_PROVIDER == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not set")
            self.llm_client = Groq(
                api_key=GROQ_API_KEY,
                timeout=LLM_CHAT_TIMEOUT_SECONDS,
            )
        elif LLM_PROVIDER == "deepinfra":
            if not DEEPINFRA_API_KEY:
                raise ValueError("DEEPINFRA_API_KEY not set")
            self.llm_client = OpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_BASE_URL,
                timeout=LLM_CHAT_TIMEOUT_SECONDS,
            )
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

        self.llm_provider = LLM_PROVIDER
        self.model = model
        self.expansion_count = QUERY_EXPANSION_COUNT
        self.retry_count = LLM_RETRY_COUNT
        self.retry_delay = LLM_RETRY_DELAY
        self.use_entity_expansion = use_entity_expansion
        self._entity_expander = None
        logger.info("LLM query preprocessor provider initialized: %s (%s)", self.llm_provider, self.model)

        if use_entity_expansion:
            try:
                self._entity_expander = MedicalEntityExpander()
                logger.info("✅ Medical entity expander initialized")
            except Exception as exc:
                logger.warning("Failed to initialize medical entity expander: %s", exc)
                self.use_entity_expansion = False

    def _should_expand_medical_entities(self, query: str) -> bool:
        """Expand entities only when it helps recall without bloating long vignettes."""
        tokens = query.split()
        acronym_hits = len(re.findall(r"\b[A-Z]{2,10}\b", query))
        return len(tokens) <= 45 or acronym_hits >= 2

    def _expand_medical_entities(self, query: str) -> str:
        """Expand medical acronyms in query when enabled and likely beneficial."""
        if not (self.use_entity_expansion and self._entity_expander):
            return query
        if not self._should_expand_medical_entities(query):
            return query

        try:
            expanded = self._entity_expander.expand_query(query, preserve_original=True)
            if expanded != query:
                logger.info("Expanded query: '%s' → '%s'", query, expanded)
            return expanded
        except Exception as exc:
            logger.warning("Entity expansion failed: %s", exc)
            return query

    def _normalize_response_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Map legacy keys to the minimal schema for robustness."""
        normalized = dict(payload)

        if "primary_query" not in normalized and "rewritten_query" in normalized:
            normalized["primary_query"] = normalized.get("rewritten_query", "")
        if "keyword_query" not in normalized and "rewritten_query_for_keyword_search" in normalized:
            normalized["keyword_query"] = normalized.get("rewritten_query_for_keyword_search", "")
        if "key_entities" not in normalized and "medical_conditions" in normalized:
            normalized["key_entities"] = normalized.get("medical_conditions") or []
        if "corrected_entities" not in normalized and "corrected_medical_conditions" in normalized:
            normalized["corrected_entities"] = normalized.get("corrected_medical_conditions") or []

        normalized.setdefault("corrected_query", "")
        normalized.setdefault("key_entities", [])
        normalized.setdefault("corrected_entities", [])
        normalized.setdefault("drug_names", [])
        normalized.setdefault("is_drug_query", False)

        if not isinstance(normalized.get("is_drug_query"), bool):
            normalized["is_drug_query"] = str(normalized.get("is_drug_query", "")).strip().lower() in {
                "true",
                "1",
                "yes",
                "on",
            }

        return normalized

    def _parse_llm_response(self, content: str) -> DecomposedQuery:
        """Parse LLM response, handling markdown wrappers and legacy keys."""
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        payload = self._normalize_response_payload(json.loads(content))
        return DecomposedQuery(**payload)

    def _get_query_decomposer_prompt(self) -> str:
        """Render query decomposer prompt with current year."""
        return QUERY_DECOMPOSER_PROMPT.replace("__CURRENT_YEAR__", str(datetime.now().year))

    def _strip_mcq_options(self, text: str) -> str:
        """Strip multiple-choice answer option blocks such as (A) ... (E)."""
        option_marker = re.compile(r"(?:\([A-E]\)|\b[A-E]\))\s*", flags=re.IGNORECASE)
        if not option_marker.search(text):
            return re.sub(r"\s+", " ", text).strip()
        stem = option_marker.split(text, maxsplit=1)[0]
        return re.sub(r"\s+", " ", stem).strip()

    def _extract_option_terms(self, text: str, limit: int = 10) -> List[str]:
        """Extract compact diagnostic hints from MCQ options for sparse query variant."""
        option_marker = re.compile(r"(?:\([A-E]\)|\b[A-E]\))\s*", flags=re.IGNORECASE)
        chunks = option_marker.split(text)
        if len(chunks) <= 1:
            return []

        terms: List[str] = []
        for chunk in chunks[1:]:
            cleaned = re.sub(r"[^a-zA-Z0-9\-\s]", " ", chunk).lower()
            tokens = [tok for tok in cleaned.split() if len(tok) >= 3 and tok not in self._FALLBACK_STOPWORDS]
            if not tokens:
                continue
            terms.append(" ".join(tokens[:4]))
            if len(terms) >= limit:
                break
        return terms[:limit]

    def _compact_text(self, text: str, max_tokens: int) -> str:
        """Normalize whitespace and truncate to max token budget."""
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return ""
        tokens = normalized.split()
        return " ".join(tokens[:max_tokens])

    def _is_long_vignette(self, query: str) -> bool:
        """Heuristic detection for long case-vignette prompts."""
        lowered = query.lower()
        indicators = (
            "presents",
            "physical examination",
            "laboratory",
            "blood pressure",
            "ct scan",
            "history",
        )
        return len(query.split()) >= 120 or sum(1 for term in indicators if term in lowered) >= 3

    def _extract_key_entities_fallback(self, text: str, limit: int = 6) -> List[str]:
        """Best-effort extraction for key entities when LLM fails."""
        cleaned = re.sub(r"[^a-zA-Z0-9\-\s]", " ", text.lower())
        candidates = [
            tok
            for tok in cleaned.split()
            if len(tok) >= 5 and tok not in self._FALLBACK_STOPWORDS
        ]
        deduped = list(dict.fromkeys(candidates))
        return deduped[:limit]

    def _guess_drug_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in self._DRUG_INTENT_TERMS)

    def _build_retrieval_queries(self, primary: str, keyword: str) -> List[str]:
        """Return authoritative retrieval query list with dedupe and non-empty fallback."""
        queries: List[str] = []
        for value in (primary, keyword):
            cleaned = re.sub(r"\s+", " ", (value or "")).strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
        if not queries:
            return [primary or keyword or ""]
        return queries

    def _fallback_decompose(self, query: str, expanded_query: str) -> DecomposedQuery:
        """Deterministic fallback decomposition when LLM parsing/call fails."""
        stripped = self._strip_mcq_options(expanded_query)
        base = stripped or expanded_query or query

        primary_limit = 25 if self._is_long_vignette(query) else 45
        primary_query = self._compact_text(base, max_tokens=primary_limit)

        option_terms = self._extract_option_terms(query)
        key_entities = self._extract_key_entities_fallback(stripped or query)

        keyword_parts = key_entities + option_terms
        if not keyword_parts:
            keyword_parts = primary_query.split()[:24]
        keyword_query = self._compact_text(" ".join(keyword_parts), max_tokens=24)

        return DecomposedQuery(
            corrected_query="",
            primary_query=primary_query,
            keyword_query=keyword_query or primary_query,
            key_entities=key_entities,
            corrected_entities=[],
            is_drug_query=self._guess_drug_intent(query),
            drug_names=[],
        )

    def _chat_completion_with_retry(self, messages: List[Dict[str, str]], operation_name: str):
        """Execute chat completion with exponential-backoff retry."""
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
        }
        if self.llm_provider == "groq":
            request_kwargs["max_completion_tokens"] = LLM_MAX_COMPLETION_TOKENS
            if LLM_REASONING_EFFORT:
                request_kwargs["reasoning_effort"] = LLM_REASONING_EFFORT

        return retry_with_exponential_backoff(
            lambda: self.llm_client.chat.completions.create(**request_kwargs),
            max_attempts=self.retry_count + 1,
            base_delay=float(self.retry_delay),
            operation_name=operation_name,
            logger=logger,
        )

    def decompose_query(self, query: str) -> LLMProcessedQuery:
        """Decompose query into compact retrieval inputs using LLM."""
        expanded_query = self._expand_medical_entities(query)

        try:
            response = self._chat_completion_with_retry(
                messages=[
                    {"role": "system", "content": self._get_query_decomposer_prompt()},
                    {"role": "user", "content": expanded_query},
                ],
                operation_name=f"{self.llm_provider} query decomposition",
            )
            decomposed = self._parse_llm_response(response.choices[0].message.content.strip())
        except Exception as exc:
            logger.warning("Query decomposition failed: %s; using deterministic fallback", exc)
            decomposed = self._fallback_decompose(query, expanded_query)

        corrected = decomposed.corrected_query.strip()
        primary_query = corrected or decomposed.primary_query or expanded_query

        # For keyword retrieval, add corrected entities to improve exact term hits.
        keyword_query = decomposed.keyword_query or primary_query
        if decomposed.corrected_entities:
            keyword_query = " ".join([keyword_query, " ".join(decomposed.corrected_entities)]).strip()

        # Keep concise query budget, especially for long vignettes.
        if self._is_long_vignette(query):
            primary_query = self._compact_text(self._strip_mcq_options(primary_query), max_tokens=25)
            keyword_query = self._compact_text(keyword_query, max_tokens=24)
        else:
            primary_query = self._compact_text(primary_query, max_tokens=45)
            keyword_query = self._compact_text(keyword_query, max_tokens=28)

        # Add option hints for keyword query in MCQ prompts.
        option_terms = self._extract_option_terms(query)
        if option_terms:
            keyword_query = self._compact_text(
                f"{keyword_query} {' '.join(option_terms)}",
                max_tokens=24,
            )

        # If LLM returns no explicit drug flag, infer conservatively from intent + extracted drugs.
        if not decomposed.is_drug_query and decomposed.drug_names and self._guess_drug_intent(query):
            decomposed = decomposed.model_copy(update={"is_drug_query": True})

        retrieval_queries = self._build_retrieval_queries(primary_query, keyword_query)

        logger.info("Decomposed query: %s", decomposed)
        logger.info("Primary query: %s", primary_query)
        logger.info("Keyword query: %s", keyword_query)

        return LLMProcessedQuery(
            primary_query=primary_query,
            keyword_query=keyword_query,
            original_query=query,
            decomposed=decomposed,
            retrieval_queries=retrieval_queries,
        )

    def expand_query(self, query: str) -> QueryExpansionResult:
        """Expand query using LLM (backward compatibility method)."""
        system_prompt = (
            "You are a medical search query expansion expert. "
            f"Generate exactly {self.expansion_count} alternative search queries for medical literature.\n\n"
            "RULES:\n"
            "1. Each query should approach the topic from a different angle\n"
            "2. Use medical synonyms and related terminology\n"
            "3. Include both technical and layman terms\n"
            "4. Output ONLY the queries, one per line, no numbering or bullets"
        )

        try:
            response = self._chat_completion_with_retry(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f'Generate {self.expansion_count} alternative medical search queries for:\n\n"{query}"\n\n'
                            "Output only the queries, one per line:"
                        ),
                    },
                ],
                operation_name=f"{self.llm_provider} query expansion",
            )

            expanded = []
            for line in response.choices[0].message.content.strip().split("\n"):
                cleaned = line.strip().lstrip("0123456789.-•*) ").strip()
                if cleaned and len(cleaned) > 10:
                    expanded.append(cleaned)

            expanded = expanded[: self.expansion_count]
            return QueryExpansionResult(query, expanded, [query] + expanded)

        except Exception as exc:
            logger.warning("Query expansion failed: %s", exc)
            return QueryExpansionResult(query, [], [query])

    def preprocess(self, query: str, use_decomposition: bool = True) -> LLMProcessedQuery:
        """Full preprocessing pipeline."""
        if use_decomposition:
            return self.decompose_query(query)

        expansion = self.expand_query(query)
        retrieval_queries = expansion.all_queries[:2] if expansion.all_queries else [query]
        primary_query = retrieval_queries[0] if retrieval_queries else query
        keyword_query = retrieval_queries[1] if len(retrieval_queries) > 1 else primary_query
        return LLMProcessedQuery(
            primary_query=primary_query,
            keyword_query=keyword_query,
            original_query=query,
            decomposed=None,
            retrieval_queries=retrieval_queries,
        )


if __name__ == "__main__":
    print("Testing Query Preprocessor")
    print("=" * 60)

    preprocessor = QueryPreprocessor()

    test_queries = [
        "What are the latest treatments for heart failure?",
        "Systematic reviews on metformin for type 2 diabetes from 2020",
        "Management of NUEROBROCELLOSIS",
        (
            "A 68-year-old female presents with 5 days of fever, chills, and painful swelling in the right groin. "
            "CT shows enlarged inflamed inguinal node. What is the most likely diagnosis? "
            "(A) Lymphogranuloma venereum (B) Cat scratch disease (C) Pyogenic lymphadenitis"
        ),
    ]

    for query in test_queries:
        print(f"\nOriginal: {query}")
        print("-" * 40)

        result = preprocessor.decompose_query(query)

        print(f"Primary: {result.primary_query}")
        print(f"Keyword: {result.keyword_query}")
        print(f"Retrieval queries: {result.retrieval_queries}")
        if result.decomposed:
            print(f"Drug intent: {result.decomposed.is_drug_query}")
            print(f"Drugs: {result.decomposed.drug_names}")
