"""
Elixir Medical RAG Pipeline.

Optimized pipeline for fast, comprehensive medical responses:
1. Query preprocessing with decomposition
2. Qdrant retrieval
3. Cohere reranking with paper aggregation
4. Direct LLM synthesis with ELIXIR system prompt

Designed for speed and quality clinical education responses.
"""

import logging
import re
from typing import List, Dict, Any, Generator
from dataclasses import asdict
from openai import OpenAI

from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, BULK_RETRIEVAL_LIMIT, MAX_ABSTRACTS, MAX_DAILYMED_PER_DRUG
from .query_preprocessor import QueryPreprocessor, LLMProcessedQuery
from .retriever_qdrant import QdrantRetriever
from .reranker import PaperFinderWithReranker
from .prompts import ELIXIR_SYSTEM_PROMPT
from .medical_entity_expander import MedicalEntityExpander

logger = logging.getLogger(__name__)


class MedicalRAGPipeline:
    """
    Elixir Medical RAG Pipeline.
    
    Optimized flow:
    1. Preprocess query (decompose, extract filters)
    2. Retrieve passages from Qdrant
    3. Rerank passages with Cohere
    4. Aggregate to paper level
    5. Direct LLM synthesis with ELIXIR prompt
    """
    
    def __init__(
        self,
        model: str = OPENROUTER_MODEL,
        n_retrieval: int = 150,
        n_rerank: int = -1,  # -1 = no limit
        context_threshold: float = 0.3  # Post-aggregation threshold on boosted_score
        # Note: This is the FINAL threshold after Cohere reranking + entity matching + evidence boosts
        # With evidence tier multipliers (1.0-1.8x), 0.3 ensures good recall while filtering noise
        # Cohere-specific filtering (0.1 minimum) is done earlier in PaperFinderWithReranker.rerank()
    ):

        """Initialize pipeline components."""
        logger.info("🚀 Initializing Elixir Medical RAG Pipeline...")

        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set")

        self.model = model
        self.openai_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            timeout=300.0
        )  # 5 minutes timeout
        
        # Components
        self.preprocessor = QueryPreprocessor(model=model)
        self.retriever = QdrantRetriever(n_retrieval=n_retrieval)
        
        # Initialize medical entity expander for post-retrieval filtering
        try:
            self.entity_expander = MedicalEntityExpander()
            logger.info("✅ Medical entity expander initialized for filtering")
        except Exception as e:
            logger.warning(f"Medical entity expander not available for filtering: {e}")
            self.entity_expander = None

        # Initialize reranker only if Cohere API key is available
        try:
            self.paper_finder = PaperFinderWithReranker(
                n_rerank=n_rerank,
                context_threshold=context_threshold
            )
            logger.info("✅ Cohere reranker initialized")
        except ValueError as e:
            logger.warning(f"⚠️ Cohere reranker not available: {e}")
            logger.warning("⚠️ Falling back to basic retrieval without reranking")
            # Create a basic paper finder without reranking
            self.paper_finder = None
        
        logger.info("✅ Pipeline initialized (Elixir direct synthesis)")
    
    # =========================================================================
    # Step 1: Query Preprocessing
    # =========================================================================
    
    def preprocess_query(self, query: str) -> LLMProcessedQuery:
        """Decompose query and extract filters."""
        logger.info("📝 Step 1: Query Preprocessing")
        result = self.preprocessor.decompose_query(query)
        logger.info(f"   Rewritten: {result.rewritten_query}")
        logger.info(f"   Filters: {result.search_filters}")
        return result
    
    # =========================================================================
    # Step 2: Retrieval
    # =========================================================================
    
    def retrieve_passages(
        self,
        processed_query: LLMProcessedQuery
    ) -> List[Dict[str, Any]]:
        """
        Retrieve passages from Qdrant using multi-query expansion.
        
        Optimized: Runs PMC variations AND DailyMed searches in parallel.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        logger.info("🔍 Step 2: Passage Retrieval (Parallel PMC + DailyMed)")
        
        all_passages = []
        dailymed_results = []
        seen_ids = set()
        
        # Get queries and drug names
        queries_to_run = processed_query.expanded_queries or [processed_query.rewritten_query]
        if processed_query.keyword_query and processed_query.keyword_query not in queries_to_run:
            queries_to_run.append(processed_query.keyword_query)
            
        drug_names = []
        if processed_query.decomposed and processed_query.decomposed.drug_names:
            drug_names = processed_query.decomposed.drug_names
        else:
            # Fallback to manual extraction if decomposition failed
            drug_names = self._extract_drug_names(processed_query.original_query, processed_query.rewritten_query)

        logger.info(f"   PMC Queries: {len(queries_to_run)} | DailyMed Drugs: {len(drug_names)}")
        
        def run_single_pmc_query(query: str) -> List[Dict[str, Any]]:
            try:
                return self.retriever.retrieve_passages(query, **processed_query.search_filters)
            except Exception as e:
                logger.warning(f"PMC query failed: {query[:40]}... - {e}")
                return []

        def run_dailymed_search(drugs: List[str]) -> List[Dict[str, Any]]:
            try:
                if not drugs: return []
                return self.retriever.search_dailymed_by_drug(drugs)
            except Exception as e:
                logger.warning(f"DailyMed search failed: {e}")
                return []

        # Run ALL retrieval tasks in parallel (PMC + DailyMed)
        with ThreadPoolExecutor(max_workers=max(len(queries_to_run) + 1, 5)) as executor:
            # PMC variations
            pmc_futures = {executor.submit(run_single_pmc_query, q): q for q in queries_to_run}
            # DailyMed search
            dm_future = executor.submit(run_dailymed_search, drug_names)
            
            # Wait for DailyMed
            try:
                dailymed_results = dm_future.result()
                if dailymed_results:
                    logger.info(f"   ✅ Parallel DailyMed search found {len(dailymed_results)} results")
            except Exception as e:
                logger.warning(f"DailyMed task failed: {e}")

            # Collect PMC results as they arrive
            for future in as_completed(pmc_futures):
                query = pmc_futures[future]
                try:
                    passages = future.result()
                    for p in passages:
                        pmcid = p.get("pmcid")
                        if pmcid and pmcid not in seen_ids:
                            seen_ids.add(pmcid)
                            all_passages.append(p)
                except Exception as e:
                    logger.warning(f"PMC query thread failed for '{query[:30]}': {e}")
        
        logger.info(f"   Retrieved {len(all_passages)} unique PMC passages")
        return all_passages, dailymed_results

    
    def _extract_drug_names(self, original_query: str, rewritten_query: str) -> List[str]:
        """
        Extract drug names from query using simple pattern matching.
        
        Looks for capitalized words that look like drug names (brand names)
        and known drug name patterns.
        """
        import re
        
        # Combine queries for better coverage
        text = f"{original_query} {rewritten_query}".lower()
        
        drug_names = set()
        
        # Common brand name drugs that are often asked about
        common_drugs = [
            "xeljanz", "tofacitinib", "humira", "adalimumab", "enbrel", "etanercept",
            "remicade", "infliximab", "rituxan", "rituximab", "orencia", "abatacept",
            "actemra", "tocilizumab", "simponi", "golimumab", "cimzia", "certolizumab",
            "rinvoq", "upadacitinib", "olumiant", "baricitinib", "kevzara", "sarilumab",
            "methotrexate", "mtx", "plaquenil", "hydroxychloroquine", "sulfasalazine",
            "leflunomide", "arava", "azathioprine", "imuran", "cyclosporine",
            "prednisone", "prednisolone", "methylprednisolone", "dexamethasone",
            "celebrex", "celecoxib", "meloxicam", "naproxen", "ibuprofen",
            "taltz", "ixekizumab", "cosentyx", "secukinumab", "stelara", "ustekinumab",
            "dupixent", "dupilumab", "otezla", "apremilast", "tremfya", "guselkumab",
        ]
        
        for drug in common_drugs:
            if drug in text:
                drug_names.add(drug)
        
        # Extract capitalized words that might be brand names
        capitalized_words = re.findall(r'\b[A-Z][a-z]{3,}\b', original_query)
        for word in capitalized_words:
            if len(word) >= 4 and word.lower() not in ["what", "when", "where", "which", "this", "that", "with", "from", "have"]:
                drug_names.add(word.lower())
        
        return list(drug_names)[:5]  # Limit to 5 drugs


    
    # =========================================================================
    # Step 3: Reranking & Aggregation
    # =========================================================================
    
    def rerank_and_aggregate(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        dailymed_results: List[Dict[str, Any]] = None
    ):
        """Rerank passages and aggregate to paper level.
        
        Args:
            query: User query
            passages: Main passages (will be reranked)
            dailymed_results: DailyMed drug labels (bypass reranking, merged directly)
        """
        import pandas as pd
        
        logger.info("📊 Step 3: Reranking & Aggregation")

        if self.paper_finder is None:
            # Skip reranking and create basic aggregation
            logger.info("   Skipping reranking (Cohere not available)")
            # Convert passages to the expected format for aggregation
            reranked = passages[:self.retriever.n_retrieval]  # Limit to n_retrieval

            # Create a basic DataFrame-like structure (abstracts only - no full text)
            papers_df = pd.DataFrame([{
                'pmcid': p.get('pmcid', ''),
                'title': p.get('title', ''),
                'authors': p.get('authors', []),
                'venue': p.get('venue', ''),
                'journal': p.get('journal', ''),
                'year': p.get('year'),
                'doi': p.get('doi', ''),
                'article_type': p.get('article_type', ''),
                'relevance_score': p.get('relevance_judgement', 0),
                'abstract': p.get('abstract', ''),
                'corpus_id': p.get('corpus_id', '')
            } for p in reranked])
        else:
            # Optimization: Deduplicate passages at paper level BEFORE reranking
            # This significantly reduces reranker latency and cost
            logger.info(f"   Deduplicating {len(passages)} passages before reranking...")
            unique_passages = []
            seen_paper_ids = set()
            for p in passages:
                pid = p.get("pmcid") or p.get("corpus_id")
                if pid and pid not in seen_paper_ids:
                    seen_paper_ids.add(pid)
                    unique_passages.append(p)
            
            logger.info(f"   Reranking {len(unique_passages)} unique candidate papers")
            reranked = self.paper_finder.rerank(query, unique_passages)

            # Aggregate to paper level and format as DataFrame
            papers_df = self.paper_finder.aggregate_into_dataframe(reranked)

        logger.info(f"   Aggregated to {len(papers_df)} papers")
        
        # Deduplicate DailyMed entries (keep max per drug)
        papers_df = self._deduplicate_dailymed(papers_df)
        
        # Post-retrieval filtering: filter out papers that don't contain key medical entities
        papers_df = self._filter_by_entities(query, papers_df)
        
        # MERGE DailyMed results directly (bypass reranking and thresholds)
        if dailymed_results:
            logger.info(f"   Merging {len(dailymed_results)} DailyMed results (bypassed reranking)")
            existing_pmcids = set(papers_df['pmcid'].tolist()) if not papers_df.empty else set()
            
            # Create rows for DailyMed - give them high relevance score to appear at top
            dailymed_rows = []
            for dm in dailymed_results:
                pmcid = dm.get('pmcid', '')
                if pmcid and pmcid not in existing_pmcids:
                    dailymed_rows.append({
                        'pmcid': pmcid,
                        'title': dm.get('title', ''),
                        'authors': dm.get('authors', []),
                        'venue': dm.get('venue', '') or dm.get('journal', 'DailyMed'),
                        'journal': dm.get('journal', 'DailyMed'),
                        'year': dm.get('year'),
                        'doi': dm.get('doi', ''),
                        'article_type': dm.get('article_type', 'drug_label'),
                        'relevance_score': 0.95,  # High score to appear near top
                        'abstract': dm.get('abstract', ''),
                        'corpus_id': pmcid,
                        'source': 'dailymed',
                        'set_id': dm.get('set_id', ''),
                        # Include all 8 DailyMed sections for intelligent selection
                        'highlights': dm.get('highlights', ''),
                        'indications': dm.get('indications', ''),
                        'dosage': dm.get('dosage', ''),
                        'contraindications': dm.get('contraindications', ''),
                        'warnings': dm.get('warnings', ''),
                        'adverse_reactions': dm.get('adverse_reactions', ''),
                        'clinical_pharmacology': dm.get('clinical_pharmacology', ''),
                        'clinical_studies': dm.get('clinical_studies', ''),
                    })
                    existing_pmcids.add(pmcid)
            
            if dailymed_rows:
                dailymed_df = pd.DataFrame(dailymed_rows)
                papers_df = pd.concat([dailymed_df, papers_df], ignore_index=True)
                logger.info(f"   Total papers after DailyMed merge: {len(papers_df)}")
        
        return papers_df, reranked

    
    def _deduplicate_dailymed(self, papers_df) -> Any:
        """
        Deduplicate DailyMed entries to keep only top entries per drug.
        
        Multiple manufacturers may have the same drug info, so we keep only
        the top MAX_DAILYMED_PER_DRUG entries per normalized drug name.
        
        Args:
            papers_df: DataFrame of papers
            
        Returns:
            Deduplicated DataFrame
        """
        if papers_df.empty:
            return papers_df
        
        import pandas as pd
        import re
        
        logger.info("🔄 Deduplicating DailyMed entries...")
        
        # Track DailyMed entries by normalized drug name
        dailymed_by_drug = {}  # drug_name -> list of (index, relevance_score)
        non_dailymed_indices = []
        
        for idx, row in papers_df.iterrows():
            pmcid = str(row.get('pmcid', '') or row.get('corpus_id', ''))
            article_type = str(row.get('article_type', ''))
            venue = str(row.get('venue', '') or row.get('journal', ''))
            
            is_dailymed = (
                pmcid.startswith('dailymed_') or 
                article_type == 'drug_label' or
                'dailymed' in venue.lower() or
                'fda drug label' in venue.lower()
            )
            
            if is_dailymed:
                # Extract and normalize drug name from title
                title = str(row.get('title', ''))
                # Remove trailing form descriptors (Tablets, Injection, etc.)
                # Use \s+ (one or more whitespace) to ensure form words only match
                # when preceded by drug name, not at the start of the title
                normalized_name = re.sub(
                    r'\s+(?:tablets?|capsules?|injection|solution|oral|intravenous|iv|im|powder|suspension|syrup|cream|ointment|gel|patch|spray)\b.*$',
                    '',
                    title,
                    flags=re.IGNORECASE
                ).strip().lower()
                
                # Further normalize: remove manufacturer info in parentheses
                normalized_name = re.sub(r'\s*\([^)]*\)\s*$', '', normalized_name).strip()
                
                # If normalized name is empty, use pmcid as unique key to avoid
                # incorrectly grouping different drugs with missing/invalid titles
                if not normalized_name:
                    normalized_name = f"__unique_{pmcid}"
                
                relevance = row.get('relevance_judgement', row.get('relevance_score', 0))
                
                if normalized_name not in dailymed_by_drug:
                    dailymed_by_drug[normalized_name] = []
                dailymed_by_drug[normalized_name].append((idx, relevance))
            else:
                non_dailymed_indices.append(idx)
        
        # Keep only top MAX_DAILYMED_PER_DRUG per drug
        kept_dailymed_indices = []
        removed_count = 0
        
        for drug_name, entries in dailymed_by_drug.items():
            # Sort by relevance score (descending)
            sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)
            
            # Keep top entries
            kept = sorted_entries[:MAX_DAILYMED_PER_DRUG]
            removed = sorted_entries[MAX_DAILYMED_PER_DRUG:]
            
            kept_dailymed_indices.extend([idx for idx, _ in kept])
            removed_count += len(removed)
            
            if len(sorted_entries) > MAX_DAILYMED_PER_DRUG:
                logger.info(f"   Drug '{drug_name}': kept {len(kept)}/{len(sorted_entries)} DailyMed entries")
        
        if removed_count > 0:
            logger.info(f"   Removed {removed_count} duplicate DailyMed entries (keeping max {MAX_DAILYMED_PER_DRUG} per drug)")
            
            # Combine non-DailyMed + kept DailyMed indices, then sort to preserve original ranking
            all_kept_indices = sorted(non_dailymed_indices + kept_dailymed_indices)
            papers_df = papers_df.loc[all_kept_indices].reset_index(drop=True)
        else:
            logger.info("   No duplicate DailyMed entries found")
        
        return papers_df
    
    def _filter_by_entities(self, query: str, papers_df) -> Any:
        """
        Filter papers based on medical entity matching.
        
        Removes papers that don't contain key medical entities from the query.
        This helps prevent irrelevant results (e.g., IgG4-RD when query is about APS).
        
        Args:
            query: Original query
            papers_df: DataFrame of papers
            
        Returns:
            Filtered DataFrame
        """
        if papers_df.empty or self.entity_expander is None:
            return papers_df
        
        import pandas as pd
        
        logger.info("🔍 Post-retrieval filtering: Checking entity matches...")
        
        # Extract medical entities from query
        query_entities = self._extract_query_entities(query)
        
        if not query_entities:
            logger.info("   No medical entities found in query, skipping filter")
            return papers_df
        
        logger.info(f"   Query entities: {query_entities}")
        
        # Check each paper for entity matches
        filtered_indices = []
        removed_count = 0
        
        for idx, row in papers_df.iterrows():
            title = str(row.get('title', '')).lower()
            abstract = str(row.get('abstract', '')).lower()
            
            # Combine title and abstract for searching (abstracts only - no full text)
            paper_text = f"{title} {abstract}".lower()
            
            # Check if paper contains at least one key entity
            has_entity = False
            for entity in query_entities:
                entity_lower = entity.lower()
                # Check for exact match or word boundary match
                if (entity_lower in paper_text or 
                    f" {entity_lower} " in paper_text or
                    entity_lower in title):
                    has_entity = True
                    break
            
            if has_entity:
                filtered_indices.append(idx)
            else:
                removed_count += 1
                logger.debug(f"   Removed: {row.get('title', 'Untitled')[:60]}... (no entity match)")
        
        if removed_count > 0:
            logger.info(f"   Filtered out {removed_count} papers without entity matches")
            papers_df = papers_df.loc[filtered_indices].reset_index(drop=True)
        else:
            logger.info("   All papers contain query entities")
        
        return papers_df
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract medical entities (disease names, acronyms) from query.
        
        Args:
            query: Original query string
            
        Returns:
            List of medical entity terms
        """
        entities = []
        
        if self.entity_expander is None:
            return entities
        
        # Extract acronyms and their expansions
        words = query.split()
        for word in words:
            # Remove punctuation
            import re
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check if it's a known acronym
            if self.entity_expander._is_likely_acronym(clean_word):
                expansions = self.entity_expander.expand_acronym(clean_word)
                if expansions:
                    # Add both acronym and full term
                    entities.append(clean_word.upper())
                    entities.append(expansions[0])
        
        # Also look for common medical condition patterns
        # (e.g., "Antiphospholipid Syndrome", "Classification Criteria")
        medical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Syndrome\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Disease\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Disorder\b',
        ]
        
        import re
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
    
    # =========================================================================
    # Step 4: Direct LLM Synthesis
    # =========================================================================
    
    def _get_papers_for_context(self, papers_df, query: str = ""):
        """Build context and get used papers from reranked articles.
        
        For DailyMed articles, intelligently selects sections:
        - Always: highlights + clinical_studies
        - Conditional: clinical_pharmacology (when query mentions mechanisms, PK/PD)
        """
        from .specialty_journals import PRIORITY_JOURNALS
        HIGH_VALUE_TYPES = {'review_article', 'clinical_trial', 'systematic_review', 'meta_analysis', 'guideline'}
        
        priority_journal_papers = []
        other_papers = []

        for idx, row in papers_df.head(MAX_ABSTRACTS).iterrows():
            journal = row.get('journal', '') or row.get('venue', '')
            is_priority_journal = journal in PRIORITY_JOURNALS
            article_type = row.get('article_type', 'other')
            is_high_value = article_type in HIGH_VALUE_TYPES

            if is_priority_journal or is_high_value:
                priority_journal_papers.append((idx, row))
            else:
                other_papers.append((idx, row))

        all_papers = priority_journal_papers + other_papers
        used_papers = []
        context_parts = []
        dailymed_section_count = 0

        for i, (idx, row) in enumerate(all_papers[:MAX_ABSTRACTS]):
            article_type = row.get('article_type', 'other')
            source_num = len(context_parts) + 1
            
            if hasattr(row, 'to_dict'):
                used_papers.append(row.to_dict())
            else:
                used_papers.append(dict(row))

            paper_info = f"**[{source_num}]** {row.get('title', 'Untitled')}"
            if row.get('venue') or row.get('journal'):
                paper_info += f" | *{row.get('venue') or row.get('journal')}*"
            if row.get('year'):
                paper_info += f" ({row.get('year')})"
            paper_info += f" [{article_type}]"

            # DailyMed: Intelligent section selection
            if article_type == "drug_label" or row.get("source") == "dailymed":
                sections = []
                
                # Always include highlights (summary + boxed warning)
                highlights = row.get('highlights', '')
                if highlights:
                    sections.append(f"### Highlights of Prescribing Information\n{highlights[:8000]}")
                
                # Always include clinical studies (efficacy data, trial results)
                clinical_studies = row.get('clinical_studies', '')
                if clinical_studies:
                    sections.append(f"### Clinical Studies\n{clinical_studies[:15000]}")
                
                # Conditionally include clinical pharmacology
                if self._needs_clinical_pharmacology(query):
                    clinical_pharm = row.get('clinical_pharmacology', '')
                    if clinical_pharm:
                        sections.append(f"### Clinical Pharmacology\n{clinical_pharm[:10000]}")
                
                if sections:
                    text_content = "\n\n".join(sections)
                    dailymed_section_count += 1
                else:
                    # Fallback to abstract if no sections available
                    text_content = (row.get('abstract', '') or row.get('text', ''))[:6000]
                
                text_content = self._clean_source_text(text_content)
                paper_info += f"\n{text_content}"
            else:
                # PMC articles: use abstract with 1200 char limit
                text_content = row.get('abstract', '') or row.get('text', '')
                text_content = self._clean_source_text(text_content)
                paper_info += f"\n{text_content[:1200]}"
            
            context_parts.append(paper_info)
            
        logger.info(f"Context built: {len(context_parts)} papers ({dailymed_section_count} DailyMed with section selection).")
        return context_parts, used_papers

    def _needs_clinical_pharmacology(self, query: str) -> bool:
        """Check if query needs clinical pharmacology section (mechanism, PK/PD)."""
        if not query:
            return False
        keywords = [
            'mechanism', 'pharmacokinetic', 'pharmacodynamic', 'metabolism',
            'half-life', 'half life', 'absorption', 'distribution', 'excretion', 
            'clearance', 'pk', 'pd', 'bioavailability', 'drug interaction',
            'cyp', 'enzyme', 'how does', 'how it works', 'mode of action',
            'moa', 'pathway', 'receptor', 'binding'
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in keywords)

    def _clean_source_text(self, text: str) -> str:
        """
        Strip internal citations from source text to prevent LLM confusion.
        Target patterns: [1], [1, 2], [1-5], [1,2,3], etc.
        """
        if not text:
            return ""
        
        # Pattern for [1], [1, 2], [1-5], [1,2,3] etc.
        # Targets brackets containing numbers, commas, spaces, and hyphens/dashes.
        # Includes leading space to avoid "statement [1]." -> "statement ."
        cleaned = re.sub(r'\s*\[[\d\s,\-\–\.]+\]', '', text)
        
        # Also handle potential superscript numbers if they were converted to text like ^1
        cleaned = re.sub(r'\^[\d,]+', '', cleaned)
        
        # Basic cleanup of double spaces resulting from removal
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        
        return cleaned.strip()

    def run_generation(
        self,
        query: str,
        papers_df,
        stream: bool = False
    ):
        """Generate answer directly using LLM with ELIXIR system prompt."""
        logger.info("🧠 Step 4: Direct LLM Synthesis")

        if papers_df.empty:
            return ("No relevant evidence found for your query.", [], [])

        context_parts, used_papers = self._get_papers_for_context(papers_df, query)
        abstract_count = len(context_parts)

        logger.info(f"   Context: {abstract_count} abstracts = {len(context_parts)} total articles")
        context = "\n\n---\n\n".join(context_parts)

        # Use ELIXIR_SYSTEM_PROMPT for comprehensive deep research responses
        system_prompt = ELIXIR_SYSTEM_PROMPT
        logger.info(f"   Mode: Deep Research | Prompt: {len(system_prompt)} chars")

        # Use comprehensive deep research prompt
        user_prompt = f"""
# [QUERY]
{query}

# [CONTEXT]
(Source Literature: {abstract_count} articles)
{context}

# [INSTRUCTIONS]
Analyze and synthesize the medical literature above to create a detailed, clinically-focused clinical review or practice guideline section.
1. **Extract specific clinical details**: Classification systems, staging criteria, detailed medication protocols (dosing, administration, duration), trial results (outcomes, p-values), and guideline recommendations.
2. **Provide comparative analyses**: Efficacy comparisons with data points and safety profiles.
3. **Structure**: Use clear hierarchical headings, markdown tables for comparisons/staging, and evidence-based recommendations.
4. **Citation**: Use inline citations **[1]**, **[2]**, etc. strictly.
5. **Depth**: High technical detail for physician decision-making. No word count limit, but maintain density.
"""

        try:
            if stream:
                # Return a generator for tokens
                return self._stream_generation(query, user_prompt, used_papers)
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"   Generated response ({len(answer)} chars)")
            return answer, used_papers

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return (f"Error generating response: {str(e)}", [])

    def _stream_generation(self, query: str, user_prompt: str, used_papers: list) -> Generator[Dict[str, Any], None, None]:
        """Generator for token-by-token streaming."""
        try:
            # Use ELIXIR_SYSTEM_PROMPT for deep research
            system_prompt = ELIXIR_SYSTEM_PROMPT
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True
            )
            
            full_answer = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_answer += token
                    yield {"step": "generation", "status": "running", "token": token}
            
            yield {"step": "generation", "status": "complete", "answer": full_answer, "used_papers": used_papers}
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield {"step": "error", "message": str(e)}
    
    # =========================================================================
    # PDF Availability Check via Europe PMC
    # =========================================================================
    
    def _check_pdf_availability(self, papers: list) -> list:
        """
        Check PDF availability for papers via Europe PMC API.
        Uses DOI or PMCID to find open access PDFs.
        
        Europe PMC API returns:
        - isOpenAccess: "Y" or "N" (string)
        - inEPMC: "Y" or "N" (string) 
        - hasPDF: "Y" or "N" (string)
        - fullTextUrlList.fullTextUrl: Array with documentStyle, availabilityCode, url
        
        PDF URL format: https://europepmc.org/articles/{PMCID}?pdf=render
        
        Args:
            papers: List of paper dicts with doi, pmcid, pmid fields
            
        Returns:
            List of source dicts with pdf_url field added where available
        """
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def check_single_pdf(paper) -> dict:
            """Check PDF for a single paper via Europe PMC API."""
            source = self._map_paper_to_source(paper)
            
            doi = source.get("doi", "")
            pmcid = source.get("pmcid", "")
            pmid = source.get("pmid", "")
            
            try:
                # Build query - try DOI first, then PMCID, then PMID
                query = None
                if doi:
                    query = f"DOI:{doi}"
                elif pmcid:
                    # Keep PMC prefix if present, add if not
                    if str(pmcid).upper().startswith("PMC"):
                        query = f"PMCID:{pmcid}"
                    else:
                        query = f"PMCID:PMC{pmcid}"
                elif pmid:
                    query = f"EXT_ID:{pmid}"
                    
                if not query:
                    return self._map_paper_to_source(paper)
                
                api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={requests.utils.quote(query)}&format=json&resultType=core&pageSize=1"
                
                response = requests.get(api_url, timeout=10)
                source = self._map_paper_to_source(paper)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("resultList", {}).get("result", [])
                    
                    if results:
                        article = results[0]
                        
                        # Method 1: Check fullTextUrlList for Open Access PDF (most reliable)
                        url_list = article.get("fullTextUrlList", {}).get("fullTextUrl", [])
                        for url_info in url_list:
                            # Look for Open Access PDF
                            doc_style = url_info.get("documentStyle", "").lower()
                            avail_code = url_info.get("availabilityCode", "")
                            availability = url_info.get("availability", "").lower()
                            
                            if doc_style == "pdf" and (avail_code == "OA" or "open access" in availability):
                                source["pdf_url"] = url_info.get("url")
                                logger.debug(f"Found PDF via fullTextUrlList: {source['pdf_url']}")
                                break
                        
                        # Method 2: If no PDF found in fullTextUrlList, check hasPDF flag
                        if not source["pdf_url"]:
                            is_open_access = article.get("isOpenAccess") == "Y"
                            in_epmc = article.get("inEPMC") == "Y"
                            has_pdf = article.get("hasPDF") == "Y"
                            
                            if has_pdf and (is_open_access or in_epmc):
                                article_pmcid = article.get("pmcid")
                                if article_pmcid:
                                    # Correct PDF URL format for Europe PMC
                                    source["pdf_url"] = f"https://europepmc.org/articles/{article_pmcid}?pdf=render"
                                    logger.debug(f"Constructed PDF URL from PMCID: {source['pdf_url']}")
                                    
            except requests.exceptions.Timeout:
                logger.debug(f"PDF check timed out for {doi or pmcid or pmid}")
                source = self._map_paper_to_source(paper)
            except Exception as e:
                logger.debug(f"PDF check failed for {doi or pmcid or pmid}: {e}")
                source = self._map_paper_to_source(paper)
            
            return source

        logger.info("📄 Step 5: Checking PDF availability via Europe PMC")
        
        if not papers:
            return []
        
        sources = []
        # Use ThreadPoolExecutor for parallel PDF checks (max 10 concurrent)
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_paper = {executor.submit(check_single_pdf, paper): paper for paper in papers}
            for future in as_completed(future_to_paper):
                try:
                    source = future.result()
                    sources.append(source)
                except Exception as e:
                    logger.debug(f"PDF check thread failed: {e}")
        
        # Preserve original order (as_completed returns in completion order)
        # Re-sort by matching against original papers list
        paper_order = {(p.get("doi", "") or p.get("pmcid", "") or p.get("corpus_id", "")): i for i, p in enumerate(papers)}
        sources.sort(key=lambda s: paper_order.get(s.get("doi", "") or s.get("pmcid", ""), 999))
        
        # Count and log PDFs found
        pdf_count = sum(1 for s in sources if s.get("pdf_url"))
        logger.info(f"   Found {pdf_count}/{len(sources)} articles with open access PDFs")
        
        return sources

    def _map_paper_to_source(self, paper: Any) -> Dict[str, Any]:
        """
        Helper to map a paper record (Dict or Series) to a standardized source object.
        Used both in initial streaming and final PDF verification.
        """
        # Convert Series/Namespace to dict if needed
        if hasattr(paper, 'to_dict'):
            p = paper.to_dict()
        else:
            p = dict(paper)

        pmcid = p.get("pmcid") or p.get("corpus_id", "")
        source_type = p.get("source", "")
        
        # Detect DailyMed articles
        is_dailymed = (
            source_type == "dailymed" or 
            p.get("article_type") == "drug_label" or
            str(pmcid).startswith("dailymed_")
        )
        
        # Extract set_id for DailyMed articles
        set_id = p.get("set_id", "")
        if is_dailymed and not set_id and str(pmcid).startswith("dailymed_"):
            set_id = pmcid.replace("dailymed_", "")

        # Helper to sanitize NaN values (pandas can return float('nan') which breaks JSON)
        def sanitize(val, default=""):
            import math
            if val is None:
                return default
            if isinstance(val, float) and math.isnan(val):
                return default
            return val

        return {
            "pmcid": pmcid,
            "pmid": sanitize(p.get("pmid"), ""),
            "title": sanitize(p.get("title"), "Untitled"),
            "authors": p.get("authors", []),
            "journal": sanitize(p.get("venue") or p.get("journal"), ""),
            "year": sanitize(p.get("year")),
            "doi": sanitize(p.get("doi"), ""),
            "article_type": sanitize(p.get("article_type"), ""),
            "relevance_score": sanitize(p.get("relevance_judgement", p.get("relevance_score", 0)), 0),
            "pdf_url": sanitize(p.get("pdf_url")), # Preserve if already present
            "source": source_type,
            "set_id": set_id,
            "dailymed_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else None,
        }
    
    # =========================================================================
    # Full Pipeline
    # =========================================================================
    
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Run complete Elixir RAG pipeline.
        
        Returns structured response with answer and sources.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔬 Elixir Pipeline: {query}")
        logger.info(f"{'='*60}")
        
        # Step 1: Preprocess
        processed_query = self.preprocess_query(query)
        
        # Step 2: Retrieve
        passages, dailymed_results = self.retrieve_passages(processed_query)
        
        # Apply hybrid scoring (dense + keyword matching)
        if passages:
            passages = self.retriever.apply_hybrid_scoring(
                query=processed_query.rewritten_query,
                passages=passages,
                dense_weight=0.7,
                sparse_weight=0.3
            )
        
        if not passages:
            return {
                "query": query,
                "report_title": "No Results Found",
                "answer": "No relevant papers found for your query.",
                "sections": [],
                "sources": [],
                "status": "no_results"
            }
        
        # Step 3: Rerank & Aggregate
        papers_df, reranked_passages = self.rerank_and_aggregate(query, passages, dailymed_results)
        
        if papers_df.empty:
            return {
                "query": query,
                "report_title": "No Results After Reranking",
                "answer": "Papers did not meet relevance threshold.",
                "sections": [],
                "sources": [],
                "status": "no_results"
            }
        
        # Step 4: Direct LLM Synthesis
        result = self.run_generation(query, papers_df)
        
        # Handle tuple return (answer, used_papers) or error string
        if isinstance(result, tuple):
            if len(result) == 2:
                answer, used_papers = result
            else:
                answer = result[0]
                used_papers = []
        else:
            # Error case
            answer = result
            used_papers = []
        
        # Step 5: Check PDF availability via Europe PMC for all used papers
        sources_with_pdf = self._check_pdf_availability(used_papers)
        
        return {
            "query": query,
            "report_title": "Clinical Response",
            "answer": answer,
            "sections": [],  # Direct synthesis - no sections
            "sources": sources_with_pdf,
            "retrieval_stats": {
                "passages_retrieved": len(passages),
                "papers_after_aggregation": len(papers_df),
                "abstracts_used": len(used_papers)
            },
            "status": "success"
        }
    
    def answer_streaming(
        self,
        query: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Streaming version for real-time UI updates with parallel PDF checks."""
        from concurrent.futures import ThreadPoolExecutor

        # Step 1: Preprocess
        yield {"step": "query_expansion", "status": "running", "message": "Analyzing query..."}
        processed_query = self.preprocess_query(query)
        yield {
            "step": "query_expansion",
            "status": "complete",
            "data": {"rewritten": processed_query.rewritten_query}
        }

        # Step 2: Retrieval
        yield {"step": "retrieval", "status": "running", "message": "Searching literature..."}
        passages, dailymed_results = self.retrieve_passages(processed_query)
        yield {
            "step": "retrieval",
            "status": "complete",
            "data": {"count": len(passages) + len(dailymed_results)}
        }

        if not passages and not dailymed_results:
            yield {"step": "complete", "status": "no_results"}
            return

        # Step 3: Reranking
        yield {"step": "reranking", "status": "running", "message": "Ranking papers..."}
        papers_df, _ = self.rerank_and_aggregate(query, passages, dailymed_results)
        
        # [NEW] Reorder papers so reference ranking matches final citation order from the start
        # This prevents the UI from "jumping" and ensures references are correctly numbered
        _, used_papers = self._get_papers_for_context(papers_df, query)
        
        # Prepare initial sources (no PDF URLs yet)
        initial_sources = []
        for p in used_papers:
            initial_sources.append(self._map_paper_to_source(p))

        yield {
            "step": "reranking",
            "status": "complete",
            "data": {"papers": len(papers_df)},
            "sources": initial_sources  # Send sources early and in corrected order!
        }
        
        if papers_df.empty:
            yield {"step": "complete", "status": "no_results"}
            return

        # Prepare context for generation
        _, used_papers = self._get_papers_for_context(papers_df, query)

        # START PARALLEL TASKS: Generation + PDF Check
        yield {"step": "pdf_check", "status": "running", "message": "Checking PDF availability..."}
        yield {"step": "generation", "status": "running", "message": "Synthesizing response..."}
        
        pdf_results = []
        pdf_yielded = False
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Start PDF check in background
            pdf_future = executor.submit(self._check_pdf_availability, used_papers)
            
            # Start streaming generation
            answer = ""
            generation_gen = self.run_generation(query, papers_df, stream=True)
            for event in generation_gen:
                if event["step"] == "generation" and event["status"] == "running":
                    if "token" in event:
                        yield event
                    
                    # [NEW] Check if PDF future is done during token synthesis
                    # Yield PDF links AS SOON AS AVAILABLE instead of waiting for end
                    if not pdf_yielded and pdf_future.done():
                        try:
                            pdf_results = pdf_future.result()
                            pdf_count = sum(1 for s in pdf_results if s.get("pdf_url"))
                            yield {
                                "step": "pdf_check", 
                                "status": "complete", 
                                "data": {"pdf_count": pdf_count},
                                "sources": pdf_results # UPDATE SOURCES WITH PDFs NOW
                            }
                            pdf_yielded = True
                            logger.info(f"   ✅ PDF check yielded early ({pdf_count} PDFs found)")
                        except Exception as e:
                            logger.error(f"Error yielding PDF results early: {e}")

                elif event["step"] == "generation" and event["status"] == "complete":
                    answer = event["answer"]
            
            # Final safety check if not already yielded
            if not pdf_results:
                pdf_results = pdf_future.result()

        if not pdf_yielded:
            yield {"step": "pdf_check", "status": "complete", "data": {"pdf_count": sum(1 for s in pdf_results if s.get("pdf_url"))}, "sources": pdf_results}
        yield {"step": "generation", "status": "complete"}
        
        # NO FILTERING: Show all sources that were provided as context
        # Use initial_sources as base to ensure we never return empty list if PDFs check fails
        final_sources = pdf_results if pdf_results else initial_sources
        
        logger.info(f"🔍 Streaming Complete: Returning {len(final_sources)} sources")

        # Final event
        yield {
            "step": "complete",
            "status": "success",
            "report_title": "Clinical Response",
            "answer": answer,
            "sources": final_sources,
            "abstracts_used": len(used_papers),
            "original_sources_count": len(pdf_results)
        }
    
    # Backward compatibility
    def answer_scholarqa_style(self, query: str) -> Dict[str, Any]:
        """Alias for answer()."""
        return self.answer(query)


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Testing Elixir Medical RAG Pipeline")
    print("=" * 70)
    
    pipeline = MedicalRAGPipeline()
    result = pipeline.answer("management of copd")
    
    print(f"\n📋 Report: {result['report_title']}")
    print(f"📚 Sources: {len(result['sources'])}")
    print(f"\n{result['answer'][:2000]}...")
