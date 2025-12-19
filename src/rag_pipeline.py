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
        context_threshold: float = 0.0
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
        """Retrieve passages from Qdrant."""
        logger.info("🔍 Step 2: Passage Retrieval")
        
        all_passages = []
        seen_ids = set()
        
        # Primary retrieval with rewritten query
        passages = self.retriever.retrieve_passages(
            processed_query.rewritten_query,
            **processed_query.search_filters
        )
        
        for p in passages:
            pmcid = p.get("pmcid")
            if pmcid not in seen_ids:
                seen_ids.add(pmcid)
                all_passages.append(p)
        
        # Additional retrieval with keyword query
        if processed_query.keyword_query:
            keyword_passages = self.retriever.retrieve_additional_papers(
                processed_query.keyword_query,
                **processed_query.search_filters
            )
            for p in keyword_passages:
                pmcid = p.get("pmcid")
                if pmcid not in seen_ids:
                    seen_ids.add(pmcid)
                    all_passages.append(p)
        
        logger.info(f"   Retrieved {len(all_passages)} passages")
        return all_passages
    
    # =========================================================================
    # Step 3: Reranking & Aggregation
    # =========================================================================
    
    def rerank_and_aggregate(
        self,
        query: str,
        passages: List[Dict[str, Any]]
    ):
        """Rerank passages and aggregate to paper level."""
        logger.info("📊 Step 3: Reranking & Aggregation")

        if self.paper_finder is None:
            # Skip reranking and create basic aggregation
            logger.info("   Skipping reranking (Cohere not available)")
            # Convert passages to the expected format for aggregation
            reranked = passages[:self.retriever.n_retrieval]  # Limit to n_retrieval

            # Create a basic DataFrame-like structure (abstracts only - no full text)
            import pandas as pd
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
            # Rerank all passages
            reranked = self.paper_finder.rerank(query, passages)

            # Aggregate to paper level and format as DataFrame
            papers_df = self.paper_finder.aggregate_into_dataframe(reranked)

        logger.info(f"   Aggregated to {len(papers_df)} papers")
        
        # Deduplicate DailyMed entries (keep max per drug)
        papers_df = self._deduplicate_dailymed(papers_df)
        
        # Post-retrieval filtering: filter out papers that don't contain key medical entities
        papers_df = self._filter_by_entities(query, papers_df)
        
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
    
    def run_generation(
        self,
        query: str,
        papers_df
    ):
        """Generate answer directly using LLM with ELIXIR system prompt."""
        logger.info("🧠 Step 4: Direct LLM Synthesis")

        if papers_df.empty:
            return ("No relevant evidence found for your query.", [], [])

        # Context configuration: Abstracts only (no full text)
        from .specialty_journals import PRIORITY_JOURNALS

        HIGH_VALUE_TYPES = {'review_article', 'clinical_trial', 'systematic_review', 'meta_analysis', 'guideline'}
        ABSTRACT_LIMIT = 1200  # Chars per abstract (increased since no full text)

        # Build context from reranked papers with journal prioritization
        context_parts = []
        abstract_count = 0

        # First pass: prioritize high-value articles from priority journals
        priority_journal_papers = []
        other_papers = []

        for idx, row in papers_df.head(MAX_ABSTRACTS).iterrows():
            corpus_id = row.get('corpus_id', '')
            journal = row.get('journal', '') or row.get('venue', '')
            is_priority_journal = journal in PRIORITY_JOURNALS
            article_type = row.get('article_type', 'other')
            is_high_value = article_type in HIGH_VALUE_TYPES

            if is_priority_journal or is_high_value:
                priority_journal_papers.append((idx, row))
            else:
                other_papers.append((idx, row))

        # Combine: priority journals first, then others
        all_papers = priority_journal_papers + other_papers

        used_papers = []  # Track order for reference generation

        for idx, row in all_papers[:MAX_ABSTRACTS]:
            article_type = row.get('article_type', 'other')
            source_num = len(context_parts) + 1

            # Track for reference section - convert pandas Series to dict for JSON serialization
            if hasattr(row, 'to_dict'):
                used_papers.append(row.to_dict())
            else:
                used_papers.append(dict(row))

            # Build paper header with metadata
            paper_info = f"**[{source_num}]** {row.get('title', 'Untitled')}"
            if row.get('venue') or row.get('journal'):
                paper_info += f" | *{row.get('venue') or row.get('journal')}*"
            if row.get('year'):
                paper_info += f" ({row.get('year')})"
            paper_info += f" [{article_type}]"

            # Use abstract only (no full text)
            abstract = row.get('abstract', '') or row.get('text', '')
            paper_info += f"\n{abstract[:ABSTRACT_LIMIT]}"
            abstract_count += 1

            context_parts.append(paper_info)

        logger.info(f"   Context: {abstract_count} abstracts = {len(context_parts)} total articles")
        context = "\n\n---\n\n".join(context_parts)

        user_prompt = f"""Analyze and synthesize the following medical literature to create a detailed, clinically-focused response for healthcare professionals.

**Clinical Query**: {query}

**Instructions**:
1. **Extract specific clinical details** from the abstracts below:
   - Classification/staging systems (e.g., Hurley staging, TNM staging) with criteria for each level
   - Specific medications with brand/generic names, dosing protocols, administration routes, and durations
   - Surgical approaches and procedural techniques with specific indications
   - Trial results with specific outcomes, p-values, and clinical significance
   - Latest guideline recommendations and regulatory updates

2. **Provide comparative analyses** when multiple treatment options exist:
   - Efficacy comparisons with specific data points
   - Safety profiles and contraindications
   - Clinical scenarios where one approach is preferred

3. **Structure your response** with:
   - Clear section headings for different aspects (pathogenesis, diagnosis, staging, treatment options, etc.)
   - Markdown tables for staging systems, treatment comparisons, and medication protocols
   - Evidence-based recommendations with specific citations to the source articles

4. **Depth requirement**: This should read like a detailed clinical review or practice guideline section, with sufficient technical detail for physician decision-making.

5. **Citation requirement**: Use inline citations **[1]**, **[2]**, etc. to cite specific articles that support each claim.

**Source Literature** ({abstract_count} abstracts):
{context}

Generate a comprehensive, evidence-based clinical response that synthesizes findings across the {abstract_count} abstracts provided. Cross-reference multiple sources to identify consensus and note any conflicting evidence."""

        try:
            response = self.openai_client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": ELIXIR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                text={
                    "format": {
                        "type": "text"
                    },
                    "verbosity": "medium"
                },
                reasoning={
                    "effort": "medium"
                },
                store=False
            )

            answer = response.output_text.strip()

            logger.info(f"   Generated response ({len(answer)} chars)")
            # Return answer and used papers (no full_text_articles tracking)
            return answer, used_papers

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return (f"Error generating response: {str(e)}", [])
    
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
            pmcid = paper.get("pmcid") or paper.get("corpus_id", "")
            source_type = paper.get("source", "")
            
            # Detect DailyMed articles
            is_dailymed = (
                source_type == "dailymed" or 
                paper.get("article_type") == "drug_label" or
                str(pmcid).startswith("dailymed_")
            )
            
            # Extract set_id for DailyMed articles
            set_id = None
            if is_dailymed:
                set_id = paper.get("set_id", "")
                if not set_id and str(pmcid).startswith("dailymed_"):
                    set_id = pmcid.replace("dailymed_", "")
            
            source = {
                "pmcid": pmcid,
                "pmid": paper.get("pmid", ""),
                "title": paper.get("title", "Untitled"),
                "authors": paper.get("authors", []),
                "journal": paper.get("venue") or paper.get("journal", ""),
                "year": paper.get("year"),
                "doi": paper.get("doi", ""),
                "article_type": paper.get("article_type", ""),
                "relevance_score": paper.get("relevance_judgement", 0),
                "pdf_url": None,
                # DailyMed-specific fields
                "source": source_type,
                "set_id": set_id,
                "dailymed_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else None,
            }
            
            doi = paper.get("doi", "")
            pmcid = paper.get("pmcid") or paper.get("corpus_id", "")
            pmid = paper.get("pmid", "")
            
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
                    return source
                
                api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={requests.utils.quote(query)}&format=json&resultType=core&pageSize=1"
                
                response = requests.get(api_url, timeout=10)
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
            except Exception as e:
                logger.debug(f"PDF check failed for {doi or pmcid or pmid}: {e}")
            
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
        passages = self.retrieve_passages(processed_query)
        
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
        papers_df, reranked_passages = self.rerank_and_aggregate(query, passages)
        
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
        """Streaming version for real-time UI updates."""

        # Step 1
        yield {"step": "query_expansion", "status": "running", "message": "Analyzing query..."}
        processed_query = self.preprocess_query(query)
        yield {
            "step": "query_expansion",
            "status": "complete",
            "data": {"rewritten": processed_query.rewritten_query}
        }

        # Step 2
        yield {"step": "retrieval", "status": "running", "message": "Searching literature..."}
        passages = self.retrieve_passages(processed_query)
        yield {
            "step": "retrieval",
            "status": "complete",
            "data": {"count": len(passages)}
        }

        if not passages:
            yield {"step": "complete", "status": "no_results"}
            return

        # Step 3
        yield {"step": "reranking", "status": "running", "message": "Ranking papers..."}
        papers_df, _ = self.rerank_and_aggregate(query, passages)
        yield {
            "step": "reranking",
            "status": "complete",
            "data": {"papers": len(papers_df)}
        }

        # Step 4: Direct LLM Synthesis
        yield {"step": "generating", "status": "running", "message": "Synthesizing response..."}
        result = self.run_generation(query, papers_df)
        
        # Handle tuple return (answer, used_papers) or error string
        if isinstance(result, tuple):
            if len(result) == 2:
                answer, used_papers = result
            else:
                answer = result[0]
                used_papers = []
        else:
            answer = result
            used_papers = []
        
        # Ensure answer is always a string
        if answer is None:
            answer = "Error: No response generated."
        elif not isinstance(answer, str):
            answer = str(answer)
        
        logger.info(f"Generated answer length: {len(answer)} chars")
            
        yield {
            "step": "generating",
            "status": "complete"
        }
        
        # Step 5: Check PDF availability
        yield {"step": "pdf_check", "status": "running", "message": "Checking PDF availability..."}
        sources_with_pdf = self._check_pdf_availability(used_papers)
        yield {
            "step": "pdf_check",
            "status": "complete",
            "data": {"pdf_count": sum(1 for s in sources_with_pdf if s.get("pdf_url"))}
        }
        
        # Final - include full source metadata with PDF URLs
        final_event = {
            "step": "complete",
            "status": "success",
            "report_title": "Clinical Response",
            "answer": answer,
            "sections": [],
            "sources": sources_with_pdf,
            "abstracts_used": len(used_papers)
        }
        logger.info(f"Final event: answer present={bool(final_event.get('answer'))}, answer_length={len(answer)}")
        yield final_event
    
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
