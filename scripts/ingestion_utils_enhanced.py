"""
Enhanced utilities for medical document ingestion.

Additional features:
- Semantic chunking with boundary detection
- Figure and formula extraction
- MeSH term hierarchy indexing
- Content deduplication
- Quality scoring
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import lxml.etree as ET
from ingestion_utils import get_text, get_text_excluding_children

logger = logging.getLogger(__name__)


@dataclass
class ChunkQuality:
    """Quality metrics for a chunk."""
    token_count: int
    sentence_count: int
    has_semantic_boundary: bool
    starts_with_header: bool
    ends_with_punctuation: bool
    medical_entity_count: int
    score: float


class SemanticChunker:
    """
    Semantic chunking that respects document structure and content boundaries.
    
    Features:
    - Sentence-aware splitting (preserves complete sentences)
    - Section boundary detection
    - Semantic coherence scoring
    - Medical entity-aware chunking
    """
    
    # Medical entity patterns for quality scoring
    MEDICAL_PATTERNS = [
        r'\b(?:diagnosis|treatment|patient|clinical|symptom|disease|syndrome)\b',
        r'\b(?:mg|mcg|ml|kg|mmHg|°C|mmol/L|U/L)\b',  # Units
        r'\b(?:p\s*[<>=]\s*0?\.\d+|95%\s*CI|OR\s*[=:]\s*\d+\.?\d*|HR\s*[=:]\s*\d+\.?\d*)\b',  # Stats
        r'\b(?:FDA|EMA|NICE|WHO|CDC)\b',  # Regulatory
        r'\b(?:phase\s+[I II III IV]+|randomized|placebo-controlled|double-blind)\b',  # Trial types
    ]
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        # Import here to avoid circular dependency issues
        try:
            from config_ingestion import IngestionConfig
            self.chunk_size = chunk_size if chunk_size is not None else IngestionConfig.CHUNK_SIZE_TOKENS
            self.overlap = overlap if overlap is not None else IngestionConfig.CHUNK_OVERLAP_TOKENS
        except ImportError:
            # Fallback defaults if config not available
            self.chunk_size = chunk_size if chunk_size is not None else 2048
            self.overlap = overlap if overlap is not None else 256
        self.min_chunk_size = 100  # Minimum tokens for a valid chunk
        self.tokenizer = None
        self._load_tokenizer()
        
    def _load_tokenizer(self):
        """Load tokenizer from ingestion_utils if available."""
        try:
            # Try to import and use the shared tokenizer from ingestion_utils
            from ingestion_utils import Chunker as BaseChunker
            base_chunker = BaseChunker(self.chunk_size, self.overlap)
            if base_chunker.tokenizer is not None:
                self.tokenizer = base_chunker.tokenizer
                logger.debug("Loaded tokenizer from ingestion_utils")
        except Exception:
            # Silently fail - will use word-based approximation
            pass
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving abbreviations."""
        # Protect common abbreviations
        protected = text
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'St.', 'Ave.', 'Blvd.',
                        'i.e.', 'e.g.', 'vs.', 'vol.', 'ed.', 'et al.', 'etc.', 'Fig.',
                        'No.', 'vol.', 'pp.', 'ca.', 'inc.']
        
        for i, abbrev in enumerate(abbreviations):
            protected = protected.replace(abbrev, f'<<ABBREV{i}>>')
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', protected)
        
        # Restore abbreviations
        for i, abbrev in enumerate(abbreviations):
            sentences = [s.replace(f'<<ABBREV{i}>>', abbrev) for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer if available, otherwise approximate."""
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception:
                pass  # Fall through to approximation
        
        # Fallback: Simple approximation (1 token ≈ 0.75 words)
        words = len(text.split())
        return int(words / 0.75)
    
    def _count_medical_entities(self, text: str) -> int:
        """Count medical entities in text."""
        count = 0
        for pattern in self.MEDICAL_PATTERNS:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    def _find_semantic_boundary(self, sentences: List[str], target_pos: int) -> int:
        """
        Find the best boundary near target position.
        Prefers boundaries after complete sentences with medical content.
        """
        if not sentences:
            return 0
        
        # Cumulative token counts
        cum_tokens = []
        total = 0
        for sent in sentences:
            total += self._count_tokens(sent)
            cum_tokens.append(total)
        
        # Find the sentence boundary closest to target
        best_idx = 0
        best_diff = abs(cum_tokens[0] - target_pos)
        
        for i, cum in enumerate(cum_tokens):
            diff = abs(cum - target_pos)
            if diff < best_diff:
                # Prefer boundaries with medical content in the sentence
                medical_score = self._count_medical_entities(sentences[i])
                if medical_score > 0:
                    diff = diff * 0.8  # Boost medical-rich boundaries
                best_diff = diff
                best_idx = i
        
        return best_idx + 1  # Return sentence count, not index
    
    def chunk_text(self, text: str, header: str = "") -> List[Dict[str, Any]]:
        """
        Create semantic chunks from text.
        
        This is an alias for chunk_text_semantic for drop-in compatibility
        with the base Chunker class.
        
        Args:
            text: The text to chunk
            header: Optional header to prepend to each chunk
            
        Returns:
            List of chunks with quality metrics
        """
        return self.chunk_text_semantic(text, header)
    
    def chunk_text_semantic(self, text: str, header: str = "") -> List[Dict[str, Any]]:
        """
        Create semantic chunks from text.
        
        Args:
            text: The text to chunk
            header: Optional header to prepend to each chunk
            
        Returns:
            List of chunks with quality metrics
        """
        if not text:
            return []
        
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        header_tokens = self._count_tokens(header) if header else 0
        available_tokens = self.chunk_size - header_tokens
        
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)
            
            # Check if adding this sentence would exceed limit
            if current_tokens + sent_tokens > available_tokens and current_chunk_sentences:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk_sentences)
                if header:
                    chunk_text = header + chunk_text
                
                chunks.append(self._create_chunk_data(chunk_text, current_chunk_sentences))
                
                # Start new chunk with overlap
                overlap_tokens = 0
                overlap_sentences = []
                for prev_sent in reversed(current_chunk_sentences):
                    prev_tokens = self._count_tokens(prev_sent)
                    if overlap_tokens + prev_tokens <= self.overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_tokens += prev_tokens
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sent_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sent_tokens
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            if header:
                chunk_text = header + chunk_text
            chunks.append(self._create_chunk_data(chunk_text, current_chunk_sentences))
        
        return chunks
    
    def _create_chunk_data(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """Create chunk data with quality metrics."""
        token_count = self._count_tokens(text)
        medical_count = self._count_medical_entities(text)
        
        # Quality score calculation
        score = 1.0
        
        # Penalize very short chunks
        if token_count < self.min_chunk_size:
            score *= 0.5
        
        # Boost chunks with medical entities
        score += min(medical_count * 0.05, 0.3)
        
        # Check semantic boundaries
        starts_with_header = bool(re.match(r'^(?:Title|Section|Table|Figure)', text))
        ends_with_punctuation = text.rstrip()[-1] in '.!?' if text else False
        
        if starts_with_header:
            score += 0.1
        if ends_with_punctuation:
            score += 0.1
        
        return {
            "text": text,
            "token_count": token_count,
            "sentence_count": len(sentences),
            "quality_score": min(score, 1.5),
            "medical_entity_count": medical_count,
            "metadata": {
                "starts_with_header": starts_with_header,
                "ends_with_punctuation": ends_with_punctuation,
            }
        }


class FigureExtractor:
    """Extract figures and their captions from PMC XML."""
    
    @staticmethod
    def extract_figures(root: ET.Element) -> List[Dict[str, Any]]:
        """Extract figures from PMC XML."""
        figures = []
        
        for i, fig in enumerate(root.xpath(".//fig"), 1):
            # Get figure ID
            fig_id = fig.get("id", f"fig-{i}")
            
            # Extract caption
            caption_elem = fig.xpath(".//caption")
            caption = get_text(caption_elem[0]) if caption_elem else ""
            
            # Extract label
            label_elem = fig.xpath(".//label")
            label = get_text(label_elem[0]) if label_elem else f"Figure {i}"
            
            # Check for graphic (image)
            graphic = fig.xpath(".//graphic")
            has_graphic = len(graphic) > 0
            
            # Get graphic href if available
            graphic_href = ""
            if graphic:
                graphic_href = graphic[0].get("{http://www.w3.org/1999/xlink}href", "")
            
            figures.append({
                "id": fig_id,
                "label": label,
                "caption": caption,
                "has_graphic": has_graphic,
                "graphic_href": graphic_href,
                "type": "figure"
            })
        
        return figures


class FormulaExtractor:
    """Extract mathematical formulas and equations."""
    
    @staticmethod
    def extract_formulas(root: ET.Element) -> List[Dict[str, Any]]:
        """Extract formulas from PMC XML."""
        formulas = []
        
        # Inline formulas
        for i, formula in enumerate(root.xpath(".//inline-formula"), 1):
            formula_text = get_text(formula)
            if formula_text:
                formulas.append({
                    "id": formula.get("id", f"formula-inline-{i}"),
                    "text": formula_text,
                    "type": "inline",
                    "context": "inline"
                })
        
        # Display formulas
        for i, formula in enumerate(root.xpath(".//disp-formula"), 1):
            formula_text = get_text(formula)
            label_elem = formula.xpath(".//label")
            label = get_text(label_elem[0]) if label_elem else f"({i})"
            
            if formula_text:
                formulas.append({
                    "id": formula.get("id", f"formula-disp-{i}"),
                    "text": formula_text,
                    "label": label,
                    "type": "display",
                    "context": "display"
                })
        
        return formulas


class MeSHProcessor:
    """Process and enhance MeSH term metadata."""
    
    # Common MeSH tree top-level categories
    MESH_CATEGORIES = {
        'A': 'Anatomy',
        'B': 'Organisms',
        'C': 'Diseases',
        'D': 'Chemicals and Drugs',
        'E': 'Analytical, Diagnostic and Therapeutic Techniques',
        'F': 'Psychiatry and Psychology',
        'G': 'Phenomena and Processes',
        'H': 'Disciplines and Occupations',
        'I': 'Anthropology, Education, Sociology',
        'J': 'Technology, Industry, Agriculture',
        'K': 'Humanities',
        'L': 'Information Science',
        'M': 'Named Groups',
        'N': 'Health Care',
        'V': 'Publication Characteristics',
        'Z': 'Geographicals'
    }
    
    @classmethod
    def categorize_mesh_terms(cls, mesh_terms: List[str]) -> Dict[str, List[str]]:
        """
        Categorize MeSH terms by their tree categories.
        Note: This is a simplified version. Full implementation would require
        the MeSH tree structure database.
        """
        categories = {cat: [] for cat in cls.MESH_CATEGORIES.values()}
        
        # Keyword-based categorization (simplified)
        disease_keywords = ['disease', 'syndrome', 'disorder', 'cancer', 'tumor', 'infection',
                          'inflammation', 'failure', 'deficiency', 'pathology']
        drug_keywords = ['drug', 'antibiotic', 'inhibitor', 'agonist', 'antagonist',
                        'therapy', 'treatment', 'medication']
        anatomy_keywords = ['organ', 'tissue', 'cell', 'membrane', 'muscle', 'bone',
                          'brain', 'heart', 'liver', 'kidney']
        
        for term in mesh_terms:
            term_lower = term.lower()
            
            if any(kw in term_lower for kw in disease_keywords):
                categories['Diseases'].append(term)
            elif any(kw in term_lower for kw in drug_keywords):
                categories['Chemicals and Drugs'].append(term)
            elif any(kw in term_lower for kw in anatomy_keywords):
                categories['Anatomy'].append(term)
            else:
                # Default to Health Care for uncategorized
                categories['Health Care'].append(term)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    @classmethod
    def extract_major_topics(cls, mesh_terms: List[str]) -> List[str]:
        """
        Identify major topics (marked with * in MeSH).
        In XML, this might be indicated differently.
        """
        # For now, return all terms as major topics
        # Full implementation would parse MeSH qualifiers
        return mesh_terms


class ContentDeduplicator:
    """Detect and handle duplicate content during ingestion."""
    
    def __init__(self):
        self.seen_hashes: Set[str] = set()
    
    def compute_content_hash(self, text: str, metadata: Dict[str, Any]) -> str:
        """Compute a hash for content deduplication."""
        # Normalize text for hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Include key metadata in hash
        doc_id = metadata.get('doc_id', '')
        chunk_id = metadata.get('chunk_id', '')
        
        content = f"{doc_id}:{chunk_id}:{normalized[:500]}"  # First 500 chars
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def is_duplicate(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Check if content is a duplicate."""
        content_hash = self.compute_content_hash(text, metadata)
        if content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(content_hash)
        return False
    
    def reset(self):
        """Clear deduplication cache."""
        self.seen_hashes.clear()


class QualityValidator:
    """Validate chunk quality before ingestion."""
    
    # Minimum thresholds
    MIN_TOKEN_COUNT = 50
    MIN_SENTENCE_COUNT = 2
    MIN_MEDICAL_ENTITIES = 0  # Optional
    
    # Quality flags
    LOW_QUALITY_PATTERNS = [
        r'^\s*\d+\s*$',  # Just a number
        r'^\s*(Figure|Table)\s*\d+\s*[.:]?\s*$',  # Just figure/table reference
        r'^\s*\[?\d+\]?\s*$',  # Just a citation
    ]
    
    @classmethod
    def validate_chunk(cls, chunk_text: str, metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a chunk and return (is_valid, list_of_issues).
        """
        issues = []
        
        # Check minimum length
        token_count = len(chunk_text.split())
        if token_count < cls.MIN_TOKEN_COUNT:
            issues.append(f"Too short: {token_count} tokens (min {cls.MIN_TOKEN_COUNT})")
        
        # Check for low-quality patterns
        for pattern in cls.LOW_QUALITY_PATTERNS:
            if re.match(pattern, chunk_text, re.IGNORECASE):
                issues.append("Low-quality content pattern detected")
                break
        
        # Check for required fields
        if not metadata.get('doc_id'):
            issues.append("Missing doc_id")
        if not metadata.get('chunk_id'):
            issues.append("Missing chunk_id")
        
        # Check for page_content (critical for retriever)
        if not chunk_text or len(chunk_text) < 50:
            issues.append("Empty or very short page_content")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @classmethod
    def compute_quality_score(cls, chunk_text: str, metadata: Dict[str, Any]) -> float:
        """
        Compute a quality score for the chunk.
        Returns a score between 0 and 1.
        """
        score = 1.0
        
        # Length scoring (optimal: 1500-2048 tokens for Qwen3-Embedding-0.6B)
        # Updated for 2048 token chunks - CHUNK_SIZE_TOKENS from .env
        token_count = len(chunk_text.split())
        if token_count < 50:
            score *= 0.3
        elif token_count < 100:
            score *= 0.6
        elif token_count > 2500:  # Allow some headroom above 2048
            score *= 0.7
        elif 1500 <= token_count <= 2048:
            score *= 1.2  # Boost for optimal size (matches CHUNK_SIZE_TOKENS)
        
        # Medical content scoring
        medical_patterns = [
            r'\b(?:patient|patients|clinical|treatment|therapy|diagnosis)\b',
            r'\b(?:study|trial|control|group|outcome|results)\b',
            r'\b(?:mg|ml|kg|mcg|mmol|μg)\b',
            r'\b(?:significant|p\s*<\s*0\.\d+|CI\s*95%)\b',
        ]
        medical_matches = sum(len(re.findall(p, chunk_text, re.I)) for p in medical_patterns)
        score += min(medical_matches * 0.02, 0.3)  # Max 0.3 boost
        
        # Completeness scoring
        if metadata.get('title'):
            score += 0.05
        if metadata.get('authors'):
            score += 0.05
        if metadata.get('year'):
            score += 0.05
        if metadata.get('abstract'):
            score += 0.05
        
        return min(score, 1.5)


def enhance_pmc_parsing(xml_path, base_article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance parsed PMC article with additional extractions.
    
    Args:
        xml_path: Path to XML file
        base_article: Base article dict from parse_pmc_xml
        
    Returns:
        Enhanced article dict with figures, formulas, categorized MeSH terms
    """
    if not base_article:
        return base_article
    
    try:
        if xml_path.name.endswith(".xml.gz"):
            import gzip
            with gzip.open(xml_path, "rb") as f:
                root = ET.fromstring(f.read())
        else:
            root = ET.parse(str(xml_path)).getroot()
        
        # Extract figures
        figures = FigureExtractor.extract_figures(root)
        if figures:
            base_article["figures"] = figures
            base_article["figure_count"] = len(figures)
        
        # Extract formulas
        formulas = FormulaExtractor.extract_formulas(root)
        if formulas:
            base_article["formulas"] = formulas
        
        # Categorize MeSH terms
        mesh_terms = base_article.get("mesh_terms", [])
        if mesh_terms:
            categorized = MeSHProcessor.categorize_mesh_terms(mesh_terms)
            base_article["mesh_categories"] = categorized
        
        return base_article
        
    except Exception as e:
        logger.warning(f"Failed to enhance article parsing: {e}")
        return base_article
