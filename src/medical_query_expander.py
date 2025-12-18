"""
Medical Query Expander using MeSH synonyms.

Expands medical queries with synonyms and related terms from MeSH
to improve retrieval recall while maintaining precision.
"""

import logging
from typing import List, Set
import re

from .medical_entity_expander import MedicalEntityExpander

logger = logging.getLogger(__name__)


class MedicalQueryExpander:
    """
    Expands medical queries with synonyms and related terms.
    
    Uses MeSH data to generate query variations for better retrieval.
    """
    
    def __init__(self):
        """Initialize medical query expander."""
        try:
            self.entity_expander = MedicalEntityExpander()
            logger.info("✅ Medical query expander initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize medical query expander: {e}")
            self.entity_expander = None
    
    def expand_query(self, query: str, max_variations: int = 5) -> List[str]:
        """
        Generate multiple query variations with medical synonyms.
        
        Args:
            query: Original query
            max_variations: Maximum number of variations to generate
            
        Returns:
            List of query variations (including original)
        """
        if not self.entity_expander:
            return [query]
        
        variations = [query]  # Always include original
        
        # Extract medical terms from query
        medical_terms = self._extract_medical_terms(query)
        
        if not medical_terms:
            return variations
        
        # Generate variations by replacing terms with synonyms
        for term in medical_terms[:3]:  # Limit to first 3 terms to avoid explosion
            synonyms = self.entity_expander.get_synonyms(term)
            if synonyms:
                # Create variation by replacing term with synonym
                for synonym in synonyms[:2]:  # Limit synonyms per term
                    if synonym.lower() != term.lower():
                        variation = query.replace(term, synonym, 1)
                        if variation not in variations:
                            variations.append(variation)
                            if len(variations) >= max_variations:
                                break
                if len(variations) >= max_variations:
                    break
        
        # Also add variations with expanded acronyms
        words = query.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if self.entity_expander._is_likely_acronym(clean_word):
                expansions = self.entity_expander.expand_acronym(clean_word)
                if expansions:
                    # Replace acronym with full term
                    variation = query.replace(word, expansions[0], 1)
                    if variation not in variations:
                        variations.append(variation)
                        if len(variations) >= max_variations:
                            break
        
        logger.info(f"Generated {len(variations)} query variations")
        return variations[:max_variations]
    
    def _extract_medical_terms(self, query: str) -> List[str]:
        """
        Extract medical terms from query.
        
        Args:
            query: Search query
            
        Returns:
            List of medical terms
        """
        terms = []
        
        if not self.entity_expander:
            return terms
        
        # Look for known medical terms (case-insensitive)
        query_lower = query.lower()
        
        # Check against synonym dictionary
        for term in self.entity_expander.synonym_dict.keys():
            if term.lower() in query_lower:
                terms.append(term)
        
        # Also extract acronyms
        words = query.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if self.entity_expander._is_likely_acronym(clean_word):
                if clean_word in self.entity_expander.acronym_dict:
                    terms.append(clean_word)
        
        return terms
    
    def expand_with_related_terms(self, query: str) -> str:
        """
        Expand query with related medical terms.
        
        Adds related terms like "classification criteria" → "diagnostic criteria"
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        # Common medical term expansions
        term_expansions = {
            "classification criteria": ["diagnostic criteria", "classification system"],
            "treatment": ["therapy", "management", "intervention"],
            "diagnosis": ["diagnostic", "detection", "identification"],
            "symptoms": ["clinical features", "manifestations", "presentation"],
            "pathophysiology": ["pathogenesis", "mechanism", "etiology"],
        }
        
        expanded_query = query
        for term, expansions in term_expansions.items():
            if term.lower() in query.lower():
                # Add first expansion
                expanded_query = f"{expanded_query} {expansions[0]}"
                break
        
        return expanded_query


if __name__ == "__main__":
    # Test the expander
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Medical Query Expander")
    print("=" * 60)
    
    expander = MedicalQueryExpander()
    
    test_queries = [
        "2023 ACR/EULAR APS Classification Criteria",
        "HFpEF treatment guidelines",
        "SGLT2 inhibitors for diabetes"
    ]
    
    for query in test_queries:
        print(f"\n📝 Original: {query}")
        variations = expander.expand_query(query, max_variations=5)
        print(f"✅ Variations ({len(variations)}):")
        for i, var in enumerate(variations, 1):
            print(f"   {i}. {var}")

