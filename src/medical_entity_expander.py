"""
Medical Entity Expander using MeSH (Medical Subject Headings).

Downloads and parses MeSH XML data to expand medical acronyms and synonyms.
Integrates with query preprocessing to improve retrieval accuracy.

Example:
    expander = MedicalEntityExpander()
    expanded = expander.expand_query("2023 ACR/EULAR APS Classification Criteria")
    # Returns: "2023 ACR/EULAR Antiphospholipid Syndrome APS Classification Criteria"
"""

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import requests

logger = logging.getLogger(__name__)

# MeSH download URLs
MESH_DESC_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml"
MESH_SUPP_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/supp2025.xml"
MESH_DOWNLOAD_PAGE = "https://www.nlm.nih.gov/databases/download/mesh.html"

# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "mesh"


class MedicalEntityExpander:
    """
    Expands medical acronyms and synonyms using MeSH dataset.
    
    Downloads MeSH XML files, parses them, and builds lookup dictionaries
    for fast acronym expansion and synonym mapping.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        auto_download: bool = True,
        cache_enabled: bool = True
    ):
        """
        Initialize medical entity expander.
        
        Args:
            data_dir: Directory to store MeSH XML files (default: data/mesh/)
            auto_download: If True, download MeSH files if not present
            cache_enabled: If True, cache parsed dictionaries to disk
        """
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        
        # Lookup dictionaries
        self.acronym_dict: Dict[str, List[str]] = {}  # {"APS": ["Antiphospholipid Syndrome", ...]}
        self.synonym_dict: Dict[str, List[str]] = {}  # {"Antiphospholipid Syndrome": ["APS", "APLS", ...]}
        self.full_term_dict: Dict[str, str] = {}  # {"APS": "Antiphospholipid Syndrome"} (primary mapping)
        
        # File paths
        self.desc_xml_path = self.data_dir / "desc2025.xml"
        self.supp_xml_path = self.data_dir / "supp2025.xml"
        self.cache_path = self.data_dir / "mesh_cache.json"
        
        # Load or build dictionaries
        if auto_download:
            self._ensure_mesh_files()
        
        if cache_enabled and self.cache_path.exists():
            logger.info("Loading MeSH data from cache...")
            self._load_from_cache()
        else:
            logger.info("Parsing MeSH XML files...")
            self._load_mesh_data()
            if cache_enabled:
                self._save_to_cache()
        
        logger.info(f"Loaded {len(self.acronym_dict)} acronym mappings and {len(self.synonym_dict)} synonym groups")
    
    def _ensure_mesh_files(self):
        """Download MeSH XML files if they don't exist."""
        if not self.desc_xml_path.exists():
            logger.info(f"Downloading MeSH descriptors from {MESH_DESC_URL}...")
            self._download_file(MESH_DESC_URL, self.desc_xml_path)
        else:
            logger.info(f"MeSH descriptors file exists: {self.desc_xml_path}")
        
        if not self.supp_xml_path.exists():
            logger.info(f"Downloading MeSH supplementary concepts from {MESH_SUPP_URL}...")
            self._download_file(MESH_SUPP_URL, self.supp_xml_path)
        else:
            logger.info(f"MeSH supplementary concepts file exists: {self.supp_xml_path}")
    
    def _download_file(self, url: str, dest_path: Path):
        """Download a file from URL to destination path."""
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                logger.info(f"Downloaded {downloaded / (1024*1024):.1f} MB ({percent:.1f}%)")
            
            logger.info(f"✅ Downloaded {dest_path.name} ({downloaded / (1024*1024):.1f} MB)")
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise
    
    def _load_mesh_data(self):
        """Parse MeSH XML files and build lookup dictionaries."""
        # Parse descriptors (main terms)
        if self.desc_xml_path.exists():
            self._parse_descriptors(self.desc_xml_path)
        
        # Parse supplementary concepts (drugs, chemicals)
        if self.supp_xml_path.exists():
            self._parse_supplementary(self.supp_xml_path)
        
        # Post-process: identify primary mappings for acronyms
        self._build_primary_mappings()
    
    def _parse_descriptors(self, xml_path: Path):
        """Parse MeSH descriptor XML file."""
        logger.info(f"Parsing descriptors from {xml_path.name}...")
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            count = 0
            for descriptor in root.findall('.//DescriptorRecord'):
                descriptor_name_elem = descriptor.find('.//DescriptorName/String')
                if descriptor_name_elem is None:
                    continue
                
                preferred_term = descriptor_name_elem.text.strip()
                if not preferred_term:
                    continue
                
                # Extract all terms (preferred + entry terms/synonyms)
                all_terms = [preferred_term]
                
                # Get entry terms (synonyms, acronyms)
                for concept in descriptor.findall('.//Concept'):
                    for term in concept.findall('.//Term'):
                        term_string = term.find('String')
                        if term_string is not None and term_string.text:
                            term_text = term_string.text.strip()
                            if term_text and term_text != preferred_term:
                                all_terms.append(term_text)
                
                # Build mappings
                for term in all_terms:
                    # Normalize term
                    normalized = self._normalize_term(term)
                    
                    # Add to synonym dictionary
                    if preferred_term not in self.synonym_dict:
                        self.synonym_dict[preferred_term] = []
                    
                    if normalized not in self.synonym_dict[preferred_term]:
                        self.synonym_dict[preferred_term].append(normalized)
                    
                    # Check if term is an acronym (short, uppercase, or mixed case)
                    if self._is_likely_acronym(term):
                        if term not in self.acronym_dict:
                            self.acronym_dict[term] = []
                        if preferred_term not in self.acronym_dict[term]:
                            self.acronym_dict[term].append(preferred_term)
                
                count += 1
                if count % 1000 == 0:
                    logger.debug(f"Processed {count} descriptors...")
            
            logger.info(f"✅ Parsed {count} descriptors")
        
        except Exception as e:
            logger.error(f"Error parsing descriptors: {e}")
            raise
    
    def _parse_supplementary(self, xml_path: Path):
        """Parse MeSH supplementary concept XML file."""
        logger.info(f"Parsing supplementary concepts from {xml_path.name}...")
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            count = 0
            for record in root.findall('.//SupplementalRecord'):
                name_elem = record.find('.//Name/String')
                if name_elem is None:
                    continue
                
                preferred_term = name_elem.text.strip()
                if not preferred_term:
                    continue
                
                # Extract all terms
                all_terms = [preferred_term]
                
                for concept in record.findall('.//Concept'):
                    for term in concept.findall('.//Term'):
                        term_string = term.find('String')
                        if term_string is not None and term_string.text:
                            term_text = term_string.text.strip()
                            if term_text and term_text != preferred_term:
                                all_terms.append(term_text)
                
                # Build mappings (similar to descriptors)
                for term in all_terms:
                    normalized = self._normalize_term(term)
                    
                    if preferred_term not in self.synonym_dict:
                        self.synonym_dict[preferred_term] = []
                    
                    if normalized not in self.synonym_dict[preferred_term]:
                        self.synonym_dict[preferred_term].append(normalized)
                    
                    if self._is_likely_acronym(term):
                        if term not in self.acronym_dict:
                            self.acronym_dict[term] = []
                        if preferred_term not in self.acronym_dict[term]:
                            self.acronym_dict[preferred_term].append(preferred_term)
                
                count += 1
                if count % 1000 == 0:
                    logger.debug(f"Processed {count} supplementary concepts...")
            
            logger.info(f"✅ Parsed {count} supplementary concepts")
        
        except Exception as e:
            logger.error(f"Error parsing supplementary concepts: {e}")
            raise
    
    def _normalize_term(self, term: str) -> str:
        """Normalize medical term for comparison."""
        # Remove extra whitespace, convert to title case
        return ' '.join(term.strip().split())
    
    # Organization acronyms that should NOT be treated as disease entities
    # These are medical societies/organizations, not conditions
    ORGANIZATION_ACRONYMS = {
        "ACR", "EULAR", "AHA", "ACC", "ESC", "WHO", "CDC", "FDA", "NIH",
        "AAN", "AMA", "AAFP", "AAP", "ASCO", "ESMO", "BSR", "OARSI",
        "GRAPPA", "SPARTAN", "ASAS", "PANLAR", "APLAR", "ILAR"
    }
    
    def _is_likely_acronym(self, term: str) -> bool:
        """
        Check if a term is likely an acronym.
        
        Criteria:
        - Short (2-10 characters)
        - Mostly uppercase or mixed case
        - Not a common word
        - NOT an organization acronym (ACR, EULAR, etc.)
        """
        term = term.strip()
        
        # Too long to be an acronym
        if len(term) > 10 or len(term) < 2:
            return False
        
        # Exclude organization acronyms - these are NOT disease entities
        if term.upper() in self.ORGANIZATION_ACRONYMS:
            return False
        
        # Check if mostly uppercase or has mixed case pattern
        upper_count = sum(1 for c in term if c.isupper())
        if upper_count >= len(term) * 0.5:  # At least 50% uppercase
            return True
        
        # Check for common acronym patterns (e.g., "HFpEF", "SGLT2")
        if re.match(r'^[A-Z][a-z]*[A-Z]', term):  # Mixed case like "HFpEF"
            return True
        
        return False
    
    def _build_primary_mappings(self):
        """Build primary mapping dictionary (acronym -> most common full term)."""
        for acronym, full_terms in self.acronym_dict.items():
            if full_terms:
                # Use the first term as primary (usually most common)
                self.full_term_dict[acronym] = full_terms[0]
    
    def _load_from_cache(self):
        """Load dictionaries from JSON cache file."""
        import json
        try:
            with open(self.cache_path, 'r') as f:
                cache_data = json.load(f)
                self.acronym_dict = cache_data.get('acronym_dict', {})
                self.synonym_dict = cache_data.get('synonym_dict', {})
                self.full_term_dict = cache_data.get('full_term_dict', {})
            logger.info("✅ Loaded from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, will rebuild")
            self._load_mesh_data()
    
    def _save_to_cache(self):
        """Save dictionaries to JSON cache file."""
        import json
        try:
            cache_data = {
                'acronym_dict': self.acronym_dict,
                'synonym_dict': self.synonym_dict,
                'full_term_dict': self.full_term_dict
            }
            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"✅ Cached MeSH data to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def expand_acronym(self, acronym: str, context: str = "") -> List[str]:
        """
        Expand a medical acronym to full terms.
        
        Args:
            acronym: The acronym to expand (e.g., "APS")
            context: Optional context to help disambiguation
            
        Returns:
            List of full terms (e.g., ["Antiphospholipid Syndrome"])
        """
        # Normalize acronym
        acronym_upper = acronym.upper().strip()
        acronym_original = acronym.strip()
        
        # Try exact match first
        if acronym_upper in self.acronym_dict:
            return self.acronym_dict[acronym_upper]
        if acronym_original in self.acronym_dict:
            return self.acronym_dict[acronym_original]
        
        # Try case-insensitive match
        for key, values in self.acronym_dict.items():
            if key.upper() == acronym_upper:
                return values
        
        return []
    
    def get_synonyms(self, term: str) -> List[str]:
        """
        Get synonyms for a medical term.
        
        Args:
            term: The medical term (e.g., "Antiphospholipid Syndrome")
            
        Returns:
            List of synonyms including acronyms
        """
        normalized = self._normalize_term(term)
        
        # Try exact match
        if normalized in self.synonym_dict:
            return self.synonym_dict[normalized]
        
        # Try case-insensitive match
        for key, synonyms in self.synonym_dict.items():
            if key.lower() == normalized.lower():
                return synonyms
        
        return []
    
    def expand_query(self, query: str, preserve_original: bool = True) -> str:
        """
        Expand medical acronyms in a query.
        
        Args:
            query: Original query string
            preserve_original: If True, keep original acronym alongside expansion
            
        Returns:
            Expanded query with full terms added
        """
        # Find potential acronyms in query (2-10 chars, mostly uppercase)
        words = query.split()
        expanded_words = []
        seen_expansions = set()
        
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check if it's an acronym
            if self._is_likely_acronym(clean_word):
                expansions = self.expand_acronym(clean_word)
                if expansions:
                    primary = expansions[0]
                    # Avoid duplicate expansions
                    if primary.lower() not in seen_expansions:
                        if preserve_original:
                            expanded_words.append(f"{primary} {word}")
                        else:
                            expanded_words.append(primary)
                        seen_expansions.add(primary.lower())
                    else:
                        expanded_words.append(word)
                else:
                    expanded_words.append(word)
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def extract_medical_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract medical entities (acronyms and full terms) from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity, type) tuples where type is 'acronym' or 'full_term'
        """
        entities = []
        words = text.split()
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check if it's a known acronym
            if clean_word in self.acronym_dict:
                entities.append((clean_word, 'acronym'))
            
            # Check if it's a known full term (case-insensitive)
            for term in self.synonym_dict.keys():
                if term.lower() in text.lower():
                    entities.append((term, 'full_term'))
        
        return entities


if __name__ == "__main__":
    # Test the expander
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Medical Entity Expander")
    print("=" * 60)
    
    expander = MedicalEntityExpander()
    
    test_queries = [
        "2023 ACR/EULAR APS Classification Criteria",
        "HFpEF treatment guidelines",
        "SGLT2 inhibitors for diabetes",
        "COPD management strategies"
    ]
    
    for query in test_queries:
        print(f"\n📝 Original: {query}")
        expanded = expander.expand_query(query)
        print(f"✅ Expanded: {expanded}")
        
        # Show acronym expansions
        words = query.split()
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if expander._is_likely_acronym(clean):
                expansions = expander.expand_acronym(clean)
                if expansions:
                    print(f"   '{clean}' → {expansions[0]}")

