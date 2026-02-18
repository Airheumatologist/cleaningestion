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
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Optional, Set, Tuple
import requests

logger = logging.getLogger(__name__)

# MeSH download locations
MESH_XML_BASE_URL = os.getenv(
    "MESH_XML_BASE_URL",
    "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh",
)
MESH_DOWNLOAD_PAGE = os.getenv(
    "MESH_DOWNLOAD_PAGE",
    "https://www.nlm.nih.gov/databases/download/mesh.html",
)

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

        # Resolve MeSH source URLs (env override > discovered latest > fallback year)
        self.mesh_desc_url, self.mesh_supp_url = self._resolve_mesh_urls()
        
        # File paths
        desc_name = Path(urlparse(self.mesh_desc_url).path).name or "desc.xml"
        supp_name = Path(urlparse(self.mesh_supp_url).path).name or "supp.xml"
        self.desc_xml_path = self.data_dir / desc_name
        self.supp_xml_path = self.data_dir / supp_name
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
    
    def _resolve_mesh_urls(self) -> Tuple[str, str]:
        """Resolve MeSH descriptor/supplementary URLs with env overrides and discovery."""
        env_desc = os.getenv("MESH_DESC_URL", "").strip()
        env_supp = os.getenv("MESH_SUPP_URL", "").strip()
        if env_desc and env_supp:
            return env_desc, env_supp

        mesh_year = os.getenv("MESH_YEAR", "").strip()
        if mesh_year:
            return (
                f"{MESH_XML_BASE_URL}/desc{mesh_year}.xml",
                f"{MESH_XML_BASE_URL}/supp{mesh_year}.xml",
            )

        discovered_desc, discovered_supp = self._discover_mesh_urls()
        if discovered_desc and discovered_supp:
            return discovered_desc, discovered_supp

        fallback_year = datetime.now().year - 1
        logger.warning(
            "Could not discover latest MeSH URLs from %s, falling back to year %s",
            MESH_DOWNLOAD_PAGE,
            fallback_year,
        )
        return (
            f"{MESH_XML_BASE_URL}/desc{fallback_year}.xml",
            f"{MESH_XML_BASE_URL}/supp{fallback_year}.xml",
        )

    def _discover_mesh_urls(self) -> Tuple[Optional[str], Optional[str]]:
        """Discover latest MeSH XML file names from the NLM MeSH download page."""
        try:
            response = requests.get(MESH_DOWNLOAD_PAGE, timeout=60)
            response.raise_for_status()
            html = response.text
            desc_matches = re.findall(r"desc(\d{4})\.xml", html, flags=re.IGNORECASE)
            supp_matches = re.findall(r"supp(\d{4})\.xml", html, flags=re.IGNORECASE)
            desc_year = max((int(y) for y in desc_matches), default=None)
            supp_year = max((int(y) for y in supp_matches), default=None)
            if desc_year and supp_year:
                return (
                    f"{MESH_XML_BASE_URL}/desc{desc_year}.xml",
                    f"{MESH_XML_BASE_URL}/supp{supp_year}.xml",
                )
        except Exception as exc:
            logger.warning("MeSH URL discovery failed: %s", exc)
        return None, None

    def _ensure_mesh_files(self):
        """Ensure both MeSH XML files exist and are valid XML with expected root tags."""
        self._ensure_valid_mesh_file(
            self.desc_xml_path,
            self.mesh_desc_url,
            expected_root_tag="DescriptorRecordSet",
        )
        self._ensure_valid_mesh_file(
            self.supp_xml_path,
            self.mesh_supp_url,
            expected_root_tag="SupplementalRecordSet",
        )

    def _ensure_valid_mesh_file(self, local_path: Path, source_url: str, expected_root_tag: str) -> None:
        """Validate local cache and download/retry once when invalid or missing."""
        if local_path.exists():
            if self._is_valid_mesh_xml(local_path, expected_root_tag):
                logger.info("MeSH file exists and validated: %s", local_path)
                return
            logger.warning("Invalid cached MeSH file detected, removing: %s", local_path)
            try:
                local_path.unlink()
            except FileNotFoundError:
                pass

        for attempt in range(2):
            try:
                logger.info("Downloading MeSH file from %s...", source_url)
                self._download_file(source_url, local_path, expected_root_tag)
                return
            except Exception as exc:
                try:
                    local_path.unlink()
                except FileNotFoundError:
                    pass
                if attempt == 0:
                    logger.warning("Download validation failed for %s, retrying once: %s", local_path.name, exc)
                    continue
                raise RuntimeError(
                    f"Unable to download valid MeSH XML file '{local_path.name}' from {source_url}. "
                    f"See {MESH_DOWNLOAD_PAGE} for updated URLs."
                ) from exc

    def _is_valid_mesh_xml(self, path: Path, expected_root_tag: str) -> bool:
        """Check XML parseability and expected root tag."""
        try:
            for _, elem in ET.iterparse(path, events=("start",)):
                root_tag = elem.tag.split("}", 1)[-1] if isinstance(elem.tag, str) else str(elem.tag)
                if root_tag != expected_root_tag:
                    logger.error(
                        "Invalid MeSH XML root in %s: expected '%s', got '%s'",
                        path,
                        expected_root_tag,
                        root_tag,
                    )
                    return False
                return True
            logger.error("MeSH XML file is empty: %s", path)
            return False
        except ET.ParseError as exc:
            logger.error("Invalid XML in %s: %s", path, exc)
            return False
        except Exception as exc:
            logger.error("Failed validating MeSH file %s: %s", path, exc)
            return False

    def _download_file(self, url: str, dest_path: Path, expected_root_tag: str):
        """Download a file from URL, validate XML content, then atomically replace target."""
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "xml" not in content_type:
                raise ValueError(
                    f"Non-XML content type '{content_type or 'unknown'}' received from {url}"
                )

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(tmp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                logger.info(f"Downloaded {downloaded / (1024*1024):.1f} MB ({percent:.1f}%)")
            
            if not self._is_valid_mesh_xml(tmp_path, expected_root_tag):
                raise ValueError(
                    f"Downloaded file {tmp_path.name} is not a valid MeSH XML ({expected_root_tag})"
                )
            tmp_path.replace(dest_path)
            logger.info(f"✅ Downloaded and validated {dest_path.name} ({downloaded / (1024*1024):.1f} MB)")
        except Exception as e:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            logger.error(f"Failed to download {url}: {e}")
            raise
    
    def _load_mesh_data(self):
        """Parse MeSH XML files and build lookup dictionaries."""
        if not self.desc_xml_path.exists() or not self.supp_xml_path.exists():
            raise FileNotFoundError(
                "MeSH XML files are missing. "
                f"Expected: {self.desc_xml_path} and {self.supp_xml_path}"
            )
        if not self._is_valid_mesh_xml(self.desc_xml_path, "DescriptorRecordSet"):
            raise ValueError(f"Descriptor MeSH XML is invalid: {self.desc_xml_path}")
        if not self._is_valid_mesh_xml(self.supp_xml_path, "SupplementalRecordSet"):
            raise ValueError(f"Supplementary MeSH XML is invalid: {self.supp_xml_path}")

        # Parse descriptors (main terms)
        self._parse_descriptors(self.desc_xml_path)
        
        # Parse supplementary concepts (drugs, chemicals)
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
                            self.acronym_dict[term].append(preferred_term)
                
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
