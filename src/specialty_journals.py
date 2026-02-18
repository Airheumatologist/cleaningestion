"""
High-impact and specialty journal configuration for prioritization.
These journals should be prioritized when selecting full-text articles.
"""

import re

# High-impact specialty journals by corpus_id
PRIORITY_JOURNALS = {
    # General/Multi-specialty High-Impact
    'S202381698',  # JAMA
    'S25405668',   # New England Journal of Medicine
    'S201448712',  # The Lancet
    'S203492740',  # BMJ
    'S144348140',  # JAMA Internal Medicine
    'S144024400',  # Annals of Internal Medicine
    'S2764825513', # PLOS Medicine
    
    # Cardiology
    'S121635830',  # Circulation
    'S114681167',  # Journal of the American College of Cardiology
    'S190648732',  # European Heart Journal
    'S2738706341', # JACC: Heart Failure
    'S140574793',  # Reviews in Cardiovascular Medicine
    'S2737969411', # Journal of Clinical Medicine
    'S9665008',    # Cardiology Journal
    'S200634258',  # Heart Rhythm
    
    # Emergency Medicine
    'S157616481',  # Emergency Medicine Clinics of North America
    'S32518312',   # Emergency Medicine International
    'S185351110',  # Annals of Emergency Medicine
    'S93457541',   # Academic Emergency Medicine
    
    # Rheumatology
    'S103824965',  # Arthritis & Rheumatology
    'S126230440',  # Annals of the Rheumatic Diseases
    'S46381573',   # Arthritis Research & Therapy
    'S2738672850', # Reumatismo
    'S30446230',   # Modern Rheumatology
    'S140083062',  # The Journal of Rheumatology
    'S198819644',  # Clinical Rheumatology
    'S70834189',   # Rheumatology (Oxford)
    
    # Pulmonology/Respiratory
    'S186153782',  # Respiratory Research
    'S48327234',   # American Journal of Respiratory and Critical Care Medicine
    'S52076854',   # European Respiratory Journal
    'S120248934',  # Chest
    'S196398362',  # Thorax
    
    # Nephrology
    'S2764444402', # Journal of the American Society of Nephrology
    'S144345290',  # Kidney International
    'S511174864',  # American Journal of Kidney Diseases
    'S106398135',  # Clinical Journal of the American Society of Nephrology
    
    # Hematology
    'S4210163874', # Blood
    'S166058934',  # British Journal of Haematology
    'S29890226',   # Haematologica
    'S2764906777', # Blood Advances
    
    # Neurology
    'S5976376',    # Neurology
    'S71901143',   # JAMA Neurology
    'S106397687',  # Annals of Neurology
    
    # Gastroenterology
    'S2734342573', # Gastroenterology
    'S94149373',   # Gut
    'S144163547',  # Clinical Gastroenterology and Hepatology
    'S33205428',   # Alimentary Pharmacology & Therapeutics
    
    # Endocrinology
    'S71460136',   # Diabetes Care
    'S56817854',   # The Journal of Clinical Endocrinology & Metabolism
    'S149348210',  # Diabetologia
    
    # Infectious Disease
    'S126995023',  # Clinical Infectious Diseases
    'S66735914',   # The Journal of Infectious Diseases
    'S161283334',  # Antimicrobial Agents and Chemotherapy
    
    # Oncology
    'S126459015',  # Journal of Clinical Oncology
    'S69976923',   # Cancer
    'S50903633',   # JAMA Oncology
    
    # Critical Care
    'S98454732',   # Critical Care Medicine
    'S88967825',   # Intensive Care Medicine
    'S4210215253', # Critical Care
    
    # Dermatology
    'S88668795',   # Journal of the American Academy of Dermatology
    'S51864137',   # JAMA Dermatology
    'S88313795',   # British Journal of Dermatology
    'S107072492',  # Journal of Investigative Dermatology
    'S2764808689', # Dermatologic Surgery
    
    # Ophthalmology
    'S84629270',   # Ophthalmology
    'S197928149',  # JAMA Ophthalmology
    'S45568642',   # American Journal of Ophthalmology
    'S110746788',  # British Journal of Ophthalmology
    'S135425670',  # Retina
    
    # Otolaryngology (ENT)
    'S43315258',   # Laryngoscope
    'S159990311',  # Otolaryngology-Head and Neck Surgery
    'S47411214',   # JAMA Otolaryngology-Head & Neck Surgery
    'S90048766',   # International Forum of Allergy & Rhinology
    
    # Orthopedics
    'S62825610',   # Journal of Bone and Joint Surgery
    'S90823577',   # Clinical Orthopaedics and Related Research
    'S133574344',  # Journal of Orthopaedic Research
    
    # Psychiatry
    'S83542699',   # American Journal of Psychiatry
    'S142977851',  # Journal of Clinical Psychiatry
    
    # Radiology
    'S35838559',   # Radiology
    'S82092968',   # American Journal of Roentgenology
    'S157835675',  # European Radiology
    'S26737130',   # Journal of Vascular and Interventional Radiology
    
    # Pathology
    'S119021622',  # American Journal of Surgical Pathology
    'S59890115',   # Modern Pathology
    
    # Anesthesiology
    'S4306400717', # Anesthesiology
    'S142815776',  # Anesthesia & Analgesia
    'S6253907',    # British Journal of Anaesthesia
    
    # General Surgery
    'S53269642',   # Annals of Surgery
    'S144196989',  # British Journal of Surgery
    
    # Pediatrics
    'S56642866',   # Pediatrics
    'S144345290',  # Journal of Pediatrics
    
    # Obstetrics & Gynecology
    'S155074832',  # Obstetrics & Gynecology
    'S95746013',   # American Journal of Obstetrics & Gynecology
    
    # Urology
    'S148827164',  # Journal of Urology
    
    # Plastic Surgery
    'S168648959',  # Plastic and Reconstructive Surgery
    
    # Radiation Oncology
    'S129055443',  # International Journal of Radiation Oncology
    
    # Nuclear Medicine
    'S73746427',   # Journal of Nuclear Medicine
}

# Normalized journal-name aliases (derived from the canonical priority list above).
# This allows runtime matching by journal title as well as corpus/document identifiers.
_PRIORITY_JOURNAL_NAME_TEXT = """
JAMA
New England Journal of Medicine
The Lancet
BMJ
JAMA Internal Medicine
Annals of Internal Medicine
PLOS Medicine
Circulation
Journal of the American College of Cardiology
European Heart Journal
JACC: Heart Failure
Reviews in Cardiovascular Medicine
Journal of Clinical Medicine
Cardiology Journal
Heart Rhythm
Emergency Medicine Clinics of North America
Emergency Medicine International
Annals of Emergency Medicine
Academic Emergency Medicine
Arthritis & Rheumatology
Annals of the Rheumatic Diseases
Arthritis Research & Therapy
Reumatismo
Modern Rheumatology
The Journal of Rheumatology
Clinical Rheumatology
Rheumatology (Oxford)
Respiratory Research
American Journal of Respiratory and Critical Care Medicine
European Respiratory Journal
Chest
Thorax
Journal of the American Society of Nephrology
Kidney International
American Journal of Kidney Diseases
Clinical Journal of the American Society of Nephrology
Blood
British Journal of Haematology
Haematologica
Blood Advances
Neurology
JAMA Neurology
Annals of Neurology
Gastroenterology
Gut
Clinical Gastroenterology and Hepatology
Alimentary Pharmacology & Therapeutics
Diabetes Care
The Journal of Clinical Endocrinology & Metabolism
Diabetologia
Clinical Infectious Diseases
The Journal of Infectious Diseases
Antimicrobial Agents and Chemotherapy
Journal of Clinical Oncology
Cancer
JAMA Oncology
Critical Care Medicine
Intensive Care Medicine
Critical Care
Journal of the American Academy of Dermatology
JAMA Dermatology
British Journal of Dermatology
Journal of Investigative Dermatology
Dermatologic Surgery
Ophthalmology
JAMA Ophthalmology
American Journal of Ophthalmology
British Journal of Ophthalmology
Retina
Laryngoscope
Otolaryngology-Head and Neck Surgery
JAMA Otolaryngology-Head & Neck Surgery
International Forum of Allergy & Rhinology
Journal of Bone and Joint Surgery
Clinical Orthopaedics and Related Research
Journal of Orthopaedic Research
American Journal of Psychiatry
Journal of Clinical Psychiatry
Radiology
American Journal of Roentgenology
European Radiology
Journal of Vascular and Interventional Radiology
American Journal of Surgical Pathology
Modern Pathology
Anesthesiology
Anesthesia & Analgesia
British Journal of Anaesthesia
Annals of Surgery
British Journal of Surgery
Pediatrics
Journal of Pediatrics
Obstetrics & Gynecology
American Journal of Obstetrics & Gynecology
Journal of Urology
Plastic and Reconstructive Surgery
International Journal of Radiation Oncology
Journal of Nuclear Medicine
"""

PRIORITY_JOURNAL_NAMES = {
    re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    for name in _PRIORITY_JOURNAL_NAME_TEXT.splitlines()
    if name.strip()
}

# Export includes both corpus/document IDs and normalized journal names.
PRIORITY_JOURNALS = PRIORITY_JOURNALS | PRIORITY_JOURNAL_NAMES

# Guideline organizations (for future use in filtering/prioritization)
GUIDELINE_ORGANIZATIONS = {
    'ACC', 'AHA', 'ESC', 'HRS', 'ACCP',
    'ACP', 'ACR', 'EULAR', 'IDSA', 'ATS', 'ERS',
    'ASCO', 'NCCN', 'ESMO', 'AAD', 'AAO', 'AAO-HNS',
    'AAOS', 'APA', 'RSNA', 'CAP', 'ASA', 'ACS',
    'AAP', 'ACOG', 'AAFP', 'AAPMR', 'AUA', 'ASPS',
    'ASTRO', 'SNMMI', 'ACG', 'AGA', 'ADA', 'AACE',
    'AAN', 'ASN', 'KDIGO', 'ASH', 'ACEP', 'SCCM', 'USPSTF'
}
