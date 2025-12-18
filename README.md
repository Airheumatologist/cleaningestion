# Medical RAG Pipeline

Production-ready ingestion pipeline for building a medical literature RAG (Retrieval-Augmented Generation) system using PubMed Central and DailyMed data.

## 📊 Pipeline Overview

| Component | Description |
|-----------|-------------|
| **Data Sources** | PMC Open Access (1.2M+ articles), DailyMed (51K+ drug labels) |
| **Vector Database** | Qdrant Cloud with Cloud Inference |
| **Embedding Model** | mixedbread-ai/mxbai-embed-large-v1 (1024-d) |
| **Quantization** | Binary quantization for memory efficiency |

## 🚀 Quick Start

### Prerequisites

- AWS EC2 instance (r6i.4xlarge recommended)
- Qdrant Cloud account with Cloud Inference enabled
- Python 3.10+

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/Airheumatologist/RAG-pipeline.git
cd RAG-pipeline

# Create environment file
cp .env.example .env
# Edit .env with your Qdrant credentials

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data (on EC2)

```bash
# Download PMC articles from S3 (4-8 hours)
python scripts/01_download_pmc.py

# Download DailyMed drug labels (30 mins)
python scripts/03_download_dailymed.py
```

### 3. Extract & Process

```bash
# Extract PMC articles to JSONL (3-4 hours)
python scripts/02_extract_pmc.py --xml-dir /data/pmc_fulltext/xml --output /data/pmc_articles.jsonl

# Process DailyMed SPL files (10 mins)
python scripts/04_process_dailymed.py --spl-dir /data/dailymed/xml --output /data/dailymed_drugs.jsonl
```

### 4. Setup Qdrant & Ingest

```bash
# Create Qdrant collection
python scripts/05_setup_qdrant.py

# Ingest PMC articles (~60 mins)
python scripts/06_ingest_pmc.py --articles-file /data/pmc_articles.jsonl

# Ingest DailyMed drugs (~5 mins)
python scripts/07_ingest_dailymed.py --xml-dir /data/dailymed/xml
```

### 5. Monthly Updates

```bash
# Run monthly incremental update
python scripts/08_monthly_update.py
```

## 📁 Directory Structure

```
RAG-pipeline/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── docs/
│   └── PRD-Pipeline.md         # Detailed PRD documentation
└── scripts/
    ├── 01_download_pmc.py      # PMC S3 download
    ├── 02_extract_pmc.py       # PMC XML extraction
    ├── 03_download_dailymed.py # DailyMed download
    ├── 04_process_dailymed.py  # DailyMed processing
    ├── 05_setup_qdrant.py      # Qdrant collection setup
    ├── 06_ingest_pmc.py        # PMC ingestion
    ├── 07_ingest_dailymed.py   # DailyMed ingestion
    └── 08_monthly_update.py    # Monthly updates
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `DATA_DIR` | Base directory for data files (default: `/data`) |

### Key Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Batch Size | 50 | Optimal for Cloud Inference |
| Parallel Workers | 4 | Matches shard count |
| Max Text Length | 2000 | Cloud Inference limit |
| Shards | 4 | For parallel ingestion |

## 📋 Metadata Schema

### PMC Articles

```json
{
  "pmcid": "PMC12345678",
  "pmid": "12345678",
  "doi": "10.1000/example",
  "title": "Article Title",
  "abstract": "Abstract text...",
  "full_text": "Full article text (10K chars)...",
  "year": 2024,
  "journal": "Nature Medicine",
  "evidence_grade": "A",
  "country": "USA",
  "mesh_terms": ["Diabetes", "Treatment"],
  "has_full_text": true
}
```

### DailyMed Drugs

```json
{
  "set_id": "uuid",
  "drug_name": "Metformin",
  "active_ingredients": ["METFORMIN HYDROCHLORIDE"],
  "indications": "Treatment of type 2 diabetes...",
  "contraindications": "...",
  "warnings": "...",
  "dosage": "..."
}
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `500 Internal Server Error` | Reduce batch size to 50 |
| `cloud_inference not found` | Enable Cloud Inference in Qdrant dashboard |
| `Index required for year` | Run `05_setup_qdrant.py` to create indexes |
| Slow ingestion | Increase parallel workers (max 8) |

## 📚 Documentation

See [docs/PRD-Pipeline.md](docs/PRD-Pipeline.md) for the complete Production Requirements Document with:

- Detailed architecture overview
- Step-by-step implementation guide
- Performance benchmarks
- Cost estimates
- Lessons learned

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

**Maintained by:** Medical AI Team

