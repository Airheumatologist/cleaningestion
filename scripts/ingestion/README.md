# PMC Full-Text Ingestion Pipeline

Scripts to download and ingest PMC Open Access articles with **full text** for the ScholarQA RAG pipeline.

## Prerequisites

- AWS EC2 instance (t3.xlarge, us-east-1 recommended)
- Python 3.10+
- Qdrant Cloud credentials

## Quick Start

### 1. AWS EC2 Setup

```bash
# SSH to your EC2 instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install dependencies
sudo yum update -y
sudo yum install python3.10 python3-pip -y

# Clone repo
git clone <your-repo> /home/ec2-user/rag-pipeline
cd /home/ec2-user/rag-pipeline

# Install Python packages
pip3 install boto3 tqdm sentence-transformers qdrant-client python-dotenv

# Create data directory
sudo mkdir -p /data/pmc_fulltext
sudo chown ec2-user:ec2-user /data/pmc_fulltext
```

### 2. Configure Environment

```bash
# Copy .env from local machine
scp .env ec2-user@your-instance-ip:/home/ec2-user/rag-pipeline/

# Verify
cat .env | grep QDRANT
```

### 3. Run Ingestion

```bash
cd /home/ec2-user/rag-pipeline/scripts/ingestion

# Step 1: Download from S3 (takes several hours)
nohup python3 01_download_pmc.py > download.log 2>&1 &
tail -f download.log

# Step 2: Parse XML and extract full text
nohup python3 02_parse_xml.py > parse.log 2>&1 &
tail -f parse.log

# Step 3: Create Qdrant collection  
python3 03_create_collection_inference.py

# Step 4: Ingest to Qdrant (can take 24+ hours)
nohup python3 04_ingest_fulltext.py > ingest.log 2>&1 &
tail -f ingest.log
```

## Configuration

Edit the scripts to adjust:

| Setting | File | Default |
|---------|------|---------|
| Years | `01_download_pmc.py` | 2015-2025 |
| Licenses | `01_download_pmc.py` | CC-BY, CC0, CC-BY-SA, CC-BY-ND |
| Full text limit | `02_parse_xml.py` | 50,000 chars |
| Batch size | `04_ingest_fulltext.py` | 100 |
| Collection name | All scripts | `pmc_medical_rag_fulltext` |

## Output Schema

Each article in Qdrant contains:

```json
{
  "pmcid": "PMC12345678",
  "title": "Article title...",
  "abstract": "Full abstract (not truncated)...",
  "full_text": "Complete body text (up to 50K chars)...",
  "year": 2024,
  "journal": "Nature Medicine",
  "keywords": ["cancer", "immunotherapy"],
  "authors": ["Smith, J.", "Jones, M."],
  "article_type": "research"
}
```

## Monitoring Progress

```bash
# Check download progress
wc -l /data/pmc_fulltext/xml/*.xml

# Check parsed articles
wc -l /data/pmc_fulltext/articles.jsonl

# Check Qdrant collection
curl -s "YOUR_QDRANT_URL/collections/pmc_medical_rag_fulltext" \
  -H "api-key: YOUR_API_KEY" | jq .result.points_count
```

## Resume Capability

All scripts support resume:
- `01_download_pmc.py`: Skips already downloaded XML files
- `04_ingest_fulltext.py`: Uses checkpoint file to skip processed articles
