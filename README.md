# cleaningestion

Ingestion-only repo for loading PMC, PubMed, and DailyMed into Turbopuffer.
Baseline (one-time bulk loads) is complete; ongoing operation is the weekly
update cron and occasional namespace maintenance.

## Layout

```
scripts/
├── baseline/                              # one-time bulk loads
│   ├── download_pubmed_baseline.py
│   ├── ingest_pubmed_ftp_stream.py
│   ├── run_pubmed_ftp_stream_supervisor.sh
│   └── ingest_pmc_s3.py
├── updates/                               # recurring refreshes
│   ├── weekly_update.py
│   └── ingest_dailymed_updates_direct.py
├── maintenance/                           # namespace ops + QA gates
│   └── delete_turbopuffer_namespace.py
├── config_ingestion.py                    # shared config (env-driven)
├── ingestion_utils.py                     # chunking, embeddings, parsing
├── turbopuffer_ingestion_sink.py          # write sink
├── pubmed_publication_filters.py
├── dailymed_rx_filters.py
└── dailymed_ingest_lib.py                 # shared DailyMed SPL parsing/chunking helpers
tests/
├── test_06_ingest_pmc_s3.py
├── test_turbopuffer_ingestion_sink.py
└── test_weekly_update_alignment.py
data/                                       # checkpoints, audit logs, state
```

Shared modules live at `scripts/` root; subdir scripts insert
`scripts/` onto `sys.path` so imports like `from config_ingestion import …`
resolve regardless of invocation directory.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env   # then edit values
```

Required env at minimum: `TURBOPUFFER_API_KEY`, `DEEPINFRA_API_KEY`,
`DATA_DIR`. See [env.example](env.example) for the full list.

## Baseline (already run; commands kept for reproducibility)

```bash
# PubMed — stream from NCBI FTP, no local staging
python3 scripts/baseline/ingest_pubmed_ftp_stream.py
# (use the supervisor wrapper for unattended long runs)
scripts/baseline/run_pubmed_ftp_stream_supervisor.sh

# PMC — stream OA + author manuscripts from the AWS S3 inventory
python3 scripts/baseline/ingest_pmc_s3.py

# DailyMed baseline was already loaded. Ongoing DailyMed ingestion uses
# the direct update streamer below.
```

## Updates

```bash
# Cron-driven weekly refresh (PubMed + DailyMed deltas)
python3 scripts/updates/weekly_update.py

# DailyMed-only direct refresh, streaming DailyMed daily ZIPs into Turbopuffer
python3 scripts/updates/ingest_dailymed_updates_direct.py
```

## Maintenance

```bash
# Delete a Turbopuffer namespace (refuses production namespaces by default)
python3 scripts/maintenance/delete_turbopuffer_namespace.py \
  --namespace medical_database_dailymed_test_1000_a --confirm-delete
```

## Tests

```bash
PYTHONPATH=. pytest tests/ -x
```

## Known follow-ups

- `ingestion_utils.py` and `updates/weekly_update.py` still import
  `qdrant_client` symbols. `qdrant-client` is therefore retained in
  `requirements.txt`. Removing the residue is tracked separately.
