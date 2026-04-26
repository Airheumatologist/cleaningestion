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
│   ├── ingest_pmc_s3.py
│   ├── download_dailymed.py
│   └── ingest_dailymed_direct.py
├── updates/                               # recurring refreshes
│   ├── weekly_update.py
│   └── prepare_dailymed_updates.py
├── maintenance/                           # namespace ops + QA gates
│   ├── delete_turbopuffer_namespace.py
│   ├── validate_dailymed_namespace.py
│   └── run_dailymed_1k_test_gate.sh
├── config_ingestion.py                    # shared config (env-driven)
├── ingestion_utils.py                     # chunking, embeddings, parsing
├── turbopuffer_ingestion_sink.py          # write sink
├── pubmed_publication_filters.py
├── dailymed_rx_filters.py
└── dailymed_ingest_lib.py                 # legacy DailyMed XML→tpuf entry point used by 1k gate
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

# DailyMed — direct path (HTTPS ZIPs, in-memory parsing)
python3 scripts/baseline/ingest_dailymed_direct.py
```

The legacy XML-on-disk DailyMed flow (download → ingest) is still available
for the 1k QA gate:

```bash
python3 scripts/baseline/download_dailymed.py
python3 scripts/dailymed_ingest_lib.py --xml-dir "$DAILYMED_XML_DIR"
```

## Updates

```bash
# Cron-driven weekly refresh (PubMed + DailyMed deltas)
python3 scripts/updates/weekly_update.py

# Clear DailyMed checkpoint entries before re-ingesting an update batch
python3 scripts/updates/prepare_dailymed_updates.py \
  --set-id-manifest "$DAILYMED_SET_ID_MANIFEST"
```

## Maintenance

```bash
# Delete a turbopuffer namespace (refuses production namespaces by default)
python3 scripts/maintenance/delete_turbopuffer_namespace.py \
  --namespace medical_database_dailymed_test_1000_a --confirm-delete

# Validate a DailyMed namespace against an audit JSON
python3 scripts/maintenance/validate_dailymed_namespace.py \
  --namespace medical_database_dailymed_test_1000_a \
  --audit-json /path/to/audit.json --report-json /path/to/report.json

# Full 1k ingest → validate → delete gate
scripts/maintenance/run_dailymed_1k_test_gate.sh
```

## Tests

```bash
PYTHONPATH=. pytest tests/ -x
```

## Known follow-ups

- `ingestion_utils.py` and `updates/weekly_update.py` still import
  `qdrant_client` symbols. `qdrant-client` is therefore retained in
  `requirements.txt`. Removing the residue is tracked separately.
