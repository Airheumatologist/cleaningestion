# Fanout Canary Runbook (API-only, Qdrant-safe)

This runbook applies the dense source-family fanout rollout without restarting or modifying Qdrant container lifecycle.

## Safety Rules

- Do not stop/restart/recreate `qdrant`.
- Do not run destructive collection commands.
- Restart only API/gateway services with `--no-deps`.

## 1) Update runtime env

Edit `/opt/RAG-pipeline/.env` on server and set:

```env
QDRANT_QUERY_BACKEND=batch
QDRANT_INDEXED_ONLY=false
RETRIEVAL_SOURCE_FANOUT_ENABLED=false
RETRIEVAL_SOURCE_FANOUT_MODE=parallel
RETRIEVAL_SOURCE_FANOUT_MIN_RESULTS=60
RETRIEVAL_SOURCE_FANOUT_FALLBACK_BROAD=false
QUANTIZATION_RESCORE=true
QUANTIZATION_OVERSAMPLING=2.0
```

## 2) Baseline truth snapshot + benchmark

```bash
cd /opt/RAG-pipeline
python3 scripts/40_qdrant_truth_snapshot.py \
  --qdrant-url http://qdrant:6333 \
  --qdrant-api-key "$QDRANT_API_KEY" \
  --collection-name rag_pipeline \
  --output data/benchmarks/truth_baseline.json

python3 scripts/41_retrieval_benchmark.py \
  --queries-file data/benchmarks/queries_50.txt \
  --qdrant-url http://qdrant:6333 \
  --qdrant-api-key "$QDRANT_API_KEY" \
  --collection-name rag_pipeline \
  --output data/benchmarks/latency_baseline.json
```

## 3) Safe tuning (dry-run then apply)

```bash
cd /opt/RAG-pipeline
python3 scripts/42_qdrant_safe_tune.py \
  --qdrant-url http://qdrant:6333 \
  --qdrant-api-key "$QDRANT_API_KEY" \
  --collection-name rag_pipeline

python3 scripts/42_qdrant_safe_tune.py \
  --qdrant-url http://qdrant:6333 \
  --qdrant-api-key "$QDRANT_API_KEY" \
  --collection-name rag_pipeline \
  --apply
```

## 4) API-only restart

```bash
ssh -i .ssh/id_ed25519 root@65.109.112.253 \
  "cd /opt/RAG-pipeline/deploy/hetzner && docker compose --env-file ../../.env up -d --build --no-deps rag-api-1 rag-api-2 rag-api-3 rag-api-4 rag-gateway"
```

## 5) Canary progression

Set `RETRIEVAL_SOURCE_FANOUT_ENABLED=true` and roll out by traffic split:

1. 5%
2. 25%
3. 50%
4. 100%

At each stage, collect benchmark output and compare to baseline:

- p95 latency improvement
- timeout/5xx regression
- top-20 overlap >= 0.85

## 6) Instant rollback

Set:

```env
RETRIEVAL_SOURCE_FANOUT_ENABLED=false
```

Then run the same API-only restart command.
