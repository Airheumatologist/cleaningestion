# RAG API Integration Guide (Canonical)

This is one of the two canonical handoff docs.

- System, infrastructure, ingestion, and operations: `README.md`
- API contract and frontend integration details: `deploy/integration/README.md` (this file)

Legacy handoff drafts (`API_INFO.md`, `API_INTEGRATION_PLAN.md`) have been merged into the canonical docs.

## 1) Network model and trust boundaries

ASCII request path:

```text
Browser UI
  -> Next.js server route (/api/chat or /api/chat/stream)
  -> Hetzner rag-gateway:8000
  -> rag-api-* (FastAPI)
  -> qdrant:6333 (internal Docker network only)
```

Rules:
- Do not call the Hetzner API directly from browser code.
- Do not expose service tokens to client-side JavaScript.
- Keep Qdrant private; only API containers should access it.

## 2) Endpoints

Primary versioned routes:
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream` (SSE)
- `GET /api/v1/health`
- `POST /api/v1/debug/decompose` (debug)

Temporary legacy aliases:
- `POST /api/chat` -> `/api/v1/chat`
- `POST /api/chat/stream` -> `/api/v1/chat/stream`
- `GET /health` -> `/api/v1/health`
- `POST /api/query/decompose` -> `/api/v1/debug/decompose`

OpenAPI:
- Spec: `/api/v1/openapi.json`
- Interactive docs: `/api/v1/docs`

## 3) Auth

All non-health routes require:

```http
Authorization: Bearer <service_token>
```

Auth error behavior:
- `401`: missing or invalid token
- `403`: token exists but is disabled

## 4) Request and response basics

Minimal request body:

```json
{
  "query": "What are guideline-based options for rheumatoid arthritis?",
  "stream": true
}
```

Non-streaming (`POST /api/v1/chat`) returns JSON including:
- `answer`
- `sources[]`
- `evidence_hierarchy`
- `retrieval_stats`
- `status` (`success`, `fallback`, or `cache_hit`)

## 5) Streaming protocol (SSE)

Wire format:

```text
data: {json_event}\n\n
...

data: [DONE]\n\n
```

Expected event order for a successful run:
1. `query_expansion` running
2. `query_expansion` complete (`data.primary_query`, `data.keyword_query`)
3. `retrieval` running
4. `retrieval` complete (`data.count`)
5. `reranking` running
6. `reranking` complete (`data.papers`, `sources[]`, `evidence_hierarchy`)
7. `pdf_check` running
8. `generation` running (message, then token events)
9. `pdf_check` complete (`data.pdf_count`, updated `sources[]`)
10. `generation` complete
11. `complete` (`status`, `answer`, `sources[]`, `retrieval_stats`, ...)
12. `[DONE]`

Notes:
- `sources[]` first appears at `reranking` complete.
- `pdf_check` complete updates the same sources with `pdf_url` when available.
- Final answer is authoritative in the `complete` event.
- Cache responses can short-circuit directly to one `complete` event with `status=cache_hit`.
- Fallback responses use `status=fallback` when no usable literature context is available.

### Error event shape

Stream failures emit an `error` step before `[DONE]`.
Typical fields:
- `step: "error"`
- `status: "error"`
- `message`
- `retry_hint`
- `request_id` (when available)

## 6) Source object fields

Each `sources[]` item can include:
- `pmcid`, `pmid`, `doi`
- `title`, `authors[]`, `journal`, `year`
- `article_type`
- `evidence_grade`, `evidence_level`, `evidence_term`, `evidence_source`
- `relevance_score`
- `pdf_url`
- `source` (`pmc`, `pubmed`, `dailymed`)
- `set_id`, `dailymed_url` (for DailyMed)
- `citation_index`

`citation_index` ordering is used for answer citations like `[1]`, `[2]`.

## 7) Next.js server-proxy pattern

Use server routes to inject auth token.

```ts
// app/api/chat/stream/route.ts
export async function POST(req: Request) {
  const body = await req.text();

  const upstream = await fetch(`${process.env.RAG_API_BASE_URL}/api/v1/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.RAG_API_TOKEN}`,
    },
    body,
  });

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
```

Browser components should call relative routes only:
- `/api/chat`
- `/api/chat/stream`

## 8) Retry and timeout policy

Retryable status codes:
- `429`, `500`, `502`, `503`, `504`

Suggested policy:
- exponential backoff with jitter: `1s`, `2s`, `4s`
- max `3` retries for idempotent retries

Non-retryable status codes:
- `400` bad request
- `401` auth missing/invalid
- `403` token disabled

Timeout guidance:
- `/api/v1/chat`: `90s` to `180s`
- `/api/v1/chat/stream`: read timeout `>=300s`
- allow long gaps during reranking/generation

## 9) Security checklist

- Keep `RAG_API_TOKEN` server-side only.
- Never place service token in `NEXT_PUBLIC_*` variables.
- Use HTTPS in production.
- Restrict CORS origins to known service origins.
- Route browser traffic through your backend proxy.

## 10) Connectivity patterns

Local tunnel example:

```bash
ssh -L 8000:localhost:8000 <user>@<hetzner-host>
```

Then call `http://localhost:8000` from server-side code.

Azure runtime:
- Use private Hetzner endpoint over WireGuard.
- Keep token injection in server-side handlers.

## 11) Legacy route migration

- Migrate all callers to `/api/v1/*`.
- Keep aliases during transition.
- Recommended sunset: 30 days after all production callers have switched.
