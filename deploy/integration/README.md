# RAG API Integration Guide

## Base paths

- Primary (versioned):
  - `POST /api/v1/chat`
  - `POST /api/v1/chat/stream` (SSE)
  - `GET /api/v1/health`
  - `POST /api/v1/debug/decompose` (debug)
- Legacy aliases (temporary):
  - `/api/chat` -> `/api/v1/chat`
  - `/api/chat/stream` -> `/api/v1/chat/stream`
  - `/health` -> `/api/v1/health`

## Auth

All non-health endpoints require:

```http
Authorization: Bearer <service_token>
```

Error behavior:
- `401` missing/invalid token
- `403` token exists but is disabled

## Sync example (curl)

```bash
curl -X POST "http://<api-host>:8000/api/v1/chat" \
  -H "Authorization: Bearer $RAG_SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Latest guideline-based treatment options for rheumatoid arthritis"}'
```

## SSE example (curl)

```bash
curl -N -X POST "http://<api-host>:8000/api/v1/chat/stream" \
  -H "Authorization: Bearer $RAG_SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "SGLT2 inhibitor evidence in heart failure", "stream": true}'
```

## Node.js / TypeScript example

```ts
const res = await fetch("http://<api-host>:8000/api/v1/chat", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${process.env.RAG_SERVICE_TOKEN}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ query: "Management of IgG4-related disease" }),
});

if (!res.ok) throw new Error(`RAG API error: ${res.status}`);
const data = await res.json();
console.log(data.answer);
```

## Python example

```python
import requests

resp = requests.post(
    "http://<api-host>:8000/api/v1/chat",
    headers={"Authorization": f"Bearer {RAG_SERVICE_TOKEN}"},
    json={"query": "Neurobrucellosis diagnosis and treatment"},
    timeout=120,
)
resp.raise_for_status()
print(resp.json()["answer"])
```

## Retry/backoff + timeout guidance

- Use idempotent retries for `429`, `500`, `502`, `503`, `504`.
- Suggested policy: exponential backoff (`1s`, `2s`, `4s`, jitter), max 3 retries.
- Client timeout recommendations:
  - `/api/v1/chat`: `90-180s`
  - `/api/v1/chat/stream`: read timeout `>=300s`

## Connectivity patterns

- Local Mac dev via SSH tunnel:

```bash
ssh -L 8000:localhost:8000 <user>@<hetzner-host>
```

Then call `http://localhost:8000`.

- Azure runtime: call the private Hetzner endpoint over WireGuard from server-side code only.

## OpenAPI

- Spec URL: `/api/v1/openapi.json`
- Docs URL: `/api/v1/docs`

## Legacy route migration

- Migrate all callers to `/api/v1/*`.
- Keep aliases only during transition window.
- Recommended sunset target: 30 days after all production callers have switched.
