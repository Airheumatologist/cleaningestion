# Frontend (Next.js)

## Local development

```bash
npm install
npm run dev
```

App runs on `http://localhost:3000`.

## API routing model

All browser calls use relative paths (for example: `/api/chat/stream`).

Next.js rewrites proxy `/api/:path*` to the backend target configured by env:

- `API_REWRITE_TARGET` (preferred)
- fallback: `NEXT_PUBLIC_API_URL`
- default: `http://localhost:8000`

Example:

```bash
API_REWRITE_TARGET=http://localhost:8000 npm run dev
```

For Azure App Service, set `API_REWRITE_TARGET` to the private Hetzner API endpoint reachable over WireGuard.

If the backend has `API_AUTH_ENABLED=true`, use a server-side proxy that injects the bearer token, or disable auth in local-only environments.
