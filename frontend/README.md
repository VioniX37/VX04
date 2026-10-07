# Frontend

Next.js 16 (App Router) · TypeScript · Tailwind CSS v4 web UI for GroundML.

```bash
npm install
npm run dev      # http://localhost:3000
npm run lint
npm run build
```

The UI talks to the FastAPI backend at `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`); set it in `.env.local` if the API runs elsewhere.

| Path | Purpose |
|---|---|
| `src/app/page.tsx` | New run: dataset upload / register by path or URL / pick registered, plus the task prompt |
| `src/app/runs/page.tsx` | Run history |
| `src/app/runs/[id]/page.tsx` | Live run view (server component resolving params; renders `RunView`) |
| `src/components/RunView.tsx` | SSE subscription and all run panels |
| `src/components/GroundingPanel.tsx` | Successive-halving table: predicted vs observed scores per rung |
| `src/lib/api.ts` | Typed API client (fetch + XHR upload progress + EventSource) |
| `src/lib/types.ts` | TypeScript mirrors of the backend schemas; keep in sync |

See the [documentation](../docs/index.md) for the full system.
