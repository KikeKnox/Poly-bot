# web/ — Poly-bot dashboard

React + Vite + TypeScript + shadcn/ui SPA. `pnpm build` writes to
`../src/api/static/dist/`, served by the bot's axum via `ServeDir`
(`static_router` in `src/api/mod.rs`).

```bash
pnpm dev      # development server with HMR
pnpm build    # production build (also invoked by ./launch.sh)
```
