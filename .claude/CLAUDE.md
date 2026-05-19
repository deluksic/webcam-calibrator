# webcam-calibrator

## Development

- `pnpm dev` — Vite dev server
- `pnpm type` — TypeScript check (`tsc --noEmit`)
- `pnpm build` — production build to `dist/`
- `pnpm test` — Vitest (also `test:subpixel` for the subpixel suite)
- `pnpm lint` / `pnpm fmt` — Oxlint, Oxfmt

Deploy by serving the `dist/` output as static files (any static host or CDN).

## Versioning

Build hash is inferred from the entry module script URL (e.g. `index-<hash>.js`). The browser console shows `[build] <hash>` so you can confirm which build is running.

---

Project docs: [`README.md`](../README.md), [`docs/architecture.md`](../docs/architecture.md), [`docs/plan.md`](../docs/plan.md).
