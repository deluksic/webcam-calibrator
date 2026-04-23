# webcam-calibrator

## Development

- `pnpm dev` — Vite dev server
- `pnpm type` — TypeScript check (`tsc --noEmit`)
- `pnpm build` — production build to `dist/`
- `pnpm test` — Vitest (also `test:subpixel`, `test:decode` for focused suites)
- `pnpm lint` / `pnpm fmt` — Oxlint, Oxfmt

Deploy by serving the `dist/` output as static files (any static host or CDN).

## Versioning

Build hash is injected at build time (`src/plugins/buildHash.ts`). The browser console shows `[build] <hash>` so you can confirm which build is running.

---

Project docs: [`README.md`](../README.md), [`ARCHITECTURE.md`](../ARCHITECTURE.md), [`docs/plan.md`](../docs/plan.md).
