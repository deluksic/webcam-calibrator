# webcam-calibrator

## Development

**IMPORTANT**: Never run `pnpm vite build`, `npx tsc`, or any build/typecheck commands manually. Watchers run in the background.

- `pnpm check` — check typecheck + build status (reads from running watchers). **Always use this** instead of `tsc` manually.
- `pnpm ship:watch` — rebuild + redeploy on file changes.
- `pnpm typecheck:watch` — typecheck on file changes.

Watchers log to:

- Typecheck: `/tmp/tsc-watch.log`
- Build: `/tmp/ship-watch.log`

## Setup

```bash
# One-time: symlink dist to webroot
rm -rf /var/www/webcam-calibration.clodhost.com/public
ln -s /webcam-calibrator/dist /var/www/webcam-calibration.clodhost.com/public
```

## Versioning

Build hash is automatic — injected by `src/plugins/buildHash.ts`. Check the `[build] <hash>` console log to verify which build is running.
