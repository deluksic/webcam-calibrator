# webcam-calibrator

## Development

**IMPORTANT**: Never run `pnpm vite build`, `npx tsc`, or any build/typecheck commands manually. A continuous watcher runs in the background. Use only the scripts in package.json:
- `pnpm ship:watch` - Watch mode (auto-rebuild + deploy via symlink)
- `pnpm typecheck:watch` - Type check in watch mode

## Setup

```bash
# One-time: symlink dist to webroot
rm -rf /var/www/webcam-calibration.clodhost.com/public
ln -s /webcam-calibrator/dist /var/www/webcam-calibration.clodhost.com/public
```

## Status
```bash
ps aux | grep "vite build" | grep -v grep
ps aux | grep tsc | grep -v grep
```

## Versioning

Increment the version number in `src/main.tsx` console.log on every change:
```
console.log('[build] v{N} - change description');
```