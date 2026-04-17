# webcam-calibrator

## Development

```bash
# One-time: symlink dist to webroot
rm -rf /var/www/webcam-calibration.clodhost.com/public
ln -s /webcam-calibrator/dist /var/www/webcam-calibration.clodhost.com/public

# Watch mode (auto-rebuild + deploy via symlink)
pnpm ship:watch

# Type check in watch mode
pnpm typecheck:watch
```

## Status
```bash
ps aux | grep "vite build" | grep -v grep
ps aux | grep tsc | grep -v grep
```