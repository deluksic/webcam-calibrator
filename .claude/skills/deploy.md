# Deploy Static Site to Clodhost

```bash
pnpm build && rsync -a --delete dist/ /var/www/webcam-calibration.clodhost.com/public/
```

DocRoot: `/var/www/webcam-calibration.clodhost.com/public`
