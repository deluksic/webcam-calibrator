#!/bin/bash
# Watch src/ for changes, build, and sync to web server

WEB_DIR="/var/www/webcam-calibration.clodhost.com/public"
SRC_DIR="/webcam-calibrator/src"

echo "Starting build watcher..."
echo "Watching: $SRC_DIR"
echo "Target: $WEB_DIR"

LAST_BUILD=0

while true; do
  # Find most recent file modification in src/
  CURRENT=$(find $SRC_DIR -type f -name "*.ts" -o -name "*.tsx" | xargs stat -c %Y 2>/dev/null | sort -n | tail -1)

  if [ -n "$CURRENT" ] && [ "$CURRENT" -gt "$LAST_BUILD" ]; then
    echo "[$(date +%H:%M:%S)] Change detected, building..."
    cd /webcam-calibrator
    pnpm build > /tmp/build.log 2>&1
    if [ $? -eq 0 ]; then
      echo "Build succeeded, syncing..."
      rm -rf $WEB_DIR/*
      cp -r /webcam-calibrator/dist/* $WEB_DIR/
      echo "Synced to $WEB_DIR"
    else
      echo "Build failed, check /tmp/build.log"
    fi
    LAST_BUILD=$(date +%s)
  fi
  sleep 2
done