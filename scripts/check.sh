#!/bin/bash
# Check status of typecheck and build watchers

TC_PATTERN="tsc --noEmit --watch"
BUILD_PATTERN="vite build --watch"
TC_LOG="/tmp/tsc-watch.log"
BUILD_LOG="/tmp/ship-watch.log"

tc_ok=0
build_ok=0

echo "=== typecheck ==="
tc_pid=$(ps aux | grep "$TC_PATTERN" | grep -v grep | awk '{print $2}' | head -1)
if [ -z "$tc_pid" ]; then
  echo "✗ not running — restart with: pnpm typecheck:watch &"
  tc_ok=1
else
  echo "✓ running (pid $tc_pid)"
  if [ -f "$TC_LOG" ]; then
    if grep -q "Found 0 errors" "$TC_LOG"; then
      echo "✓ pass"
    else
      echo "✗ fail — errors:"
      tail -20 "$TC_LOG" | grep -E "error TS|lib/|\\.ts:"
    fi
  fi
fi

echo ""
echo "=== build ==="
build_pid=$(ps aux | grep "$BUILD_PATTERN" | grep -v grep | awk '{print $2}' | head -1)
if [ -z "$build_pid" ]; then
  echo "✗ not running — restart with: pnpm ship:watch &"
  build_ok=1
else
  echo "✓ running (pid $build_pid)"
  if [ -f "$BUILD_LOG" ]; then
    if grep -q "built in" "$BUILD_LOG"; then
      echo "✓ pass"
    else
      echo "✗ fail:"
      tail -20 "$BUILD_LOG" | grep -vE "^$"
    fi
  fi
fi

if [ $tc_ok -eq 0 ] && [ $build_ok -eq 0 ]; then
  exit 0
else
  exit 1
fi
