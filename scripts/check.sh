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
    tc_content=$(cat "$TC_LOG" | tr -d '\0' 2>/dev/null)
    if echo "$tc_content" | grep -qa "Found 0 errors"; then
      echo "✓ pass"
    elif echo "$tc_content" | grep -qaE "error TS"; then
      echo "✗ fail — errors:"
      echo "$tc_content" | grep -E "error TS|lib/" | tail -10
    else
      echo "⚠ checking..."
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
    build_content=$(cat "$BUILD_LOG" | tr -d '\0' 2>/dev/null)
    if echo "$build_content" | grep -qa "built in"; then
      echo "✓ pass"
    elif echo "$build_content" | grep -qa "build started"; then
      echo "⚠ building..."
    else
      echo "✗ fail:"
      echo "$build_content" | tail -20
    fi
  fi
fi

if [ $tc_ok -eq 0 ] && [ $build_ok -eq 0 ]; then
  exit 0
else
  exit 1
fi