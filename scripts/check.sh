#!/bin/bash
# Check status of typecheck and build watchers (append-only logs)

TC_LOG="/tmp/tsc-watch.log"
BUILD_LOG="/tmp/ship-watch.log"

echo "=== typecheck ==="
if [ ! -f "$TC_LOG" ]; then
  echo "⚠ no log file"
  exit 1
fi

tc_last=$(tail -n 1 "$TC_LOG" | sed 's/\x1b\[[0-9;]*[A-Za-z]//g')
if echo "$tc_last" | grep -q "Found 0 errors"; then
  echo "✓ pass"
elif echo "$tc_last" | grep -q "watching for file changes"; then
  echo "⚠ building..."
else
  echo "✗ fail — last:"
  # Find last "Found" or "Starting" boundary and print from there
  last_boundary=$(grep -n "Found\|Starting" "$TC_LOG" | tail -1 | cut -d: -f1)
  if [ -n "$last_boundary" ]; then
    tail -n +$last_boundary "$TC_LOG" | head -20
  else
    tail -20 "$TC_LOG"
  fi
  exit 1
fi

echo ""
echo "=== build ==="
if [ ! -f "$BUILD_LOG" ]; then
  echo "⚠ no log file"
  exit 1
fi

build_last=$(tail -n 1 "$BUILD_LOG")
if echo "$build_last" | grep -q "built in"; then
  echo "✓ pass"
  BUILD_HASH=$(cat dist/build-hash.txt 2>/dev/null)
  [ -n "$BUILD_HASH" ] && echo "Last: [$BUILD_HASH]"
elif echo "$build_last" | grep -q "build started"; then
  echo "⚠ building..."
elif echo "$build_last" | grep -q "watching for file changes"; then
  echo "⚠ building..."
else
  echo "✗ fail — last:"
  # Find last "build" boundary
  last_boundary=$(grep -n "build" "$BUILD_LOG" | tail -1 | cut -d: -f1)
  if [ -n "$last_boundary" ]; then
    tail -n +$last_boundary "$BUILD_LOG" | head -20
  else
    tail -20 "$BUILD_LOG"
  fi
  exit 1
fi

exit 0