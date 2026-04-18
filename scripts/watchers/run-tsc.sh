#!/bin/bash
# Run TypeScript typecheck in watch mode (append-only log)
exec tsc --noEmit --watch 2>&1 | tee /tmp/tsc-watch.log