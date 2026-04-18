#!/bin/bash
# Run Vite build in watch mode (append-only log)
exec vite build --watch 2>&1 | tee /tmp/ship-watch.log