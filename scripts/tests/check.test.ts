import { readFile } from 'fs/promises';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { describe, it, expect } from 'vitest';
import {
  parseTscLog,
  parseBuildLog,
  sanitizeLine,
  findLastFoundSummaryLineIndex,
  parseFoundErrorCount,
  checkResultsNeedFailureExit,
  resolveTscWatcherStatus,
  resolveBuildWatcherStatus,
  readWatcherLogIfRunning,
} from '../check';

const FIXTURE_DIR = dirname(fileURLToPath(import.meta.url));

describe('sanitizeLine', () => {
  it('strips common ANSI SGR sequences', () => {
    expect(sanitizeLine('\x1b[32mFound 0 errors\x1b[0m')).toBe('Found 0 errors');
    expect(sanitizeLine('\x1b[1;31merror\x1b[m')).toBe('error');
  });
});

describe('parseFoundErrorCount', () => {
  it('parses plural and singular summaries', () => {
    expect(parseFoundErrorCount('Found 0 errors. Watching for file changes.')).toBe(0);
    expect(parseFoundErrorCount('Found 1 error. Watching for file changes.')).toBe(1);
    expect(parseFoundErrorCount('Found 12 errors. Watching for file changes.')).toBe(12);
  });

  it('parses tsc watch timestamp prefix before Found', () => {
    expect(
      parseFoundErrorCount('9:06:31 PM - Found 17 errors. Watching for file changes.'),
    ).toBe(17);
  });

  it('returns null for non-summary lines', () => {
    expect(parseFoundErrorCount('src/a.ts(1,1): error TS2304: not found')).toBe(null);
    expect(parseFoundErrorCount('Found configuration at ./tsconfig.json')).toBe(null);
  });
});

describe('findLastFoundSummaryLineIndex', () => {
  it('prefers the last Found summary in the buffer', () => {
    const lines = [
      'File change detected. Starting incremental compilation...',
      'Found 0 errors. Watching for file changes.',
      'more noise',
      'Found 2 errors. Watching for file changes.',
    ];
    expect(findLastFoundSummaryLineIndex(lines)).toBe(3);
  });
});

describe('parseTscLog', () => {
  it('should pass on Found 0 errors', async () => {
    const log = `Starting compilation in watch mode...
src/app.ts:5:10: error TS2339: Property does not exist.
Found 0 errors. Watching for file changes.`;
    expect(await parseTscLog(log)).toEqual({ status: 'pass' });
  });

  it('should fail on Found X errors', async () => {
    const log = `Starting compilation in watch mode...
src/app.ts:5:10: error TS2339: Property does not exist.
Found 2 errors. Watching for file changes.`;
    const result = await parseTscLog(log);
    expect(result.status).toBe('fail');
    if (result.status === 'fail') {
      expect(result.content).toContain('Found 2 errors');
      expect(result.content).toContain('Watching for file changes');
    }
  });

  it('uses the latest Found summary after a new incremental cycle', async () => {
    const log = `Starting compilation in watch mode...
Found 2 errors. Watching for file changes.
File change detected. Starting incremental compilation...
src/app.ts:1:1: error TS1000: fixme
Found 0 errors. Watching for file changes.`;
    expect(await parseTscLog(log)).toEqual({ status: 'pass' });
  });

  it('should capture second compilation cycle failure', async () => {
    const log = `Starting compilation in watch mode...
src/app.ts:5:10: error TS2339: Property does not exist.
Found 2 errors. Watching for file changes.
File change detected. Starting incremental compilation...
src/app.ts:7:10: error TS2339: Another error.
Found 1 error. Watching for file changes.`;
    const result = await parseTscLog(log);
    expect(result.status).toBe('fail');
    if (result.status === 'fail') {
      expect(result.content).toContain('Found 1 error');
    }
  });

  it('should return building status without Found line', async () => {
    const log = `Starting compilation in watch mode...`;
    const result = await parseTscLog(log);
    expect(result.status).toBe('building');
    if (result.status === 'building') {
      expect(result.content.length).toBeGreaterThan(0);
    }
  });

  it('parses scripts/tests/tsc-watch.log fixture (timestamped Found lines)', async () => {
    const log = await readFile(join(FIXTURE_DIR, 'tsc-watch.log'), 'utf8');
    const result = await parseTscLog(log);
    expect(result.status).toBe('fail');
    if (result.status === 'fail') {
      expect(result.content).toContain('Found 17 errors');
    }
  });

  it('parses multi-cycle tsc log like tee output (snippet total is full current cycle)', async () => {
    const log = await readFile(join(FIXTURE_DIR, 'tsc-watch-multi-cycle.log'), 'utf8');
    const result = await parseTscLog(log);
    expect(result.status).toBe('fail');
    if (result.status === 'fail') {
      expect(result.content).toContain('Found 2 errors');
      expect(result.content).toContain('rrors. Watching for file changes.');
      expect(result.snippet?.kind).toBe('lines');
      if (result.snippet?.kind === 'lines') {
        expect(result.snippet.shown).toBeLessThan(result.snippet.total);
      }
    }
  });

  it('returns fail when no tsc start marker', async () => {
    const result = await parseTscLog('random noise\nno compilation here\n');
    expect(result.status).toBe('fail');
    if (result.status === 'fail') {
      expect(result.hint).toContain('No tsc watch start marker');
      expect(result.snippet).toMatchObject({ kind: 'first' });
      expect(result.snippet?.shown).toBeLessThanOrEqual(8);
      expect(result.content).toContain('random noise');
    }
  });

  it('does not include a prior watch cycle in building excerpts', async () => {
    const log = `Starting compilation in watch mode
EARLIER_CYCLE_UNIQUE_MARKER
Found 0 errors. Watching for file changes.
File change detected. Starting incremental compilation...
only_latest_cycle
p1
p2
p3
p4
p5
p6
p7`;
    const result = await parseTscLog(log);
    expect(result.status).toBe('building');
    if (result.status === 'building') {
      expect(result.content).not.toContain('EARLIER_CYCLE_UNIQUE_MARKER');
      expect(result.content).toContain('only_latest_cycle');
    }
  });
});

describe('parseBuildLog', () => {
  it('should pass on modules transformed', async () => {
    const log = `vite v7.3.2 building client environment for production...
watching for file changes...
build started...
[tsover] Type checking warnings:
✓ 191 modules transformed.`;
    expect(await parseBuildLog(log)).toEqual({ status: 'pass' });
  });

  it('should pass when success markers are far from end of log', async () => {
    const filler = Array.from({ length: 25 }, (_, i) => `line ${i} after transform`).join('\n');
    const log = `watching for file changes...
build started...
transforming...
✓ 190 modules transformed.
${filler}
rendering chunks...
computing gzip size...
dist/index.html                   0.50 kB
dist/assets/index-abc123.js     100.00 kB
✓ built in 2.02s`;
    expect(await parseBuildLog(log)).toEqual({ status: 'pass' });
  });

  it('should capture second build cycle', async () => {
    const log = `vite v7.3.2 building client environment for production...
watching for file changes...
build started...
✓ 191 modules transformed.
build started...
[tsover] warnings:
✓ 195 modules transformed.`;
    expect(await parseBuildLog(log)).toEqual({ status: 'pass' });
  });

  it('should pass on built in only (no modules transformed line)', async () => {
    const log = `watching for file changes...
build started...
rendering chunks...
✓ built in 0.42s`;
    expect(await parseBuildLog(log)).toEqual({ status: 'pass' });
  });

  it('should fail on error: prefix', async () => {
    const log = `vite v7.3.2 building client environment for production...
watching for file changes...
build started...
error: something went wrong`;
    const result = await parseBuildLog(log);
    expect(result.status).toBe('fail');
    if (result.status === 'fail') {
      expect(result.content).toContain('something went wrong');
    }
  });

  it('should fail on error during build', async () => {
    const log = `build started...
transforming...
error during build:
Could not resolve "./missing"`;
    const result = await parseBuildLog(log);
    expect(result.status).toBe('fail');
  });

  it('should fail on Rollup failed', async () => {
    const log = `build started...
Rollup failed to resolve import`;
    expect((await parseBuildLog(log)).status).toBe('fail');
  });

  it('should fail on error TS in log', async () => {
    const log = `build started...
src/foo.ts(1,1): error TS2304: Cannot find name 'x'.`;
    expect((await parseBuildLog(log)).status).toBe('fail');
  });

  it('should not false-fail on path containing /error/', async () => {
    const log = `build started...
✓ 10 modules transformed.
(!) /Users/me/proj/src/error/utils.ts is dynamically imported
rendering chunks...
✓ built in 1.00s`;
    expect(await parseBuildLog(log)).toEqual({ status: 'pass' });
  });

  it('returns fail without build started', async () => {
    const result = await parseBuildLog('vite v7 watching\n');
    expect(result.status).toBe('fail');
    if (result.status === 'fail') {
      expect(result.hint).toContain('No "build started"');
      expect(result.snippet).toMatchObject({ kind: 'first' });
      expect(result.snippet?.shown).toBeLessThanOrEqual(8);
      expect(result.content).toContain('vite v7 watching');
    }
  });

  it('returns building when run has started but no completion yet', async () => {
    const log = `watching for file changes...
build started...
transforming...`;
    const result = await parseBuildLog(log);
    expect(result.status).toBe('building');
    if (result.status === 'building') {
      expect(result.content).toContain('transforming');
    }
  });

  it('does not include a prior build cycle in building excerpts', async () => {
    const log = `watching for file changes...
build started...
EARLIER_BUILD_UNIQUE_MARKER
✓ 1 modules transformed.
✓ built in 0.01s
build started...
only_latest_build_cycle
p1
p2
p3
p4
p5
p6
p7`;
    const result = await parseBuildLog(log);
    expect(result.status).toBe('building');
    if (result.status === 'building') {
      expect(result.content).not.toContain('EARLIER_BUILD_UNIQUE_MARKER');
      expect(result.content).toContain('only_latest_build_cycle');
    }
  });
});

describe('readWatcherLogIfRunning', () => {
  it('returns skip when watcher is not running (no read)', async () => {
    expect(await readWatcherLogIfRunning(false, '/this/path/does/not/exist', 'typecheck')).toEqual(
      { kind: 'skip' },
    );
  });

  it('returns missing when watcher is running but log file does not exist', async () => {
    const path = '/nonexistent-tsc-watch-log-xyz.log';
    expect(await readWatcherLogIfRunning(true, path, 'typecheck')).toEqual({
      kind: 'missing',
      path,
      label: 'typecheck',
    });
  });
});

describe('resolveTscWatcherStatus / resolveBuildWatcherStatus', () => {
  it('returns stopped when process is not running, without parsing pass from stale log', async () => {
    const stalePassLog = `Starting compilation in watch mode...
Found 0 errors. Watching for file changes.`;
    expect(await resolveTscWatcherStatus(false, stalePassLog)).toEqual({ status: 'stopped' });

    const staleBuildLog = `watching for file changes...
build started...
✓ 999 modules transformed.
✓ built in 0.01s`;
    expect(await resolveBuildWatcherStatus(false, staleBuildLog)).toEqual({ status: 'stopped' });
  });

  it('parses log when process is running', async () => {
    const log = `Starting compilation in watch mode...
Found 0 errors. Watching for file changes.`;
    expect(await resolveTscWatcherStatus(true, log)).toEqual({ status: 'pass' });

    const blog = `build started...
✓ 1 modules transformed.`;
    expect(await resolveBuildWatcherStatus(true, blog)).toEqual({ status: 'pass' });
  });
});

describe('checkResultsNeedFailureExit', () => {
  it('is true when either watcher failed', () => {
    expect(
      checkResultsNeedFailureExit({ status: 'fail', content: 'x' }, { status: 'pass' }),
    ).toBe(true);
    expect(
      checkResultsNeedFailureExit({ status: 'pass' }, { status: 'fail', content: 'y' }),
    ).toBe(true);
  });

  it('is false when both pass or non-fail', () => {
    expect(checkResultsNeedFailureExit({ status: 'pass' }, { status: 'pass' })).toBe(false);
    expect(
      checkResultsNeedFailureExit(
        { status: 'building', content: '', snippet: { kind: 'last', shown: 0, total: 0 } },
        { status: 'pass' },
      ),
    ).toBe(false);
    expect(checkResultsNeedFailureExit({ status: 'stopped' }, { status: 'stopped' })).toBe(
      false,
    );
  });
});
