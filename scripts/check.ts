import { readFile } from 'fs/promises';

/**
 * Checks watcher status by reading tsc-watch and ship-watch logs.
 *
 * Log files:
 * - /tmp/tsc-watch.log - TypeScript typecheck watcher output
 * - /tmp/ship-watch.log - Vite build watcher output
 *
 * Status detection:
 * - stopped: No compilation/build started (no matching keyword found)
 * - building: Process is running but not yet complete (no pass/fail indicators found)
 * - pass: Typecheck found "Found 0 errors" OR build completed (`modules transformed` / `built in …`).
 *   No `content` property (omit entirely).
 * - stopped: No matching cycle; no `content`.
 * - building / fail: include `content` (detail or tail of log).
 *
 * tsc: Uses the **last** `Found N errors` summary line in the slice after the latest start
 * marker (avoids stale earlier summaries in the same buffer).
 *
 * vite fail: Prefers explicit failure phrases over bare word "error" (reduces false positives
 * e.g. paths containing `/error/`).
 */
const TC_LOG = '/tmp/tsc-watch.log';
const BUILD_LOG = '/tmp/ship-watch.log';
const STATUS_MAP: Record<string, string> = {
  pass: '✓ pass',
  building: '⏳ building',
  fail: '✗ fail',
  stopped: '× stopped',
};

export type ParseCheckResult =
  | { status: 'pass' }
  | { status: 'stopped' }
  | { status: 'building'; content: string }
  | { status: 'fail'; content: string };

/** Strip common ANSI sequences from one line. */
export function sanitizeLine(line: string): string {
  return line
    .replace(/\x1b\[[0-9;]*m/g, '')
    .replace(/\x1b\[[0-9;]*[A-Za-z]/g, '');
}

/** Last line in `lines` that looks like a tsc `Found N error(s)` summary, or -1. */
export function findLastFoundSummaryLineIndex(lines: string[]): number {
  for (let i = lines.length - 1; i >= 0; i--) {
    const s = sanitizeLine(lines[i]).trim();
    if (/^Found \d+ errors?\.?/i.test(s)) {
      return i;
    }
  }
  return -1;
}

/** Parses N from `Found N errors` / `Found N error` summary line; null if not a summary. */
export function parseFoundErrorCount(line: string): number | null {
  const s = sanitizeLine(line).trim();
  const m = s.match(/^Found (\d+) errors?\.?/i);
  if (!m) return null;
  return parseInt(m[1], 10);
}

/** True when either watcher reports failure (for CLI exit code). */
export function checkResultsNeedFailureExit(
  tsc: ParseCheckResult,
  build: ParseCheckResult,
): boolean {
  return tsc.status === 'fail' || build.status === 'fail';
}

export async function parseTscLog(content: string): Promise<ParseCheckResult> {
  const lines = content.split('\n');
  const lastStartIdx = lines.findLastIndex(
    line =>
      line.includes('Starting compilation in watch mode') ||
      line.includes('File change detected. Starting incremental compilation'),
  );

  if (lastStartIdx === -1) {
    return { status: 'stopped' };
  }

  const lastRunOutput = lines.slice(lastStartIdx);
  const foundLineIdx = findLastFoundSummaryLineIndex(lastRunOutput);
  const tail = lastRunOutput.slice(-8).map(l => sanitizeLine(l).trim()).join('\n');

  if (foundLineIdx === -1) {
    return { status: 'building', content: tail };
  }

  const foundLine = lastRunOutput[foundLineIdx];
  const count = parseFoundErrorCount(foundLine);
  if (count === 0) {
    return { status: 'pass' };
  }
  if (count !== null && count > 0) {
    return { status: 'fail', content: lastRunOutput.slice(foundLineIdx).join('\n').trim() };
  }

  return { status: 'building', content: tail };
}

function buildRunLooksComplete(sanitizedLines: string[]): boolean {
  return sanitizedLines.some(
    l =>
      l.includes('modules transformed') ||
      /\bbuilt in\s+[\d.]+/i.test(l) ||
      /\b✓\s*built in\b/i.test(l),
  );
}

function buildRunLooksFailed(sanitizedLines: string[]): boolean {
  const joined = sanitizedLines.join('\n');
  if (/error during build/i.test(joined)) return true;
  if (/Rollup failed/i.test(joined)) return true;
  if (/Build failed/i.test(joined)) return true;

  return sanitizedLines.some(line => {
    const t = sanitizeLine(line).trim();
    return (
      /^\s*error:\s*/i.test(t) ||
      /error TS\d+/i.test(t) ||
      /\bnpm ERR!/i.test(t)
    );
  });
}

export async function parseBuildLog(content: string): Promise<ParseCheckResult> {
  const lines = content.split('\n');
  const lastBuildIdx = lines.findLastIndex(line => line.includes('build started'));

  if (lastBuildIdx === -1) {
    return { status: 'stopped' };
  }

  const lastRunOutput = lines.slice(lastBuildIdx);
  const sanitizedRun = lastRunOutput.map(l => sanitizeLine(l));

  if (buildRunLooksFailed(sanitizedRun)) {
    return { status: 'fail', content: lastRunOutput.join('\n').trim() };
  }

  if (buildRunLooksComplete(sanitizedRun)) {
    return { status: 'pass' };
  }

  const tail = lastRunOutput.slice(-12).map(l => sanitizeLine(l)).join('\n');
  return { status: 'building', content: tail };
}

export async function runCheck(): Promise<void> {
  let tscContent = '';
  let buildContent = '';

  try {
    tscContent = await readFile(TC_LOG, 'utf8');
  } catch {
    // File doesn't exist yet
  }

  try {
    buildContent = await readFile(BUILD_LOG, 'utf8');
  } catch {
    // File doesn't exist yet
  }

  const tsc = await parseTscLog(tscContent);
  const build = await parseBuildLog(buildContent);

  console.log(`typecheck: ${STATUS_MAP[tsc.status]}`);
  console.log(`build: ${STATUS_MAP[build.status]}`);

  if (tsc.status !== 'pass' && 'content' in tsc && tsc.content) {
    console.log(`typecheck: ${tsc.content.trim()}`);
  }

  if (build.status !== 'pass' && 'content' in build && build.content) {
    console.log(`build: ${build.content.trim()}`);
  }

  if (checkResultsNeedFailureExit(tsc, build)) {
    process.exit(1);
  }
}

const isVitest = process.env.VITEST === 'true';

if (!isVitest) {
  runCheck().catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}
