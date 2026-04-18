import { execFile } from 'child_process';
import { readFile } from 'fs/promises';

/**
 * Checks watcher status: first verifies **watch processes are running** (`pgrep`), then
 * reads `/tmp` logs only for running watchers. If a watcher is not running, status is
 * **stopped** and that side does not parse logs or print extra detail.
 *
 * Log files (when the matching process is running):
 * - /tmp/tsc-watch.log — `pnpm typecheck:watch`
 * - /tmp/ship-watch.log — `pnpm ship:watch`
 *
 * Process patterns (ERE for `pgrep -f`): match `tsc … --noEmit … --watch` and
 * `vite … build … --watch` as started from this repo’s package scripts.
 *
 * Under **Vitest** (`VITEST=true`), process checks are skipped so unit tests stay hermetic.
 *
 * **Linux only** for the real CLI: requires `pgrep` (typically **procps-ng**). Other platforms
 * are not supported for `pnpm check` (use Vitest for parser coverage).
 *
 * If a watcher **is** running but the expected log file is **missing** (`ENOENT`), `runCheck`
 * **throws** (misconfigured `tee` path or permissions).
 */
const TC_LOG = '/tmp/tsc-watch.log';
const BUILD_LOG = '/tmp/ship-watch.log';

/** `pgrep -f` extended-regex patterns (`man pgrep` on Linux). */
export const PGREP_TSC_WATCH = String.raw`tsc .*--noEmit.*--watch`;
export const PGREP_VITE_BUILD_WATCH = String.raw`vite .*build.*--watch`;

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

function skipProcessChecks(): boolean {
  return process.env.VITEST === 'true';
}

/** Real `pnpm check` is intended for Linux hosts with `pgrep` (procps-ng). */
function assertLinuxCheckEnvironment(): void {
  if (process.env.VITEST === 'true') {
    return;
  }
  if (process.platform !== 'linux') {
    throw new Error(
      `pnpm check is only supported on Linux (got platform=${process.platform}). On other OSes use \`pnpm test\` for log-parser coverage.`,
    );
  }
}

/** Whether a local process matches `pgrep -f pattern` (exit 0 and non-empty stdout). */
export function pgrepHasMatch(pattern: string): Promise<boolean> {
  return new Promise((resolve, reject) => {
    execFile('pgrep', ['-f', pattern], { encoding: 'utf8', maxBuffer: 512 * 1024 }, (err, stdout) => {
      if (err) {
        const e = err as NodeJS.ErrnoException & { status?: number };
        if (e.code === 'ENOENT') {
          reject(
            new Error(
              'pgrep not found: install procps-ng (e.g. `apt install procps`) — required on Linux for pnpm check',
            ),
          );
          return;
        }
        const exit = typeof e.code === 'number' ? e.code : e.status;
        if (exit === 1) {
          resolve(false);
          return;
        }
        reject(err);
        return;
      }
      resolve((stdout as string).trim().length > 0);
    });
  });
}

/** Typecheck watcher (`tsc --noEmit --watch` …) is running. */
export async function isTypecheckWatcherRunning(): Promise<boolean> {
  if (skipProcessChecks()) {
    return true;
  }
  if (await pgrepHasMatch(PGREP_TSC_WATCH)) return true;
  return pgrepHasMatch(String.raw`tsover .*--noEmit.*--watch`);
}

/** Vite production build watch (`vite build --watch` …) is running. */
export async function isShipWatchRunning(): Promise<boolean> {
  if (skipProcessChecks()) {
    return true;
  }
  return pgrepHasMatch(PGREP_VITE_BUILD_WATCH);
}

/** If the watcher process is not running → stopped; otherwise parse log content. */
export async function resolveTscWatcherStatus(
  processRunning: boolean,
  logContent: string,
): Promise<ParseCheckResult> {
  if (!processRunning) {
    return { status: 'stopped' };
  }
  return parseTscLog(logContent);
}

/** If the watcher process is not running → stopped; otherwise parse log content. */
export async function resolveBuildWatcherStatus(
  processRunning: boolean,
  logContent: string,
): Promise<ParseCheckResult> {
  if (!processRunning) {
    return { status: 'stopped' };
  }
  return parseBuildLog(logContent);
}

/**
 * When `running` is false, returns `''` without touching the filesystem.
 * When `running` is true, reads `path` as UTF-8; throws if the file is missing (`ENOENT`)
 * or on other read errors.
 */
export async function readWatcherLogIfRunning(
  running: boolean,
  path: string,
  watcherLabel: string,
): Promise<string> {
  if (!running) {
    return '';
  }
  try {
    return await readFile(path, 'utf8');
  } catch (e) {
    const err = e as NodeJS.ErrnoException;
    if (err.code === 'ENOENT') {
      throw new Error(
        `${watcherLabel}: log file is missing but the watcher process appears to be running (expected tee to write ${path})`,
      );
    }
    throw e;
  }
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
  assertLinuxCheckEnvironment();

  const tscRunning = await isTypecheckWatcherRunning();
  const buildRunning = await isShipWatchRunning();

  const tscContent = await readWatcherLogIfRunning(tscRunning, TC_LOG, 'typecheck');
  const buildContent = await readWatcherLogIfRunning(buildRunning, BUILD_LOG, 'build');

  const tsc = await resolveTscWatcherStatus(tscRunning, tscContent);
  const build = await resolveBuildWatcherStatus(buildRunning, buildContent);

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
