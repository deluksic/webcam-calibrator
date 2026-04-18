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
 * - pass: Typecheck found "Found 0 errors" OR build found "✓ X modules transformed"
 * - fail: Typecheck found error OR build found "error" in output
 *
 * The parser finds the most recent compilation/build cycle by looking for
 * "Starting compilation in watch mode", "File change detected. Starting incremental compilation",
 * or "build started" and captures output from that point forward.
 */
const TC_LOG = '/tmp/tsc-watch.log';
const BUILD_LOG = '/tmp/ship-watch.log';
const STATUS_MAP: Record<string, string> = {
  pass: '✓ pass',
  building: '⏳ building',
  fail: '✗ fail',
  stopped: '× stopped',
};

export async function parseTscLog(content: string): Promise<{ status: string; content: string }> {
  const lines = content.split('\n');
  const lastStartIdx = lines.findLastIndex(line =>
    line.includes('Starting compilation in watch mode') || line.includes('File change detected. Starting incremental compilation')
  );

  const searchFrom = lastStartIdx;

  if (searchFrom === -1) {
    return { status: 'stopped', content: '' };
  }

  const lastRunOutput = lines.slice(searchFrom);
  const foundLineIdx = lastRunOutput.findIndex(l => l.includes('Found '));
  const tcLine = foundLineIdx > -1 ? lastRunOutput[foundLineIdx] : lastRunOutput[0];
  const last8Lines = lastRunOutput.slice(-8).map(l => l.replace(/\x1b\[[0-9;]*[A-Za-z]/g, '').trim()).join('\n');

  if (tcLine.includes('Found 0 errors')) {
    return { status: 'pass', content: lastRunOutput.slice(foundLineIdx).join('\n').trim() };
  }

  if (foundLineIdx > -1 && tcLine.includes('error')) {
    return { status: 'fail', content: lastRunOutput.slice(foundLineIdx).join('\n').trim() };
  }

  return { status: 'building', content: last8Lines };
}

export async function parseBuildLog(content: string): Promise<{ status: string; content: string }> {
  const lines = content.split('\n');
  const lastBuildIdx = lines.findLastIndex(line => line.includes('build started'));

  const searchFrom = lastBuildIdx;

  if (searchFrom === -1) {
    return { status: 'stopped', content: '' };
  }

  const lastRunOutput = lines.slice(searchFrom);
  const last8Lines = lastRunOutput.slice(-8).map(l => l.trim()).join('\n');

  if (last8Lines.includes('✓') && last8Lines.includes('modules transformed')) {
    return { status: 'pass', content: last8Lines };
  }

  if (last8Lines.includes('error') || last8Lines.includes('Error')) {
    return { status: 'fail', content: last8Lines };
  }

  return { status: 'building', content: last8Lines };
}

function sanitizeLog(line: string): string {
  return line.replace(/\x1b\[[0-9;]*[A-Za-z]/g, '');
}

async function runCheck() {
  try {
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

    if (tsc.status !== 'pass' && tsc.content) {
      console.log(`typecheck: ${tsc.content.trim()}`);
    }

    if (build.status !== 'pass' && build.content) {
      console.log(`build: ${build.content.trim()}`);
    }
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

runCheck().catch(error => {
  console.error('Unhandled error:', error);
  process.exit(1);
});