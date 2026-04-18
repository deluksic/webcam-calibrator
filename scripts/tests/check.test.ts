import { describe, it, expect } from 'vitest';
import { parseTscLog, parseBuildLog } from '../check';

describe('parseTscLog', () => {
  it('should pass on Found 0 errors', async () => {
    const log = `Starting compilation in watch mode...
src/app.ts:5:10: error TS2339: Property does not exist.
Found 0 errors. Watching for file changes.`;
    const result = await parseTscLog(log);
    expect(result.status).toBe('pass');
    expect(result.content).toContain('Found 0 errors');
  });

  it('should fail on Found X errors', async () => {
    const log = `Starting compilation in watch mode...
src/app.ts:5:10: error TS2339: Property does not exist.
Found 2 errors. Watching for file changes.`;
    const result = await parseTscLog(log);
    expect(result.status).toBe('fail');
    expect(result.content).toContain('Found 2 errors');
    expect(result.content).toContain('Watching for file changes');
  });

  it('should capture second compilation cycle', async () => {
    const log = `Starting compilation in watch mode...
src/app.ts:5:10: error TS2339: Property does not exist.
Found 2 errors. Watching for file changes.
File change detected. Starting incremental compilation...
src/app.ts:7:10: error TS2339: Another error.
Found 1 error. Watching for file changes.`;
    const result = await parseTscLog(log);
    expect(result.status).toBe('fail');
    expect(result.content).toContain('Found 1 error');
  });

  it('should return building status without Found line', async () => {
    const log = `Starting compilation in watch mode...`;
    const result = await parseTscLog(log);
    expect(result.status).toBe('building');
  });
});

describe('parseBuildLog', () => {
  it('should pass on modules transformed', async () => {
    const log = `vite v7.3.2 building client environment for production...
watching for file changes...
build started...
[tsover] Type checking warnings:
✓ 191 modules transformed.`;
    const result = await parseBuildLog(log);
    expect(result.status).toBe('pass');
    expect(result.content).toContain('191 modules transformed');
  });

  it('should capture second build cycle', async () => {
    const log = `vite v7.3.2 building client environment for production...
watching for file changes...
build started...
✓ 191 modules transformed.
build started...
[tsover] warnings:
✓ 195 modules transformed.`;
    const result = await parseBuildLog(log);
    expect(result.status).toBe('pass');
    expect(result.content).toContain('195 modules transformed');
  });

  it('should fail on build errors', async () => {
    const log = `vite v7.3.2 building client environment for production...
watching for file changes...
build started...
error: something went wrong`;
    const result = await parseBuildLog(log);
    expect(result.status).toBe('fail');
  });
});