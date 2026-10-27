/** Clones the pinned ruNNtime core into the git-ignored `vendor/` directory.
 *
 *  Why a clone instead of a dependency: ruNNtime's `core` layer is not published
 *  — the package's `exports` map only has `./zoo`, and `tsdown` has no `core`
 *  entry, so `dist/` never contains it. Its `prepack` build also fails when run
 *  from `node_modules`. Cloning the sources and importing them directly is the
 *  only route that does not involve forking or vendoring code into git.
 *
 *  See docs/runntime-evaluation.md.
 */
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

export const RUNNTIME_REPO = 'https://github.com/software-mansion/runntime.git'
/** Tested commit. Bump deliberately, then re-run `pnpm test` and the spike. */
export const RUNNTIME_COMMIT = 'b7aaa6444bd01c3359c095fdd5112410ba384f41'

export const RUNNTIME_DIR = path.resolve(root, 'vendor/runntime')
const CORE_ENTRY = path.resolve(
  RUNNTIME_DIR,
  'packages/runntime/src/core/index.ts',
)

export type FetchCoreOptions = {
  log?: (message: string) => void
  /** Re-clone even when the pinned commit is already present. */
  force?: boolean
}

function currentCommit(dir: string): string | undefined {
  try {
    return execFileSync('git', ['-C', dir, 'rev-parse', 'HEAD'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
  } catch {
    return undefined
  }
}

/** Ensures `vendor/runntime` exists at RUNNTIME_COMMIT. Returns true if it
 *  cloned, false if the checkout was already correct. */
export function ensureRunntimeCore(options: FetchCoreOptions = {}): boolean {
  const log = options.log ?? (() => {})
  if (fs.existsSync(CORE_ENTRY) && !options.force) {
    const head = currentCommit(RUNNTIME_DIR)
    if (head === RUNNTIME_COMMIT) {
      return false
    }
    if (head === undefined) {
      // A directory without git metadata can't be verified; leave it alone
      // rather than deleting something the user put there.
      log(`vendor/runntime present but unversioned; using it as-is`)
      return false
    }
  }

  if (fs.existsSync(RUNNTIME_DIR)) {
    log(`re-cloning vendor/runntime at ${RUNNTIME_COMMIT.slice(0, 7)}`)
    fs.rmSync(RUNNTIME_DIR, { recursive: true, force: true })
  } else {
    log(`cloning ${RUNNTIME_REPO}`)
  }

  fs.mkdirSync(path.dirname(RUNNTIME_DIR), { recursive: true })
  // A full clone so the pinned commit resolves even if `main` has moved on.
  execFileSync('git', ['clone', '--quiet', RUNNTIME_REPO, RUNNTIME_DIR], {
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  execFileSync('git', ['-C', RUNNTIME_DIR, 'checkout', '--quiet', RUNNTIME_COMMIT], {
    stdio: ['ignore', 'pipe', 'pipe'],
  })

  if (!fs.existsSync(CORE_ENTRY)) {
    throw new Error(
      `runntime checkout at ${RUNNTIME_COMMIT} has no packages/runntime/src/core/index.ts`,
    )
  }
  log(`vendor/runntime at ${RUNNTIME_COMMIT.slice(0, 7)}`)
  return true
}
