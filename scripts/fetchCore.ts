/** `pnpm vendor:core` — clone the pinned ruNNtime core into vendor/. */
import { ensureRunntimeCore, RUNNTIME_COMMIT, RUNNTIME_DIR } from './coreAsset'

const log = (message: string): void => {
  console.log(`[core] ${message}`)
}

try {
  const cloned = ensureRunntimeCore({ log, force: process.argv.includes('--force') })
  console.log(
    cloned
      ? `[core] ready at ${RUNNTIME_DIR} (${RUNNTIME_COMMIT.slice(0, 7)})`
      : `[core] already at ${RUNNTIME_COMMIT.slice(0, 7)}`,
  )
} catch (cause) {
  console.error(`[core] failed: ${cause instanceof Error ? cause.message : String(cause)}`)
  process.exit(1)
}
