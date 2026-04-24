/**
 * Confirms how `snapshot(store)` interacts with `createMemo` in Solid 2.
 *
 * Import the **client** build (`dist/solid.js`). The default `solid-js` entry
 * on Node resolves to `dist/server.js`, which does not model the same
 * fine-grained updates for this check.
 *
 * Run: node scripts/verify-snapshot-store-tracking.mjs
 */
import { createMemo, createRoot, createStore, flush, snapshot } from '../node_modules/solid-js/dist/solid.js'

createRoot((dispose) => {
  const [store, setStore] = createStore({ n: 0, items: [1, 2] })

  let snapOnlyRuns = 0
  const memoSnapOnly = createMemo(() => {
    snapOnlyRuns += 1
    return snapshot(store).n
  })

  let directRuns = 0
  const memoDirect = createMemo(() => {
    directRuns += 1
    return store.n
  })

  let snapAfterTrackRuns = 0
  const memoSnapAfterTrack = createMemo(() => {
    snapAfterTrackRuns += 1
    void store.n
    return snapshot(store).n
  })

  let trackThenSpreadRuns = 0
  const memoTrackThenSpread = createMemo(() => {
    trackThenSpreadRuns += 1
    const n = store.n
    return { n, items: [...store.items] }
  })

  // Initial compute
  void memoSnapOnly()
  void memoDirect()
  void memoSnapAfterTrack()
  void memoTrackThenSpread()
  const afterInit = {
    snapOnlyRuns,
    directRuns,
    snapAfterTrackRuns,
    trackThenSpreadRuns,
  }
  console.log('after first read', afterInit)

  setStore((s) => {
    s.n = 7
  })
  flush()

  void memoSnapOnly()
  void memoDirect()
  void memoSnapAfterTrack()
  void memoTrackThenSpread()

  const afterUpdate = {
    snapOnlyRuns,
    directRuns,
    snapAfterTrackRuns,
    trackThenSpreadRuns,
  }
  console.log('after n=7 + flush + second read', afterUpdate)

  setStore((s) => {
    s.items = [3, 4, 5]
  })
  flush()
  void memoSnapOnly()
  void memoDirect()
  void memoSnapAfterTrack()
  void memoTrackThenSpread()
  console.log('after items replaced', {
    snapOnlyRuns,
    directRuns,
    snapAfterTrackRuns,
    trackThenSpreadRuns,
  })

  console.log('\nResults (solid.js client build):')
  console.log(
    `  snapshot-only memo runs: ${snapOnlyRuns} — does NOT re-run when store.n or store.items change.`,
  )
  console.log(
    `  direct / track+snapshot / track+spread memos re-ran on updates (see counts above).`,
  )

  dispose()
})
