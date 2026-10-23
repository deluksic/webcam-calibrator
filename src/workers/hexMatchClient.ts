import * as Comlink from 'comlink'

import type { HexTargetConfig } from '@/lib/hexTarget/config'
import type { HexObservedPoint, HexTargetMatch } from '@/lib/hexTarget/match'

import type { HexMatchJob, HexMatchWorkerApi } from './hexMatch.worker'

let worker: Worker | undefined
let proxy: Comlink.Remote<HexMatchWorkerApi> | undefined

function remote(): Comlink.Remote<HexMatchWorkerApi> {
  if (proxy === undefined) {
    worker = new Worker(new URL('./hexMatch.worker.ts', import.meta.url), { type: 'module' })
    proxy = Comlink.wrap<HexMatchWorkerApi>(worker)
  }
  return proxy
}

/** Rebuild the worker's target cache before the first frame that needs it. */
export function configureHexMatchWorker(config: HexTargetConfig): void {
  void Promise.resolve(remote().configure(config)).catch((cause: unknown) => {
    console.error('[hex-match-worker] configure failed', cause)
  })
}

/**
 * Run observed-marker matching against the target design in the worker. The
 * returned match keeps the full `MatchedIdentityPoint` metadata (status,
 * recovery distance and conflicts) used by calibration acceptance.
 */
export function matchHexTargetInWorker(
  config: HexTargetConfig,
  points: readonly HexObservedPoint[],
  options: Pick<HexMatchJob, 'depth' | 'recovery'>,
): Promise<HexTargetMatch> {
  return remote().match(config, {
    points,
    depth: options.depth,
    recovery: options.recovery,
  })
}

export function disposeHexMatchWorker(): void {
  worker?.terminate()
  worker = undefined
  proxy = undefined
}
