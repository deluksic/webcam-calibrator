import * as Comlink from 'comlink'

import type { IdentityDesignSpec, IdentityRunOutput } from '@/lib/pointIdentity/identityPipeline'
import {
  filterRefinedPeaks,
  type RoiNeighbourFilterOptions,
  type RoiNeighbourFilterResult,
  type RoiRefinedPeak,
} from '@/lib/roiNeighbourFilter'

import type { IdentityJob, IdentityWorkerApi } from './identity.worker'

let worker: Worker | undefined
let proxy: Comlink.Remote<IdentityWorkerApi> | undefined

function remote(): Comlink.Remote<IdentityWorkerApi> {
  if (proxy === undefined) {
    worker = new Worker(new URL('./identity.worker.ts', import.meta.url), { type: 'module' })
    proxy = Comlink.wrap<IdentityWorkerApi>(worker)
  }
  return proxy
}

/** Start the worker before the first frame that needs it. */
export function warmIdentityWorker(): void {
  void Promise.resolve(remote()).catch((cause: unknown) => {
    console.error('[identity-worker] startup failed', cause)
  })
}

/** Rebuild the worker's design cache ahead of the first frame that needs it. */
export function configureIdentityWorker(spec: IdentityDesignSpec): void {
  void Promise.resolve(remote().configure(spec)).catch((cause: unknown) => {
    console.error('[identity-worker] configure failed', cause)
  })
}

export function runIdentityInWorker(spec: IdentityDesignSpec, job: IdentityJob): Promise<IdentityRunOutput> {
  return remote().run(spec, job)
}

/**
 * Run the refined-peak neighbour filter in the existing identity worker.
 * Falls back to the main thread only if the worker call fails.
 */
export async function filterRefinedPeaksInWorker(
  peaks: readonly RoiRefinedPeak[],
  options: RoiNeighbourFilterOptions = {},
): Promise<RoiNeighbourFilterResult> {
  try {
    return await remote().filter(peaks, options)
  } catch (cause: unknown) {
    console.error('[identity-worker] peak filter failed; falling back to main thread', cause)
    return filterRefinedPeaks(peaks, options)
  }
}

export function disposeIdentityWorker(): void {
  worker?.terminate()
  worker = undefined
  proxy = undefined
}
