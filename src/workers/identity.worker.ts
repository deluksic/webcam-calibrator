import * as Comlink from 'comlink'

import {
  createIdentityDesignBundle,
  identityDesignKey,
  runIdentityPipeline,
  type IdentityDesignBundle,
  type IdentityDesignSpec,
  type IdentityObservedPoint,
  type IdentityRunOutput,
} from '@/lib/pointIdentity/identityPipeline'

export type IdentityJob = {
  points: readonly IdentityObservedPoint[]
  depth: number
  recovery: boolean
  frameId: number
}

export type IdentityWorkerApi = {
  /**
   * Build (or confirm) the design bundle for `spec`. Called when the ROI lab's
   * design settings change so the one-off lattice/index build never lands on a
   * frame the user is waiting on.
   */
  configure: (spec: IdentityDesignSpec) => void
  run: (spec: IdentityDesignSpec, job: IdentityJob) => IdentityRunOutput
}

// The worker owns the design: it is derived from `spec` alone, so caching a
// single entry is enough (the UI only ever shows one design at a time).
let cachedDesign: IdentityDesignBundle | undefined
let cachedKey = ''

function designFor(spec: IdentityDesignSpec): IdentityDesignBundle {
  const key = identityDesignKey(spec)
  if (cachedDesign === undefined || cachedKey !== key) {
    cachedDesign = createIdentityDesignBundle(spec)
    cachedKey = key
  }
  return cachedDesign
}

const api: IdentityWorkerApi = {
  configure(spec) {
    designFor(spec)
  },
  run(spec, job) {
    return runIdentityPipeline({
      design: designFor(spec),
      points: job.points,
      depth: job.depth,
      recovery: job.recovery,
      frameId: job.frameId,
    })
  },
}

Comlink.expose(api)
