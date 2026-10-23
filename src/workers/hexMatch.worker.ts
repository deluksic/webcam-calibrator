import * as Comlink from 'comlink'

import type { HexTargetConfig } from '@/lib/hexTarget/config'
import { matchHexTargetPoints, type HexObservedPoint, type HexTargetMatch } from '@/lib/hexTarget/match'
import { createHexTargetDesign, type HexTargetModel } from '@/lib/hexTarget/target'

export type HexMatchJob = {
  points: readonly HexObservedPoint[]
  depth: number
  recovery: boolean
}

export type HexMatchWorkerApi = {
  /** Build the design-side target model ahead of the first frame that needs it. */
  configure: (config: HexTargetConfig) => void
  match: (config: HexTargetConfig, job: HexMatchJob) => HexTargetMatch
}

/**
 * Identity matching is expensive (observed graph + local ICP), so the
 * Calibrate page runs it here. `createHexTargetDesign` is cached because the
 * design-side graph and fingerprint indexes depend only on the target config.
 * The key omits presentation-only fields such as display resolution.
 */
function targetKey(config: HexTargetConfig): string {
  return [config.columns, config.rows, config.pointsPerEdge, config.colorSeed, config.identityDepth].join(':')
}

let cachedModel: HexTargetModel | undefined
let cachedKey = ''

function modelFor(config: HexTargetConfig): HexTargetModel {
  const key = targetKey(config)
  if (cachedModel === undefined || cachedKey !== key) {
    cachedModel = createHexTargetDesign(config)
    cachedKey = key
  }
  return cachedModel
}

const api: HexMatchWorkerApi = {
  configure(config) {
    modelFor(config)
  },
  match(config, job) {
    return matchHexTargetPoints(modelFor(config), job.points, {
      depth: job.depth,
      recovery: job.recovery,
    })
  },
}

Comlink.expose(api)
