import type { DetectedQuad } from '@/gpu/detectedQuad'
import { patternHasWeakOrTie } from '@/lib/tagModuleCell'

/**
 * Single gate for GPU tag grid, ID overlay, calibration snapshots, and reprojection.
 *
 * **Permissive (`showFallbacks`):** any quad returned by the GPU path.
 *
 * **Strict (calibration):** successful corners, no weak/tie pattern cells, decoded tag36h11 id.
 */
export function acceptQuadForTagUse(quad: DetectedQuad, showFallbacks: boolean): boolean {
  if (!quad?.corners?.length) {
    return false
  }
  if (showFallbacks) {
    return true
  }
  if (!quad.hasCorners || quad.cornerDebug?.failureCode !== 0) {
    return false
  }
  if (patternHasWeakOrTie(quad.pattern)) {
    return false
  }
  return typeof quad.decodedTagId === 'number'
}
