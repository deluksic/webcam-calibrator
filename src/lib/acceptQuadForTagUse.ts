import type { DetectedQuad } from '@/gpu/contour'
import { DECODED_TAG_ID_DICT_MISS } from '@/gpu/pipelines/gridVizPipeline'
import { patternHasWeakOrTie } from '@/lib/tagModuleCell'

const DICT_MISS_U32 = DECODED_TAG_ID_DICT_MISS >>> 0

/**
 * Single gate for GPU tag grid, ID overlay, calibration snapshots, and reprojection.
 *
 * **Debug fallbacks (`showFallbacks`):** any quad the detector emitted (including axis-aligned
 * bbox substitutes when line intersections fail) — same spirit as the old “pass all quads”
 * grid buffer.
 *
 * **Strict (calibration / Fallbk off):** real corners, successful corner pipeline, no weak/tie
 * cells, and a decoded id or dictionary-miss sentinel for tint/labels.
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
  return typeof quad.decodedTagId === 'number' || (quad.vizTagId ?? 0) === DICT_MISS_U32
}
