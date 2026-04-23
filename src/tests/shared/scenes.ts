import { decodeStressAxisStrip, decodeStressFitPerspectiveStrip } from '@/lib/decodeStressHarness'
import type { Corners } from '@/lib/geometry'
import { rotateStripAroundCentroid, scaleStripToMaxEdgePx } from '@/tests/shared/stripGeometry'
import type { SceneSpec } from '@/tests/shared/types'

const { min } = Math

/** Build canonical strip for scene (before optional max-edge scaling / rotation). */
export function buildBaseStrip(spec: SceneSpec): Corners {
  const margin = spec.margin ?? 6
  if (spec.kind === 'axis') {
    const side = spec.axisSidePx ?? min(spec.width, spec.height) - 2 * margin
    return decodeStressAxisStrip(spec.width, spec.height, margin, side)
  }
  return decodeStressFitPerspectiveStrip(spec.width, spec.height, {
    margin,
    perspectiveBoost: spec.perspectiveBoost ?? 1,
  })
}

export function buildGroundTruthStrip(spec: SceneSpec): Corners {
  let strip = buildBaseStrip(spec)
  const rot = spec.rotationRad ?? 0
  if (rot !== 0) {
    strip = rotateStripAroundCentroid(strip, rot)
  }
  if (spec.maxEdgePx !== undefined) {
    strip = scaleStripToMaxEdgePx(strip, spec.maxEdgePx)
  }
  return strip
}
