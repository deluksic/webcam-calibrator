import { decodeStressAxisStrip, decodeStressFitPerspectiveStrip } from '@/lib/decodeStressHarness'
import { rotateStripAroundCentroid, scaleStripToMaxEdgePx } from '@/tests/shared/stripGeometry'
import type { SceneSpec, StripCorners } from '@/tests/shared/types'

/** Build canonical strip for scene (before optional max-edge scaling / rotation). */
export function buildBaseStrip(spec: SceneSpec): StripCorners {
  const margin = spec.margin ?? 6
  if (spec.kind === 'axis') {
    const side = spec.axisSidePx ?? Math.min(spec.width, spec.height) - 2 * margin
    return decodeStressAxisStrip(spec.width, spec.height, margin, side)
  }
  return decodeStressFitPerspectiveStrip(spec.width, spec.height, {
    margin,
    perspectiveBoost: spec.perspectiveBoost ?? 1,
  })
}

export function buildGroundTruthStrip(spec: SceneSpec): StripCorners {
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
