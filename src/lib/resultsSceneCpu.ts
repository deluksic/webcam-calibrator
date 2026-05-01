import { d } from 'typegpu'

import {
  MarkerCenterCentroidBoard,
  MAX_RESULTS_MARKER_POINTS,
  type MarkerCenterCentroidBoardRow,
} from '@/gpu/resultsVizLayouts'
import { WORLD_AXIS_HALF_LEN } from '@/lib/orbitOrthoMath'
import type { CalibrationOk } from '@/workers/calibration.worker'

export function centroidOfUpdatedPoints(ok: CalibrationOk): Float32Array {
  const pts = ok.updatedTargetPoints
  let cx = 0
  let cy = 0
  let cz = 0
  for (const p of pts) {
    cx += p.position.x
    cy += p.position.y
    cz += p.position.z
  }
  const inv = pts.length ? 1 / pts.length : 0
  return new Float32Array([cx * inv, cy * inv, cz * inv])
}

/** Half vertical ortho extent in world/board units (~axis half-length + padded point spread). */
export function orthoExtentYForPoints(ok: CalibrationOk, centroid: Float32Array): number {
  let maxH = WORLD_AXIS_HALF_LEN
  const cx = centroid[0]!
  const cy = centroid[1]!
  const cz = centroid[2]!
  for (const p of ok.updatedTargetPoints) {
    const x = p.position.x - cx
    const y = p.position.y - cy
    const z = p.position.z - cz
    const h = Math.hypot(x, y, z)
    if (h > maxH) {
      maxH = h
    }
  }
  return maxH * 1.35 + WORLD_AXIS_HALF_LEN * 0.15
}

export function markerCenterWritesForGpu(ok: CalibrationOk, centroid: Float32Array): MarkerCenterCentroidBoardRow[] {
  const cap = Math.min(ok.updatedTargetPoints.length, MAX_RESULTS_MARKER_POINTS)
  const rows: MarkerCenterCentroidBoardRow[] = []
  const cx = centroid[0]!
  const cy = centroid[1]!
  const cz = centroid[2]!
  for (let i = 0; i < cap; i++) {
    const p = ok.updatedTargetPoints[i]!.position
    rows.push(
      MarkerCenterCentroidBoard({
        positionCentroidRelativeBoardUnits: d.vec3f(p.x - cx, p.y - cy, p.z - cz),
      }),
    )
  }
  const dead = MarkerCenterCentroidBoard({
    positionCentroidRelativeBoardUnits: d.vec3f(0, 0, -1e9),
  })
  for (let i = cap; i < MAX_RESULTS_MARKER_POINTS; i++) {
    rows.push(dead)
  }
  return rows
}
