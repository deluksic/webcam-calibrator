import { d } from 'typegpu'

import {
  MarkerCenter,
  MAX_RESULTS_MARKER_POINTS,
  MAX_RESULTS_TAG_QUADS,
  type MarkerCenterRow,
  type TagQuadRow,
} from '@/gpu/resultsVizLayouts'
import type { CalibrationOk } from '@/workers/calibration.worker'
import type { Corners3, Point3 } from '@/lib/calibrationTypes'
import { WORLD_AXIS_HALF_LEN } from '@/lib/orbitOrthoMath'
import { tagIdPattern } from '@/lib/tag36h11'

export function* iterCalibrationCornerPositions(ok: CalibrationOk): Generator<Point3> {
  for (const t of ok.updatedTargets) {
    for (const p of t.corners) {
      yield p
    }
  }
}

export function calibrationDefinedCornerCount(ok: CalibrationOk): number {
  let n = 0
  for (const _ of iterCalibrationCornerPositions(ok)) {
    n++
  }
  return n
}

/** Half vertical ortho extent in world/board units (~axis half-length + padded spread from board origin). */
export function orthoExtentYForPoints(ok: CalibrationOk): number {
  let maxH = WORLD_AXIS_HALF_LEN
  for (const p of iterCalibrationCornerPositions(ok)) {
    const h = Math.hypot(p.x, p.y, p.z)
    if (h > maxH) {
      maxH = h
    }
  }
  return maxH * 1.35 + WORLD_AXIS_HALF_LEN * 0.15
}

/** TL, TR, BL, BR (matches `Corners3` ordering); shared between marker and tag-quad writers. */
function cornersToVec3fArray(corners: Corners3): [d.v3f, d.v3f, d.v3f, d.v3f] {
  return [
    d.vec3f(corners[0].x, corners[0].y, corners[0].z),
    d.vec3f(corners[1].x, corners[1].y, corners[1].z),
    d.vec3f(corners[2].x, corners[2].y, corners[2].z),
    d.vec3f(corners[3].x, corners[3].y, corners[3].z),
  ]
}

export function markerCenterWritesForGpu(ok: CalibrationOk): MarkerCenterRow[] {
  const rows: MarkerCenterRow[] = []
  outer: for (const t of ok.updatedTargets) {
    const cs = cornersToVec3fArray(t.corners)
    for (const c of cs) {
      if (rows.length >= MAX_RESULTS_MARKER_POINTS) {
        break outer
      }
      rows.push(MarkerCenter({ positionBoardUnits: c }))
    }
  }
  const dead = MarkerCenter({
    positionBoardUnits: d.vec3f(0, 0, -1e9),
  })
  for (let i = rows.length; i < MAX_RESULTS_MARKER_POINTS; i++) {
    rows.push(dead)
  }
  return rows
}

/** Pack the row-major 6×6 0/1 interior pattern (36 bits) into `vec2u` (lo: bits 0..31, hi: 32..35). */
function packTagPattern(tagId: number): d.v2u {
  const pattern = tagIdPattern(tagId)
  let lo = 0
  let hi = 0
  for (let i = 0; i < 36; i++) {
    if (pattern[i] !== 1) {
      continue
    }
    if (i < 32) {
      lo = (lo | (1 << i)) >>> 0
    } else {
      hi = (hi | (1 << (i - 32))) >>> 0
    }
  }
  return d.vec2u(lo, hi)
}

/** One row per calibrated tag: board-space corners + packed bit pattern. */
export function tagQuadWritesForGpu(ok: CalibrationOk): TagQuadRow[] {
  const rows: TagQuadRow[] = []
  for (const t of ok.updatedTargets) {
    if (rows.length >= MAX_RESULTS_TAG_QUADS) {
      break
    }
    rows.push({
      corners: cornersToVec3fArray(t.corners),
      packedPattern: packTagPattern(t.tagId),
    })
  }
  return rows
}
