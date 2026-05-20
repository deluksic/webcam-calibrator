import { d, std, tgpu } from 'typegpu'
import { abs, atan2, max, sqrt } from 'typegpu/std'

import { MAX_EDGES_PER_LABEL } from '@/gpu/lineFitThresholds'
import { QUAD_MIN_EDGE_PX, QUAD_MIN_SIGNED_AREA_REL } from '@/gpu/lineFitThresholds'
import { EdgeLineEntry } from '@/gpu/pipelines/edgeLineFitPipeline'
import { invalidGridHomography, tryHomographyFromCorners } from '@/gpu/shaders/homographyDlt'
import { LineNormalD, lineIntersectNormal, lineNormalFromEdge } from '@/gpu/shaders/lineIntersect'

/** Bitmask: matches CPU `FAIL_PLAUSIBILITY`. */
export const FAIL_PLAUSIBILITY = d.u32(1 << 3)
/** Bitmask: matches CPU `FAIL_NO_INTERSECTIONS`. */
export const FAIL_NO_INTERSECTIONS = d.u32(1 << 4)

export const Corners4 = d.arrayOf(d.vec2f, MAX_EDGES_PER_LABEL)
export const U32x4 = d.arrayOf(d.u32, MAX_EDGES_PER_LABEL)

export const QuadCornerSolveResult = d.struct({
  failureCode: d.u32,
  intersectionCount: d.u32,
  homography: d.mat3x3f,
  homographyOk: d.u32,
  /** TL, TR, BL, BR in image pixels — used for grid overlay rasterization. */
  corners: Corners4,
})

/** Sort edge slot indices 0..3 by increasing TLS normal angle. */
export const sortEdgeSlotsByLineNormal = tgpu.fn(
  [d.arrayOf(LineNormalD, MAX_EDGES_PER_LABEL), U32x4],
  U32x4,
)((lines, slotsIn) => {
  'use gpu'
  const slots = U32x4()
  for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    slots[i] = slotsIn[i]!
  }

  for (const _ of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    for (const j of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL - 1))) {
      const sj = slots[j]!
      const sjp1 = slots[j + d.u32(1)]!
      const la = lines[sj]!
      const lb = lines[sjp1]!
      const aj = atan2(la.ny, la.nx)
      const ajp1 = atan2(lb.ny, lb.nx)
      if (aj > ajp1) {
        slots[j] = sjp1
        slots[j + d.u32(1)] = sj
      }
    }
  }
  return U32x4(slots)
})

export const AdjacentCornersResult = d.struct({
  corners: Corners4,
  count: d.u32,
})

/** Intersect adjacent lines in sorted normal order → up to four corners. */
export const cornersFromAdjacentLines = tgpu.fn(
  [d.arrayOf(LineNormalD, MAX_EDGES_PER_LABEL), U32x4],
  AdjacentCornersResult,
)((lines, sortedSlots) => {
  'use gpu'
  const corners = Corners4()
  let count = d.u32(0)
  for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    const i1 = (i + d.u32(1)) % d.u32(MAX_EDGES_PER_LABEL)
    const la = lines[sortedSlots[i]!]!
    const lb = lines[sortedSlots[i1]!]!
    const hit = lineIntersectNormal(la.nx, la.ny, la.d, lb.nx, lb.ny, lb.d)
    corners[i] = d.vec2f(hit.point)
    if (hit.ok !== d.u32(0)) {
      count = count + d.u32(1)
    }
  }
  return AdjacentCornersResult({ corners: corners, count: count })
})

function quadSignedArea(corners: d.Infer<typeof Corners4>) {
  'use gpu'
  // corners is in strip order (TL, TR, BL, BR). Convert to cyclic order (TL, TR, BR, BL)
  // for the shoelace formula.
  const cyclicIdx = [d.u32(0), d.u32(1), d.u32(3), d.u32(2)] as const
  let area = d.f32(0)
  for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    const i1 = (i + d.u32(1)) % d.u32(MAX_EDGES_PER_LABEL)
    const p0 = corners[cyclicIdx[i]!]!
    const p1 = corners[cyclicIdx[i1]!]!
    area = area + p0.x * p1.y - p1.x * p0.y
  }
  return area * d.f32(0.5)
}

/** Sort corners by polar angle around centroid (cyclic order only). */
export const sortCornersByPolarAngle = tgpu.fn(
  [Corners4],
  Corners4,
)((cornersIn) => {
  'use gpu'
  let cx = d.f32(0)
  let cy = d.f32(0)
  for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    cx = cx + cornersIn[i]!.x
    cy = cy + cornersIn[i]!.y
  }
  cx = cx / d.f32(MAX_EDGES_PER_LABEL)
  cy = cy / d.f32(MAX_EDGES_PER_LABEL)

  const slots = U32x4()
  for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    slots[i] = i
  }

  for (const _ of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    for (const j of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL - 1))) {
      const sj = slots[j]!
      const sjp1 = slots[j + d.u32(1)]!
      const pj = cornersIn[sj]!
      const pjp1 = cornersIn[sjp1]!
      const aj = atan2(pj.y - cy, pj.x - cx)
      const ajp1 = atan2(pjp1.y - cy, pjp1.x - cx)
      if (aj > ajp1) {
        slots[j] = sjp1
        slots[j + d.u32(1)] = sj
      }
    }
  }

  const out = Corners4()
  for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    out[i] = d.vec2f(cornersIn[slots[i]!]!)
  }
  return out
})

/**
 * `ring` is a CCW cyclic array (from fixed-slot adjacency intersections).
 * Pick geometric TL by lexicographic (y, x) min, then walk the ring for TR, BR, BL.
 */
export const orderCornersTLTRBLBR = tgpu.fn(
  [Corners4],
  Corners4,
)((ring) => {
  'use gpu'
  let tlIdx = d.u32(0)
  for (const i of tgpu.unroll(std.range(1, MAX_EDGES_PER_LABEL))) {
    const p = ring[i]!
    const t = ring[tlIdx]!
    const pick = p.y < t.y || (p.y === t.y && p.x < t.x)
    if (pick) {
      tlIdx = i
    }
  }

  const tl = ring[tlIdx]!
  const tr = ring[(tlIdx + d.u32(1)) % d.u32(MAX_EDGES_PER_LABEL)]!
  const br = ring[(tlIdx + d.u32(2)) % d.u32(MAX_EDGES_PER_LABEL)]!
  const bl = ring[(tlIdx + d.u32(3)) % d.u32(MAX_EDGES_PER_LABEL)]!

  const strip = Corners4()
  strip[d.u32(0)] = d.vec2f(tl)
  strip[d.u32(1)] = d.vec2f(tr)
  strip[d.u32(2)] = d.vec2f(bl)
  strip[d.u32(3)] = d.vec2f(br)
  return strip
})

export const quadDegeneracyOk = tgpu.fn(
  [Corners4],
  d.u32,
)((corners) => {
  'use gpu'
  let scale = d.f32(0)
  for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    const p = corners[i]!
    scale = max(scale, abs(p.x))
    scale = max(scale, abs(p.y))
  }
  const area = abs(quadSignedArea(corners))
  if (area < d.f32(QUAD_MIN_SIGNED_AREA_REL) * scale * scale) {
    return 0
  }

  const cyclicIdx = [d.u32(0), d.u32(1), d.u32(3), d.u32(2)] as const
  for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    const i1 = (i + d.u32(1)) % d.u32(MAX_EDGES_PER_LABEL)
    const p0 = corners[cyclicIdx[i]!]!
    const p1 = corners[cyclicIdx[i1]!]!
    const dx = p1.x - p0.x
    const dy = p1.y - p0.y
    if (sqrt(dx * dx + dy * dy) < d.f32(QUAD_MIN_EDGE_PX)) {
      return 0
    }
  }
  return 1
})

const collapsedScreenQuad = tgpu.fn(
  [Corners4, d.u32],
  Corners4,
)((raw, count) => {
  'use gpu'
  const out = Corners4()
  let anchor = d.vec2f(0, 0)
  if (count > d.u32(0)) {
    anchor = d.vec2f(raw[0]!)
  }
  for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    out[i] = d.vec2f(anchor)
  }
  return out
})

/**
 * Rotate triangle-strip corners (TL, TR, BL, BR) by `k` quarter-turns CW — same permutations as
 * Strip-order quarter-turn CW permutations (not a polar-ring start index).
 */
export const rotateStripCorners = tgpu.fn(
  [Corners4, d.u32],
  Corners4,
)((strip, k) => {
  'use gpu'
  const r = k & d.u32(3)
  const out = Corners4()
  if (r === d.u32(0)) {
    out[d.u32(0)] = d.vec2f(strip[d.u32(0)]!)
    out[d.u32(1)] = d.vec2f(strip[d.u32(1)]!)
    out[d.u32(2)] = d.vec2f(strip[d.u32(2)]!)
    out[d.u32(3)] = d.vec2f(strip[d.u32(3)]!)
  } else if (r === d.u32(1)) {
    out[d.u32(0)] = d.vec2f(strip[d.u32(2)]!)
    out[d.u32(1)] = d.vec2f(strip[d.u32(0)]!)
    out[d.u32(2)] = d.vec2f(strip[d.u32(3)]!)
    out[d.u32(3)] = d.vec2f(strip[d.u32(1)]!)
  } else if (r === d.u32(2)) {
    out[d.u32(0)] = d.vec2f(strip[d.u32(3)]!)
    out[d.u32(1)] = d.vec2f(strip[d.u32(2)]!)
    out[d.u32(2)] = d.vec2f(strip[d.u32(1)]!)
    out[d.u32(3)] = d.vec2f(strip[d.u32(0)]!)
  } else {
    out[d.u32(0)] = d.vec2f(strip[d.u32(1)]!)
    out[d.u32(1)] = d.vec2f(strip[d.u32(3)]!)
    out[d.u32(2)] = d.vec2f(strip[d.u32(0)]!)
    out[d.u32(3)] = d.vec2f(strip[d.u32(2)]!)
  }
  return out
})

/** Rotate cyclic `ring`: vertex `start` is TL; same TR,BR,BL walk as {@link orderCornersTLTRBLBR}. */
export const cornersStripFromCwRingStart = tgpu.fn(
  [Corners4, d.u32],
  Corners4,
)((cw, start) => {
  'use gpu'
  const tl = cw[start]!
  const tr = cw[(start + d.u32(1)) % d.u32(MAX_EDGES_PER_LABEL)]!
  const br = cw[(start + d.u32(2)) % d.u32(MAX_EDGES_PER_LABEL)]!
  const bl = cw[(start + d.u32(3)) % d.u32(MAX_EDGES_PER_LABEL)]!
  const out = Corners4()
  out[d.u32(0)] = d.vec2f(tl)
  out[d.u32(1)] = d.vec2f(tr)
  out[d.u32(2)] = d.vec2f(bl)
  out[d.u32(3)] = d.vec2f(br)
  return out
})

/**
 * Corners from fixed CCW slot adjacency (peaks were canonically sorted in findPeaks).
 * Lines are in CCW circular order — slot i and (i+1)%4 are adjacent sides.
 * Corner order from intersections: [i∩(i+1)] is a CCW ring.
 *
 * Produces TL,TR,BL,BR strip and one DLT homography. No normal sorting, no polar sort,
 * no ring-start / BL-BR guesses — edge order is already authoritative.
 */
export const solveQuadCornersAndHomography = tgpu.fn(
  [d.arrayOf(LineNormalD, MAX_EDGES_PER_LABEL)],
  QuadCornerSolveResult,
)((lines) => {
  'use gpu'
  // Fixed adjacency intersections — lines are already in CCW circular order.
  const corners = Corners4()
  let count = d.u32(0)
  for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    const i1 = (i + d.u32(1)) % d.u32(MAX_EDGES_PER_LABEL)
    const la = lines[i]!
    const lb = lines[i1]!
    const hit = lineIntersectNormal(la.nx, la.ny, la.d, lb.nx, lb.ny, lb.d)
    corners[i] = d.vec2f(hit.point)
    if (hit.ok !== d.u32(0)) {
      count = count + d.u32(1)
    }
  }

  if (count !== d.u32(MAX_EDGES_PER_LABEL)) {
    return QuadCornerSolveResult({
      failureCode: FAIL_NO_INTERSECTIONS,
      intersectionCount: count,
      homography: invalidGridHomography(),
      homographyOk: d.u32(0),
      corners: collapsedScreenQuad(corners, count),
    })
  }

  // corners are already in CCW ring order (fixed adjacency).  Convert to strip.
  const ordered = orderCornersTLTRBLBR(corners)
  if (quadDegeneracyOk(ordered) === d.u32(0)) {
    return QuadCornerSolveResult({
      failureCode: FAIL_PLAUSIBILITY,
      intersectionCount: count,
      homography: invalidGridHomography(),
      homographyOk: d.u32(0),
      corners: Corners4(ordered),
    })
  }

  // Single DLT — strip order is authoritative, no guesswork.
  const h = tryHomographyFromCorners(ordered[0]!, ordered[1]!, ordered[2]!, ordered[3]!)
  if (h.ok !== d.u32(0)) {
    return QuadCornerSolveResult({
      failureCode: 0,
      intersectionCount: count,
      homography: h.homography,
      homographyOk: 1,
      corners: Corners4(ordered),
    })
  }

  // DLT pivot failed — return stable corners for screen-space fallback.
  return QuadCornerSolveResult({
    failureCode: 0,
    intersectionCount: count,
    homography: invalidGridHomography(),
    homographyOk: 0,
    corners: Corners4(ordered),
  })
})

export const linesFromEdgeEntries = tgpu.fn(
  [d.arrayOf(EdgeLineEntry, MAX_EDGES_PER_LABEL)],
  d.arrayOf(LineNormalD, MAX_EDGES_PER_LABEL),
)((entries) => {
  'use gpu'
  const lines = d.arrayOf(LineNormalD, MAX_EDGES_PER_LABEL)()
  for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    lines[i] = lineNormalFromEdge(entries[i]!)
  }
  return lines
})
