import { d, std, tgpu } from 'typegpu'
import { abs, max, mul, sqrt } from 'typegpu/std'

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

export const QuadCornerSolveResult = d.struct({
  failureCode: d.u32,
  intersectionCount: d.u32,
  homography: d.mat3x3f,
  homographyOk: d.u32,
  /** TL, TR, BL, BR in image pixels — used for grid overlay rasterization. */
  corners: Corners4,
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

/**
 * `ring` is a CCW cyclic array (from fixed-slot adjacency intersections).
 * Pick geometric TL by lexicographic (y, x) min with ε ties, then walk the ring for TR, BR, BL.
 */
export const orderCornersTLTRBLBR = tgpu.fn(
  [Corners4],
  Corners4,
)((ring) => {
  'use gpu'
  let tlIdx = d.u32(0)
  let bestY = ring[0]!.y
  let bestX = ring[0]!.x
  const tlEps = d.f32(1e-4)
  for (const i of tgpu.unroll(std.range(1, MAX_EDGES_PER_LABEL))) {
    const p = ring[i]!
    const pick =
      p.y < bestY - tlEps || (abs(p.y - bestY) <= tlEps && p.x < bestX - tlEps)
    if (pick) {
      tlIdx = i
      bestY = p.y
      bestX = p.x
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
  const signedArea = quadSignedArea(corners)
  if (signedArea <= d.f32(0)) {
    return 0
  }
  if (signedArea < d.f32(QUAD_MIN_SIGNED_AREA_REL) * scale * scale) {
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
 * strip-order quarter-turn CW permutations (not a polar-ring start index).
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

/**
 * Corners from fixed CCW slot adjacency (peaks canonically sorted in findPeaks).
 * Intersect line i with line (i+1)%4, then TL,TR,BL,BR + single DLT.
 */
export const solveQuadCornersAndHomography = tgpu.fn(
  [d.arrayOf(LineNormalD, MAX_EDGES_PER_LABEL)],
  QuadCornerSolveResult,
)((lines) => {
  'use gpu'
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

  const ordered = orderCornersTLTRBLBR(corners)
  if (quadDegeneracyOk(ordered) === d.u32(0)) {
    // Collapse to zero-area — grid viz won't render a degenerate triangle strip.
    return QuadCornerSolveResult({
      failureCode: FAIL_PLAUSIBILITY,
      intersectionCount: count,
      homography: invalidGridHomography(),
      homographyOk: d.u32(0),
      corners: collapsedScreenQuad(ordered, count),
    })
  }

  const h = tryHomographyFromCorners(ordered[0]!, ordered[1]!, ordered[2]!, ordered[3]!)
  if (h.ok !== d.u32(0)) {
    // Verify the homography maps the unit-square midpoint to a point in front
    // of the camera (z > 0).  Non-positive z means a self-intersecting or
    // flipped quad that slipped past the winding check.
    const mid = mul(h.homography, d.vec3f(d.f32(0.5), d.f32(0.5), d.f32(1)))
    if (mid.z > d.f32(0)) {
      return QuadCornerSolveResult({
        failureCode: 0,
        intersectionCount: count,
        homography: h.homography,
        homographyOk: 1,
        corners: Corners4(ordered),
      })
    }
  }

  // DLT failed or homography is degenerate — return stable corners for screen-space fallback.
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
