// Contour detection: CPU region extraction + quad fitting (labels from GPU pointer-jump).

import { findCornersFromEdgesWithDebug, type CornerDebugInfo } from '@/lib/corners'
import type { Point } from '@/lib/geometry'
import type { GridResult } from '@/lib/grid'
import { buildTagGrid, decodeTagPattern } from '@/lib/grid'
import { decodeTag36h11AnyRotation, type TagPattern } from '@/lib/tag36h11'
import { hasExactlyFourElements } from '@/utils/assertArray'

import { ALLOWED_ERROR_COUNT } from './pipelines/constants'

const { min, max, floor } = Math

export const COMPONENT_LABEL_INVALID = 0xffffffff

export interface DetectedQuad {
  corners: [Point, Point, Point, Point]
  label: number
  count: number
  aspectRatio: number
  area: number
  gridCells: GridResult | undefined
  pattern: TagPattern | undefined
  hasCorners: boolean // true if detected via corner finding, false if fallback bbox
  cornerDebug: CornerDebugInfo | undefined
  /** Test / viz: random or decoded id for GPU hash + HTML label (omit = unknown/black). */
  vizTagId?: number
  /** tag36h11 id when `decodeTag36h11AnyRotation` succeeds on `pattern`. */
  decodedTagId?: number
  /** Clockwise quarter-turns (0–3) from pattern grid to canonical tag orientation. */
  decodedRotation?: number
}

/** `buildTagGrid` / AprilTag decode need TL,TR,BR,BL; homography uses TL,TR,BL,BR (triangle strip). */
function cornersForTagGrid(c: [Point, Point, Point, Point]): [Point, Point, Point, Point] {
  return [c[0], c[1], c[3], c[2]]
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU-side processing: Extract quads from labeled image
// ──────────────────────────────────────���──────────────────────────────────────

export interface RegionData {
  label: number
  rootLabel: number
  count: number
  minX: number
  minY: number
  maxX: number
  maxY: number
  pixels: [number, number][]
}

export function extractRegions(
  labelData: Uint32Array,
  width: number,
  height: number,
  _edgeData: Float32Array,
): RegionData[] {
  const regions = new Map<number, RegionData>()

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x
      const label = labelData[idx]

      if (label === undefined || label === COMPONENT_LABEL_INVALID) {
        continue
      }

      if (!regions.has(label)) {
        regions.set(label, {
          label,
          rootLabel: label, // raw pixel-index label IS the root
          count: 0,
          minX: x,
          minY: y,
          maxX: x,
          maxY: y,
          pixels: [],
        })
      }

      const region = regions.get(label)!
      region.count++
      region.minX = min(region.minX, x)
      region.minY = min(region.minY, y)
      region.maxX = max(region.maxX, x)
      region.maxY = max(region.maxY, y)

      if (region.pixels.length < 500) {
        region.pixels.push([x, y])
      }
    }
  }

  const result = [...regions.values()]
  return result
}

export function fitQuadToRegion(region: RegionData): [number, number][] | undefined {
  const { minX, minY, maxX, maxY } = region

  const boundingBoxWidth = maxX - minX
  const boundingBoxHeight = maxY - minY

  const aspectRatio = boundingBoxWidth / boundingBoxHeight
  if (aspectRatio < 0.5 || aspectRatio > 2.0) {
    return undefined
  }

  let leftEdge = maxX
  let rightEdge = minX
  let topEdge = maxY
  let bottomEdge = minY

  for (const [px, py] of region.pixels) {
    if (px - minX < leftEdge - minX) {
      leftEdge = px
    }
    if (maxX - px < maxX - rightEdge) {
      rightEdge = px
    }
    if (py - minY < topEdge - minY) {
      topEdge = py
    }
    if (maxY - py < maxY - bottomEdge) {
      bottomEdge = py
    }
  }

  return [
    [leftEdge, topEdge],
    [rightEdge, topEdge],
    [rightEdge, bottomEdge],
    [leftEdge, bottomEdge],
  ]
}

export function validateAndFilterQuads(
  regions: RegionData[],
  sobelData: Float32Array,
  labelData: Uint32Array,
  width: number,
  minArea: number = 400,
  maxArea: number = 200000,
): DetectedQuad[] {
  const quads: DetectedQuad[] = []
  const imageHeight = floor(labelData.length / width)

  for (const region of regions) {
    const w = region.maxX - region.minX
    const h = region.maxY - region.minY
    const area = w * h

    if (area < minArea) {
      continue
    }
    if (area > maxArea) {
      continue
    }

    const aspectRatio = w / h
    if (aspectRatio < 0.3 || aspectRatio > 3.5) {
      continue
    }

    const perimeter = 2 * (w + h)
    // Oblique/far tags have fewer edge pixels — relax density threshold
    const edgeDensity = region.count / perimeter
    if (edgeDensity < 0.2 || edgeDensity > 10) {
      continue
    }

    // Try to detect real corners via line intersection
    const cornerResult = findCornersFromEdgesWithDebug(
      sobelData,
      labelData,
      width,
      region.label,
      region.minX,
      region.minY,
      region.maxX,
      region.maxY,
    )
    const detectedCorners = cornerResult.corners
    const corners: [Point, Point, Point, Point] = hasExactlyFourElements(detectedCorners)
      ? detectedCorners
      : [
          { x: region.minX, y: region.minY },
          { x: region.maxX, y: region.minY },
          { x: region.minX, y: region.maxY },
          { x: region.maxX, y: region.maxY },
        ]
    const tagGrid = buildTagGrid(cornersForTagGrid(corners))
    if (!tagGrid || !tagGrid.cells || tagGrid.cells.length === 0) {
      continue
    }
    // Decode uses bilinear Sobel everywhere in the cell; do not gate on a label/mag floor mask —
    // floor(sample) often lands on flat interiors (NMS ≈ 0) while the float bilinear tap still sees edges.
    const pattern = decodeTagPattern(tagGrid, sobelData, width, undefined, imageHeight)
    const decoded = decodeTag36h11AnyRotation(pattern, ALLOWED_ERROR_COUNT)
    quads.push({
      corners,
      label: region.label,
      count: region.count,
      aspectRatio,
      area,
      gridCells: tagGrid,
      pattern,
      hasCorners: detectedCorners.length === 4,
      cornerDebug: cornerResult.debug,
      ...(decoded ? { decodedTagId: decoded.id, decodedRotation: decoded.rotation } : {}),
    })
  }

  return quads
}
