// Contour detection: CPU region extraction + quad fitting (labels from GPU pointer-jump).

import { DECODED_TAG_ID_DICT_MISS } from '@/gpu/pipelines/gridVizPipeline'
import { findCornersFromEdgesWithDebug, type CornerDebugInfo, rotateRing } from '@/lib/corners'
import type { Corners } from '@/lib/geometry'
import { decodeTagPatternWithVoteMaps, votePatternAcceptable } from '@/lib/grid'
import { patternIsFullyBinary } from '@/lib/tagModuleCell'
import {
  canonicalizeBinaryPatternMinCode,
  customTagIdFromCanonicalCode,
  decodeTag36h11AnyRotation,
  matchCustomCodewordsAnyRotation,
  type TagPattern,
} from '@/lib/tag36h11'

const { min, max, floor } = Math

/** Max Hamming errors allowed when matching a 6×6 tag pattern to the dictionary. */
export const ALLOWED_ERROR_COUNT = 3

export const COMPONENT_LABEL_INVALID = 0xffffffff

export type DecodedTagKind = 'tag36h11' | 'custom'

export interface QuadDecodeOptions {
  /** True once a target layout exists for the calibration session. */
  layoutEstablished?: boolean
  /** Canonical 36-bit codewords for session custom tags (from layout). */
  sessionCustomCodewords?: bigint[]
}

export interface DetectedQuad {
  corners: Corners
  label: number
  count: number
  aspectRatio: number
  area: number
  pattern: TagPattern | undefined
  hasCorners: boolean // true if detected via corner finding, false if fallback bbox
  cornerDebug: CornerDebugInfo | undefined
  /** Test / viz: random or decoded id for GPU hash + HTML label (omit = unknown/black). */
  vizTagId?: number
  /** Decoded tag id (tag36h11 or custom range). */
  decodedTagId?: number
  /** Clockwise quarter-turns (0–3) from pattern grid to canonical tag orientation. */
  decodedRotation?: number
  /** Source of {@link decodedTagId} when set. */
  decodedTagKind?: DecodedTagKind
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU-side processing: Extract quads from labeled image
// ─────────────────────────────────────────────────────────────────────────────

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
  minArea: number,
  maxArea: number,
  decodeOptions?: QuadDecodeOptions,
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
    let { corners, debug } = findCornersFromEdgesWithDebug(
      sobelData,
      labelData,
      width,
      region.label,
      region.minX,
      region.minY,
      region.maxX,
      region.maxY,
    )

    if (corners === undefined) {
      quads.push({
        corners: [
          { x: region.minX, y: region.minY },
          { x: region.maxX, y: region.minY },
          { x: region.minX, y: region.maxY },
          { x: region.maxX, y: region.maxY },
        ],
        label: region.label,
        count: region.count,
        aspectRatio,
        area,
        pattern: undefined,
        hasCorners: true,
        cornerDebug: debug,
      })
      continue
    }

    const { pattern } = decodeTagPatternWithVoteMaps(corners, sobelData, width, undefined, imageHeight)
    if (!votePatternAcceptable(pattern)) {
      continue
    }

    let decodedTagId: number | undefined
    let decodedRotation: number | undefined
    let decodedTagKind: DecodedTagKind | undefined

    const t36 = decodeTag36h11AnyRotation(pattern, ALLOWED_ERROR_COUNT)
    if (t36) {
      corners = rotateRing(corners, t36.rotation)
      decodedTagId = t36.id
      decodedRotation = t36.rotation
      decodedTagKind = 'tag36h11'
    } else if (patternIsFullyBinary(pattern)) {
      const canon = canonicalizeBinaryPatternMinCode(pattern)
      if (canon) {
        corners = rotateRing(corners, canon.rotation)
        decodedTagId = customTagIdFromCanonicalCode(canon.code)
        decodedRotation = canon.rotation
        decodedTagKind = 'custom'
      }
    } else if (
      decodeOptions?.layoutEstablished &&
      decodeOptions.sessionCustomCodewords &&
      decodeOptions.sessionCustomCodewords.length > 0
    ) {
      const custom = matchCustomCodewordsAnyRotation(pattern, decodeOptions.sessionCustomCodewords, ALLOWED_ERROR_COUNT)
      if (custom) {
        corners = rotateRing(corners, custom.rotation)
        decodedTagId = custom.tagId
        decodedRotation = custom.rotation
        decodedTagKind = 'custom'
      }
    }

    quads.push({
      corners,
      label: region.label,
      count: region.count,
      aspectRatio,
      area,
      pattern,
      hasCorners: true,
      cornerDebug: debug,
      ...(decodedTagId !== undefined
        ? { decodedTagId, decodedRotation, decodedTagKind }
        : { vizTagId: DECODED_TAG_ID_DICT_MISS >>> 0 }),
    })
  }

  return quads
}
