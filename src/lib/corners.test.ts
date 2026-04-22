import { describe, it, expect } from 'vitest'

import { extractLabeledEdgePixels, findCornersFromEdges } from '@/lib/corners'

const { abs, random } = Math

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build synthetic Sobel data for a rectangular border.
// Generates gx/gy along the edges of a rectangle (minX,minY,maxX,maxY).
// gx positive on left/right edges, gy positive on top/bottom edges.
// ─────────────────────────────────────────────────────────────────────────────
function makeRectSobel(
  width: number,
  height: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): Float32Array {
  const sobelData = new Float32Array(width * height * 2)

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const idx = (y * width + x) * 2
      const onLeft = x === minX
      const onRight = x === maxX
      const onTop = y === minY
      const onBottom = y === maxY
      const onBorder = onLeft || onRight || onTop || onBottom
      if (!onBorder) {
        continue
      }

      if (onLeft) {
        sobelData[idx] = 10
        sobelData[idx + 1] = 0
      } else if (onRight) {
        sobelData[idx] = -10
        sobelData[idx + 1] = 0
      } else if (onTop) {
        sobelData[idx] = 0
        sobelData[idx + 1] = 10
      } else if (onBottom) {
        sobelData[idx] = 0
        sobelData[idx + 1] = -10
      }
    }
  }

  return sobelData
}

// Label data: all pixels inside the rect have label=1, others have label=0.
function makeRectLabel(
  width: number,
  height: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): Uint32Array {
  const labelData = new Uint32Array(width * height)
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      labelData[y * width + x] = 1
    }
  }
  return labelData
}

describe('extractLabeledEdgePixels', () => {
  it('extracts only pixels matching the region label', () => {
    const w = 20,
      h = 20
    const sobelData = makeRectSobel(w, h, 5, 5, 15, 15)
    const labelData = makeRectLabel(w, h, 5, 5, 15, 15)

    const pixels = extractLabeledEdgePixels(sobelData, labelData, w, 1, 5, 5, 15, 15)
    expect(pixels.length).toBeGreaterThan(0)

    for (const p of pixels) {
      const xi = Math.floor(p.x)
      const yi = Math.floor(p.y)
      expect(labelData[yi * w + xi]).toBe(1)
    }
  })

  it('ignores pixels with a different label', () => {
    const w = 10,
      h = 10
    const sobelData = new Float32Array(w * h * 2)
    const labelData = new Uint32Array(w * h)

    sobelData[5 * w * 2 + 0] = 10
    sobelData[5 * w * 2 + 1] = 0
    labelData[5 * w + 5] = 2 // label is 2, querying for 1

    const pixels = extractLabeledEdgePixels(sobelData, labelData, w, 1, 0, 0, 9, 9)
    expect(pixels).toHaveLength(0)
  })

  it('returns empty when Sobel magnitude is zero', () => {
    const w = 10,
      h = 10
    const sobelData = new Float32Array(w * h * 2)
    const labelData = makeRectLabel(w, h, 0, 0, 9, 9)

    const pixels = extractLabeledEdgePixels(sobelData, labelData, w, 1, 0, 0, 9, 9)
    expect(pixels).toHaveLength(0)
  })

  it('stores raw Sobel gradient for clustering', () => {
    const w = 10,
      h = 10
    const sobelData = new Float32Array(w * h * 2)
    const labelData = makeRectLabel(w, h, 0, 0, 9, 9)

    sobelData[5 * w * 2 + 0] = 10
    sobelData[5 * w * 2 + 1] = 0
    labelData[5 * w + 5] = 1

    const pixels = extractLabeledEdgePixels(sobelData, labelData, w, 1, 0, 0, 9, 9)
    expect(pixels.length).toBeGreaterThan(0)
    expect(pixels[0].gx).toBeCloseTo(10, 5)
    expect(pixels[0].gy).toBeCloseTo(0, 5)
    expect(pixels[0].magnitude).toBeCloseTo(10, 5)
  })
})

describe('findCornersFromEdges', () => {
  it('returns empty for zero edge pixels', () => {
    const w = 50,
      h = 50
    const sobelData = new Float32Array(w * h * 2)
    const labelData = new Uint32Array(w * h)

    const corners = findCornersFromEdges(sobelData, labelData, w, 1, 0, 0, 49, 49)
    expect(corners).toBeUndefined()
  })

  it('detects 4 corners for an axis-aligned rectangle', () => {
    const w = 100,
      h = 100
    const sobelData = makeRectSobel(w, h, 10, 10, 90, 90)
    const labelData = makeRectLabel(w, h, 10, 10, 90, 90)

    const corners = findCornersFromEdges(sobelData, labelData, w, 1, 10, 10, 90, 90)
    expect(corners).toBeDefined()
    expect(corners).toHaveLength(4)

    const [tl, tr, bl, br] = corners!
    expect(abs(tl.x - 10)).toBeLessThan(10)
    expect(abs(tl.y - 10)).toBeLessThan(10)
    expect(abs(tr.x - 90)).toBeLessThan(10)
    expect(abs(tr.y - 10)).toBeLessThan(10)
    expect(abs(bl.x - 10)).toBeLessThan(10)
    expect(abs(bl.y - 90)).toBeLessThan(10)
    expect(abs(br.x - 90)).toBeLessThan(10)
    expect(abs(br.y - 90)).toBeLessThan(10)
  })

  it('returns empty when there are insufficient edge pixels', () => {
    const w = 20,
      h = 20
    const sobelData = new Float32Array(w * h * 2)
    const labelData = makeRectLabel(w, h, 5, 5, 15, 15)

    // Only one edge pixel
    sobelData[5 * w * 2 + 0] = 10
    sobelData[5 * w * 2 + 1] = 0
    labelData[5 * w + 5] = 1

    const corners = findCornersFromEdges(sobelData, labelData, w, 1, 5, 5, 15, 15)
    expect(corners).toBeUndefined()
  })

  it('orders corners as TL, TR, BL, BR', () => {
    const w = 100,
      h = 100
    const sobelData = makeRectSobel(w, h, 10, 10, 90, 90)
    const labelData = makeRectLabel(w, h, 10, 10, 90, 90)

    const corners = findCornersFromEdges(sobelData, labelData, w, 1, 10, 10, 90, 90)
    expect(corners).toBeDefined()

    const [tl, tr, bl, br] = corners!
    // TL above BL and left of TR
    expect(tl.y).toBeLessThan(bl.y)
    expect(tl.x).toBeLessThan(tr.x)
    // BR below TR and right of BL
    expect(br.y).toBeGreaterThan(tr.y)
    expect(br.x).toBeGreaterThan(bl.x)
  })

  it('returns empty for random noise (no clean line structure)', () => {
    const w = 50,
      h = 50
    const sobelData = new Float32Array(w * h * 2)
    const labelData = new Uint32Array(w * h)

    for (let y = 5; y < 45; y++) {
      for (let x = 5; x < 45; x++) {
        if (random() < 0.3) {
          sobelData[(y * w + x) * 2] = random() * 10
          sobelData[(y * w + x) * 2 + 1] = random() * 10
          labelData[y * w + x] = 1
        }
      }
    }

    const corners = findCornersFromEdges(sobelData, labelData, w, 1, 5, 5, 45, 45)
    // Random noise should not produce a valid quad
    expect(corners).toBeUndefined()
  })
})
