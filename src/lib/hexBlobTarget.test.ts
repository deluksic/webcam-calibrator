import { describe, expect, it } from 'vitest'

import { buildHexTargetLayout } from '@/lib/hexBlobTarget'

const baseSpec = {
  widthPx: 1280,
  heightPx: 720,
  columns: 8,
  pointsPerEdge: 4,
  markerScale: 0.45,
  colorSeed: 1,
}

describe('buildHexTargetLayout', () => {
  it('produces integer, in-bounds, half-open quads', () => {
    const layout = buildHexTargetLayout(baseSpec)
    expect(layout.quads.length).toBeGreaterThan(0)
    const designIds = new Set<number>()
    for (const quad of layout.quads) {
      designIds.add(quad.designId)
      expect(Number.isInteger(quad.left)).toBe(true)
      expect(Number.isInteger(quad.top)).toBe(true)
      expect(Number.isInteger(quad.right)).toBe(true)
      expect(Number.isInteger(quad.bottom)).toBe(true)
      expect(quad.right - quad.left).toBe(layout.sidePx)
      expect(quad.bottom - quad.top).toBe(layout.sidePx)
      expect(quad.left).toBeGreaterThanOrEqual(0)
      expect(quad.top).toBeGreaterThanOrEqual(0)
      expect(quad.right).toBeLessThanOrEqual(layout.widthPx)
      expect(quad.bottom).toBeLessThanOrEqual(layout.heightPx)
    }
    expect(designIds.size).toBe(layout.quads.length)
  })

  it('leaves a gray gap between every pair of quads', () => {
    const layout = buildHexTargetLayout(baseSpec)
    for (let a = 0; a < layout.quads.length; a += 1) {
      const first = layout.quads[a]!
      for (let b = a + 1; b < layout.quads.length; b += 1) {
        const second = layout.quads[b]!
        const separated =
          first.right <= second.left ||
          second.right <= first.left ||
          first.bottom <= second.top ||
          second.bottom <= first.top
        expect(separated).toBe(true)
      }
    }
  })

  it('is deterministic for a seed and changes only colors with a new seed', () => {
    const first = buildHexTargetLayout(baseSpec)
    const again = buildHexTargetLayout(baseSpec)
    const shuffled = buildHexTargetLayout({ ...baseSpec, colorSeed: 2 })
    expect(first.quads).toEqual(again.quads)
    expect(shuffled.quads.map((quad) => [quad.left, quad.top, quad.right, quad.bottom])).toEqual(
      first.quads.map((quad) => [quad.left, quad.top, quad.right, quad.bottom]),
    )
    expect(shuffled.quads.map((quad) => quad.color)).not.toEqual(first.quads.map((quad) => quad.color))
  })

  it('keeps at least 10 percent padding on each axis', () => {
    const layout = buildHexTargetLayout(baseSpec)
    expect(layout.paddingXPx).toBeGreaterThanOrEqual(Math.round(layout.widthPx * 0.1))
    expect(layout.paddingYPx).toBeGreaterThanOrEqual(Math.round(layout.heightPx * 0.1))
  })

  it('centers the marker field vertically', () => {
    const layout = buildHexTargetLayout(baseSpec)
    const minTop = Math.min(...layout.quads.map((quad) => quad.top))
    const maxBottom = Math.max(...layout.quads.map((quad) => quad.bottom))
    expect(Math.abs((minTop + maxBottom) / 2 - layout.heightPx / 2)).toBeLessThanOrEqual(2)
  })

  it('handles odd/even canvas sizes and marker scales', () => {
    for (const size of [
      [1279, 719],
      [1280, 720],
      [1001, 777],
    ] as const) {
      for (const markerScale of [0.3, 0.45, 0.6]) {
        const layout = buildHexTargetLayout({
          ...baseSpec,
          widthPx: size[0],
          heightPx: size[1],
          markerScale,
        })
        expect(layout.quads.length).toBeGreaterThan(0)
        for (const quad of layout.quads) {
          expect(quad.left).toBeGreaterThanOrEqual(0)
          expect(quad.right).toBeLessThanOrEqual(layout.widthPx)
          expect(quad.top).toBeGreaterThanOrEqual(0)
          expect(quad.bottom).toBeLessThanOrEqual(layout.heightPx)
        }
      }
    }
  })

  it('returns an empty layout for degenerate sizes', () => {
    expect(buildHexTargetLayout({ ...baseSpec, widthPx: 0 }).quads).toEqual([])
    expect(buildHexTargetLayout({ ...baseSpec, heightPx: 0 }).quads).toEqual([])
  })
})
